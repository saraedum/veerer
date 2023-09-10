import compress_pickle
import click
import os
import asyncio
from contextlib import contextmanager


def dumps(*args, **kwargs):
    return compress_pickle.dumps(*args, compression="bz2", **kwargs)


def loads(*args, **kwargs):
    return compress_pickle.loads(*args, compression="bz2", **kwargs)


def geometric_neighbors(vtd, backend):
    r"""
    Return the pair ``(vt, list_of_neighbors)`` that consists of the input
    triangulation ``vt`` together with the list ``list_of_neighbors`` of
    neighbors in the flip graph.
    """
    vt = loads(vtd)
    ans = []
    for edges, col in vt.geometric_flips(backend=backend):
        new_vt = vt.copy(mutable=True)
        for e in edges:
            new_vt.flip(e, col, check=False)
        new_vt.set_canonical_labels()
        new_vt.set_immutable()
        ans.append(dumps(new_vt))
    return vtd, ans


def md5(vt):
    import hashlib
    return hashlib.md5(vt).digest()


class ExplorerStats:
    def __init__(self, explored_vertices, unexplored_vertices):
        self._vertices_unscheduled = 0
        self._vertices_scheduled = 0
        self._vertices_explored = explored_vertices
        self._vertices_unexplored = unexplored_vertices
        self._batches_scheduled = 0
        self._batches_processed = 0
        self._batches_replicated = 0
        self._batches_unreplicated = 0

        from rich.progress import Progress, TextColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn
        self._progress = Progress(TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True, refresh_per_second=2)
        self._task_exploring = self._progress.add_task("exploring vertices", visible=False)
        self._task_batches = self._progress.add_task("computing neighborhoods (batches)", visible=False)
        self._task_consuming = self._progress.add_task("processing neighborhoods", visible=False)
        self._task_scheduling = self._progress.add_task("not scheduling computations", visible=False)
        self._task_scattering = self._progress.add_task("scattering vertices", visible=False)
        self._task_replicating = self._progress.add_task("replicating workload", visible=False)

    @contextmanager
    def __call__(self):
        self._progress.__enter__()
        try:
            self._update_exploring()
            self._progress.update(self._task_exploring, visible=True)
            self._progress.update(self._task_batches, visible=True)
            yield
        finally:
            self._progress.__exit__(None, None, None)

    def enqueue_for_scheduling(self, vertex):
        self._vertices_unscheduled += 1
        self._vertices_unexplored += 1
        self._update_exploring()
        self._update_scheduling()

    @contextmanager
    def scattering(self, batches):
        self._progress.reset(self._task_scattering, visible=True, total=sum(len(batch) for batch in batches))
        yield
        self._progress.update(self._task_scattering, visible=False)

    @contextmanager
    def scheduling(self, batches):
        self._progress.update(self._task_scheduling, description="scheduling computations")
        vertices = sum(len(batch) for batch in batches)
        yield
        self._vertices_unscheduled -= vertices
        self._vertices_scheduled += vertices
        if self._vertices_unscheduled == 0:
            self._vertices_scheduled = 0
        self._batches_scheduled += len(batches)
        self._update_scheduling()
        self._update_batches()
        self._progress.update(self._task_scheduling, description="not scheduling computations")

    @contextmanager
    def replicating(self, batches):
        if self._batches_unreplicated == 0:
            self._progress.reset(self._task_replicating, visible=True)
        self._batches_unreplicated += len(batches)
        self._progress.update(self._task_replicating, total=self._batches_replicated + self._batches_unreplicated)
        yield
        self._progress.update(self._task_replicating, advance=len(batches))
        self._batches_replicated += len(batches)
        self._batches_unreplicated -= len(batches)
        if self._batches_unreplicated == 0:
            self._batches_unreplicated = 0
            self._progress.update(self._task_replicating, visible=False)

    @contextmanager
    def consuming(self, batches):
        total = len(batches)
        self._batches_processed += total
        self._update_batches()
        self._progress.reset(self._task_consuming, visible=True, total=total)

        def expand_batch(vertices):
            nonlocal total
            total += (len(vertices) - 1)
            self._progress.update(self._task_consuming, total=total)

        def advance():
            self._vertices_explored += 1
            self._vertices_unexplored -= 1
            self._progress.update(self._task_consuming, advance=1)
            self._update_exploring()

        yield (expand_batch, advance)
        self._progress.update(self._task_consuming, visible=False, total=0, completed=0)

    def _update_scheduling(self):
        if self._vertices_unscheduled == 0:
            self._progress.update(self._task_scheduling, completed=0, total=0, visible=False)
            return

        if self._vertices_scheduled == 0:
            self._progress.reset(self._task_scheduling, total=self._vertices_unscheduled, visible=True)
            return

        self._progress.update(self._task_scheduling, total=self._vertices_unscheduled + self._vertices_scheduled, completed=self._vertices_scheduled)

    def _update_exploring(self):
        self._progress.update(self._task_exploring, total=self._vertices_explored + self._vertices_unexplored, completed=self._vertices_explored)

    def _update_batches(self):
        self._progress.update(self._task_batches, total=self._batches_scheduled, completed=self._batches_processed)


class Explorer:
    def __init__(self, graph, seen, pool, threads, backend='ppl'):
        self._to_be_scheduled = []
        self._pending_batches = []
        self._replicators = set()
        self._graph = graph
        self._seen = seen
        self._pool = pool
        self._threads = threads
        self._backend = backend
        self._stats = ExplorerStats(explored_vertices=len(graph), unexplored_vertices=len(seen) - len(graph))
        self._scheduler_wakeup = asyncio.Future()

    async def explore(self, roots):
        for root in roots:
            self._register_node(root)
        assert len(self._to_be_scheduled) == len(roots)

        with self._stats():
            _scheduler = asyncio.create_task(self._schedule())
            _consumer = asyncio.create_task(self._consume())

            completed, pending = await asyncio.wait([_scheduler, _consumer], return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                await task
            for task in pending:
                task.cancel()

    def _schedule_nbatches(self):
        ntasks = len(self._to_be_scheduled)

        if ntasks == 0:
            return "WAIT"

        # We assume that everything in _pending_batches is actually being computed currently.
        # So _threads many batches are being computed (but presumably almost
        # done) and another _threads many are going to be computed shortly.
        # We aim for there to be a buffer of another 3 threads many batches that can be picked up by workers.
        TARGET_BATCHES = 5 * self._threads
        batch_count_from_TARGET_BATCHES = max(1, TARGET_BATCHES - len(self._pending_batches))

        # We do not want batches to become too big as this causes delays in
        # network communication. But we also do not want them to be too small
        # as there is an overhead in spawning a task.
        TARGET_BATCH_SIZE = 32
        batch_count_from_TARGET_BATCH_SIZE = max(1, ntasks // TARGET_BATCH_SIZE)

        if batch_count_from_TARGET_BATCHES == 1 and batch_count_from_TARGET_BATCH_SIZE < self._threads // 2:
            # There's still enough pending batches in the queue. Wait a bit for more work to hand to accumulate.
            return "WAIT"

        return min(max(batch_count_from_TARGET_BATCHES, batch_count_from_TARGET_BATCH_SIZE), ntasks)

    async def _schedule(self):
        while True:
            nbatches = self._schedule_nbatches()
            if nbatches == "WAIT":
                self._scheduler_wakeup = asyncio.Future()
                await self._scheduler_wakeup
                continue

            to_be_scheduled = [(arg,) for arg in self._to_be_scheduled]

            batches = [to_be_scheduled[offset::nbatches] for offset in range(nbatches)]
            assert all(batch for batch in batches)

            with self._stats.scheduling(batches):
                with self._stats.scattering(batches):
                    batches = await self._pool.scatter(batches)

                self._to_be_scheduled = self._to_be_scheduled[len(to_be_scheduled):]

                from veerer.worker import Batched
                from veerer.veering_triangulations_dask_futures import geometric_neighbors
                self._pending_batches.extend(self._pool.map(Batched(geometric_neighbors, backend=self._backend), batches))

                async def replicate(batches):
                    with self._stats.replicating(batches):
                        await self._pool.replicate(batches, 2)

                replicator = asyncio.create_task(replicate(batches))
                self._replicators.add(replicator)
                replicator.add_done_callback(self._replicators.discard)

    async def _consume(self):
        while True:
            while self._pending_batches:
                completed, pending = await asyncio.wait(self._pending_batches[:self._threads], return_when=asyncio.FIRST_COMPLETED)
                self._pending_batches = list(pending) + self._pending_batches[len(completed) + len(pending):]

                completed = await self._pool.gather(completed)

                with self._stats.consuming(completed) as (expand_batch, advance):
                    for batch in completed:
                        batch = batch.result()
                        expand_batch(batch)
                        for result in batch:
                            vt, neighbors = result
                            for neighbor in neighbors:
                                self._register_node(neighbor)
                            self._graph[vt] = b"".join(md5(neighbor) for neighbor in neighbors)
                            advance()
                        await asyncio.sleep(0)

            if not self._to_be_scheduled:
                return

            await asyncio.sleep(0)

    def _register_node(self, vt):
        if not self._scheduler_wakeup.done():
            self._scheduler_wakeup.set_result("progress")

        key = md5(vt)
        if key in self._seen:
            return
        self._seen.add(key)
        self._to_be_scheduled.append(vt)
        self._stats.enqueue_for_scheduling(vt)


def loose_ends(db):
    seen = set()
    for pickle in db.keys():
        seen.add(md5(pickle))

    print(f"{len(seen)} cells have been explored in the previous run")

    loose_ends_md5 = set()
    for value in db.values():
        assert len(value) % (128 // 8) == 0
        while value:
            loose_ends_md5.add(value[:128 // 8])
            value = value[128 // 8:]

    for explored in seen:
        if explored in loose_ends_md5:
            loose_ends_md5.remove(explored)

    print(f"{len(loose_ends_md5)} cells had been detected but not explored")

    # We still have to run the exploration for all the nodes that we identified
    # as loose ends. However, we do not have their pickles only the pickles of
    # the nodes that lead to their exploration.
    loose_ends = set()
    for key, value in db.items():
        assert len(value) % (128 // 8) == 0
        while value:
            _md5 = value[:128 // 8]
            if _md5 in loose_ends_md5:
                loose_ends_md5.remove(_md5)
                loose_ends.add(key)
                if md5(key) in seen:
                    seen.remove(md5(key))
            value = value[128 // 8:]

    assert not loose_ends_md5, f"{len(loose_ends_md5)} had no pickle registered that lead to their discovery"

    print(f"Need to explore from {len(loose_ends)} root cells to continue.")

    return list(loose_ends), seen


@click.command()
@click.option('--recover/--no-recover', default=False, help='Continue from a previous aborted run.')
@click.option('--threads', default=os.cpu_count(), help='The number of simultaneous hyperthreads for statistical purposes.')
@click.option('--scheduler', default=None, help='The scheduler file to use, if not specified, the main program will serve as the scheduler.')
@click.option('--database', default='/tmp/ruth.cache', help='The path to the database to store the graph for --recover and later analysis.')
@click.option('--stratum-component', default=0, type=int)
@click.option('--backend', default='ppl')
def main(recover, threads, scheduler, database, stratum_component, backend):
    async def main():
        nonlocal stratum_component

        import dask.config
        dask.config.set({'distributed.worker.daemon': False})
        import dask.distributed
        pool = await dask.distributed.Client(scheduler_file=scheduler, direct_to_workers=True, connection_limit=2**16, asynchronous=True, n_workers=8, nthreads=1)

        from veerer import VeeringTriangulation
        from surface_dynamics import AbelianStrata, QuadraticStrata

        if stratum_component == 3413:
            from veerer.linear_family import VeeringTriangulationLinearFamilies

            print('Computing geometric veering of the (3, 4, 13) triangle')
            vt = VeeringTriangulationLinearFamilies.triangle_3_4_13_unfolding_orbit_closure().copy(mutable=True)
        else:
            stratum_component = ([C for d in range(6, 9) for H in AbelianStrata(dimension=d) for C in H.components()] + [C for d in range(3, 9) for Q in QuadraticStrata(dimension=d, nb_poles=0) for C in Q.components()])[stratum_component]

            print('Computing geometric veering triangulations in %s' % stratum_component)
            vt = VeeringTriangulation.from_stratum(stratum_component).copy(mutable=True)
            while not vt.is_geometric():
                print(".", end="")
                vt.random_forward_flip()
        vt.set_canonical_labels()
        vt.set_immutable()

        roots = [dumps(vt)]
        seen = set()

        import dbm
        if recover:
            with dbm.open(database, 'r') as graph:
                roots, seen = loose_ends(graph)

            if not roots and not seen:
                roots = [dumps(vt)]

        with dbm.open(database, 'c' if recover else 'n') as graph:
            import datetime
            t0 = datetime.datetime.now()

            await Explorer(graph=graph, seen=seen, pool=pool, backend=backend, threads=threads).explore(roots=roots)
            t1 = datetime.datetime.now()
            elapsed = t1 - t0
            print(f'{len(graph)} triangulations computed in {elapsed * threads} CPU time; {elapsed} wall time')
        await pool.close()

    asyncio.run(main())


if __name__ == '__main__':
    main()
