import compress_pickle
import click
import os
import asyncio


BATCH_LEN = 48


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


async def explore(roots, pool, threads, graph, seen=None, completed=0, backend='ppl'):
    if seen is None:
        seen = set()

    def is_new(vt):
        key = md5(vt)
        if key in seen:
            return False
        seen.add(key)
        return True

    assert all(is_new(vt) for vt in roots)

    from dask.distributed import as_completed
    jobs = as_completed([], with_results=True)
    jobs = []

    async def enqueue(tasks):
        nonlocal submitted_jobs, progress

        if not tasks:
            return

        progress.reset(task_sending)
        progress.update(task_sending, completed=0, total=len(tasks), visible=True)

        # We want there to always be 2 * threads jobs in the queue.
        nbatches_from_threads = max(1, 2 * threads - (submitted_jobs - finished_jobs))
        # We do not want to ship any tasks that contain more than BATCH_LEN jobs.
        nbatches_from_package_size = max(1, len(tasks) // BATCH_LEN)

        nbatches = max(nbatches_from_threads, nbatches_from_package_size)
        nbatches = min(len(tasks), nbatches)

        batches = [tasks[offset::nbatches] for offset in range(nbatches)]
        batches = await pool.scatter(batches)
        await pool.replicate(batches, n=2)

        from veerer.worker import Batched
        from veerer.veering_triangulations_dask_futures import geometric_neighbors
        jobs.extend(pool.map(Batched(geometric_neighbors, backend=backend), batches))

        submitted_jobs += len(batches)
        progress.update(task_jobs, total=submitted_jobs)
        progress.update(task_sending, visible=False)

    submitted_jobs = 1
    finished_jobs = 0

    from rich.progress import Progress, TextColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn
    with Progress(TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True, refresh_per_second=1) as progress:
        task_exploring = progress.add_task("exploring graph", completed=completed, total=len(seen), visible=True)
        task_jobs = progress.add_task("processing batched jobs", total=submitted_jobs, visible=True)
        task_batching = progress.add_task("...", total=0, visible=False)
        task_sending = progress.add_task("sending tasks to workers", visible=False)

        await enqueue([(root,) for root in roots])

        while jobs:
            tasks = []

            progress.reset(task_batching)
            progress.update(task_batching, description="waiting for finished batches", total=1, completed=0, visible=True)
            completed, _ = await asyncio.wait(jobs)
            jobs = []
            # progress.update(task_batching, description="loading data for finished batches", total=len(completed), completed=0)
            completed = await pool.gather(completed)

            progress.update(task_batching, total=sum(len(result[1]) for batch in completed for result in batch), completed=0)
            for batch in completed:
                batch = batch.result()
                progress.update(task_batching, description="processing discovered triangulations")
                for result in batch:
                    vt, vt_neighbors = result
                    for vt2 in vt_neighbors:
                        if is_new(vt2):
                            tasks.append((vt2,))
                            progress.update(task_exploring, total=len(seen))
                        progress.update(task_batching, advance=1)
                    graph[vt] = b"".join(md5(neighbor) for neighbor in vt_neighbors)
                    progress.update(task_exploring, advance=1)
                finished_jobs += 1
                progress.update(task_jobs, advance=1)

                if len(tasks) >= BATCH_LEN * 2 * threads:
                    await enqueue(tasks)
                    tasks = []

            await enqueue(tasks)

    return len(seen)


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


async def _main(recover, threads, scheduler, database, stratum_component, backend):
    import dask.config
    dask.config.set({'distributed.worker.daemon': False})
    import dask.distributed
    pool = await dask.distributed.Client(scheduler_file=scheduler, direct_to_workers=True, connection_limit=2**16, asynchronous=True)

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
    seen = None

    import dbm
    if recover:
        with dbm.open(database, 'r') as graph:
            roots, seen = loose_ends(graph)

        if not roots and not seen:
            roots = [dumps(vt)]

    with dbm.open(database, 'c' if recover else 'n') as graph:
        import datetime
        t0 = datetime.datetime.now()
        nodes = await explore(roots=roots, pool=pool, threads=threads, graph=graph, seen=seen, completed=max(0, len(graph) - len(roots)), backend=backend)
        t1 = datetime.datetime.now()
        elapsed = t1 - t0
        print(f'{nodes} triangulations computed in {elapsed * threads} CPU time; {elapsed} wall time')
    await pool.close()


@click.command()
@click.option('--recover/--no-recover', default=False, help='Continue from a previous aborted run.')
@click.option('--threads', default=os.cpu_count(), help='The number of simultaneous hyperthreads for statistical purposes.')
@click.option('--scheduler', default=None, help='The scheduler file to use, if not specified, the main program will serve as the scheduler.')
@click.option('--database', default='/tmp/ruth.cache', help='The path to the database to store the graph for --recover and later analysis.')
@click.option('--stratum-component', default=0, type=int)
@click.option('--backend', default='ppl')
def main(recover, threads, scheduler, database, stratum_component, backend):
    asyncio.run(_main(recover, threads, scheduler, database, stratum_component, backend))


if __name__ == '__main__':
    main()
