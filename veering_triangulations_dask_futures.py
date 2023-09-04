import sage.all
import compress_pickle
import click
import os


def dumps(*args, **kwargs):
    return compress_pickle.dumps(*args, compression="bz2", **kwargs)


def loads(*args, **kwargs):
    return compress_pickle.loads(*args, compression="bz2", **kwargs)


def geometric_neighbors(vtd):
    r"""
    Return the pair ``(vt, list_of_neighbors)`` that consists of the input
    triangulation ``vt`` together with the list ``list_of_neighbors`` of
    neighbors in the flip graph.
    """
    vt = loads(vtd)
    ans = []
    for edges, col in vt.geometric_flips(backend='ppl'):
        new_vt = vt.copy(mutable=True)
        for e in edges:
            new_vt.flip(e, col, check=False)
        new_vt.set_canonical_labels()
        new_vt.set_immutable()
        ans.append(dumps(new_vt))
    return vtd, ans


def geometric_neighbors_batched(vts):
    return [geometric_neighbors(vt) for vt in vts]


def md5(vt):
    import hashlib
    return hashlib.md5(vt).digest()


def explore(roots, pool, threads, graph, seen=None, completed=0):
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
    jobs = as_completed([])

    def enqueue(tasks):
        nonlocal submitted_jobs, progress

        if not tasks:
            return

        # We want there to always be 2 * threads jobs in the queue.
        nbatches_from_threads = max(1, 3 * threads - (submitted_jobs - finished_jobs))
        # We do not want to ship any tasks that contain more than 64 jobs.
        nbatches_from_package_size = max(1, len(tasks) // 64)

        nbatches = max(nbatches_from_threads, nbatches_from_package_size)
        nbatches = min(len(tasks), nbatches)

        batches = [tasks[offset::nbatches] for offset in range(nbatches)]
        batches = pool.scatter(batches)

        jobs.update(pool.map(geometric_neighbors_batched, batches))

        submitted_jobs += len(batches)
        progress.update(task_jobs, total=submitted_jobs)

    submitted_jobs = 1
    finished_jobs = 0

    from rich.progress import Progress, TextColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn
    with Progress(TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True, refresh_per_second=1) as progress:
        task_exploring = progress.add_task("exploring graph", completed=completed, total=len(seen))
        task_jobs = progress.add_task("processing batched jobs", total=submitted_jobs)

        enqueue(roots)

        while not jobs.is_empty():
            tasks = []

            from itertools import chain
            completed = jobs.next_batch()
            completed = pool.gather(completed)

            for result in chain.from_iterable(completed):
                vt, vt_neighbors = result
                for vt2 in vt_neighbors:
                    if is_new(vt2):
                        tasks.append(vt2)
                        progress.update(task_exploring, total=len(seen))
                graph[vt] = b"".join(md5(neighbor) for neighbor in vt_neighbors)
                progress.update(task_exploring, advance=1)

            finished_jobs += len(completed)
            progress.update(task_jobs, advance=len(completed))

            enqueue(tasks)


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


@click.command()
@click.option('--recover/--no-recover', default=False, help='Continue from a previous aborted run.')
@click.option('--threads', default=os.cpu_count(), help='The number of simultaneous hyperthreads for statistical purposes.')
@click.option('--scheduler', default=None, help='The scheduler file to use, if not specified, the main program will serve as the scheduler.')
@click.option('--database', default='/tmp/ruth.cache', help='The path to the database to store the graph for --recover and later analysis.')
def main(recover, threads, scheduler, graph):
    import dask.distributed
    pool = dask.distributed.Client(scheduler_file=scheduler, direct_to_workers=True)

    from veerer import VeeringTriangulation
    from surface_dynamics import AbelianStratum

    # stratum_component = AbelianStratum(4).odd_component()
    # stratum_component = AbelianStratum(2, 2).hyperelliptic_component()
    # stratum_component = AbelianStratum(4).hyperelliptic_component()
    # stratum_component = AbelianStratum(3, 1).unique_component()
    # stratum_component = AbelianStratum(2, 2).hyperelliptic_component()
    # stratum_component = AbelianStratum(2, 2).odd_component()
    stratum_component = AbelianStratum(6).hyperelliptic_component()
    # stratum_component = AbelianStratum(6).odd_component()

    import os
    print('Computing geometric veering triangulations in %s' % stratum_component)
    vt0 = VeeringTriangulation.from_stratum(stratum_component).copy(mutable=True)
    vt0.set_canonical_labels()
    vt0.set_immutable()

    roots = [dumps(vt0)]
    seen = None

    import dbm
    if recover:
        with dbm.open(database, 'r') as graph:
            roots, seen = loose_ends(graph)

        if not roots and not seen:
            roots = [dumps(vt0)]

    with dbm.open(database, 'c' if recover else 'n') as graph:
        import datetime
        t0 = datetime.datetime.now()
        nodes = explore(roots=roots, pool=pool, threads=threads, graph=graph, seen=seen, completed=max(0, len(graph) - len(roots)))
        t1 = datetime.datetime.now()
        elapsed = t1 - t0
        print(f'{nodes} triangulations computed in {elapsed * threads} CPU time; {elapsed} wall time')


if __name__ == '__main__':
    main()
