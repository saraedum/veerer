import sage.all
import compress_pickle


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


def explore(root, pool, threads, graph):
    vt0 = dumps(root)

    def is_new(vt):
        key = md5(vt)
        if key in seen:
            return False
        seen.add(key)
        return True

    seen = set()
    assert is_new(vt0)

    from dask.distributed import as_completed
    jobs = as_completed([pool.submit(geometric_neighbors_batched, [vt0])])

    submitted_jobs = 1
    finished_jobs = 0

    from rich.progress import Progress, TextColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn
    with Progress(TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True, refresh_per_second=1) as progress:
        task_exploring = progress.add_task("exploring graph", total=len(seen))
        task_jobs = progress.add_task("processing batched jobs", total=submitted_jobs)

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

            if not tasks:
                continue

            # We want there to always be 2 * threads jobs in the queue.
            target = min(len(tasks), max(1, 3 * threads - (submitted_jobs - finished_jobs)))

            batches = [tasks[offset::target] for offset in range(target)]

            for batch in batches:
                jobs.add(pool.submit(geometric_neighbors_batched, batch))
                submitted_jobs += 1

            progress.update(task_jobs, total=submitted_jobs)

    return len(seen)


def main():
    import os
    threads = os.cpu_count()

    import dask.distributed
    pool = dask.distributed.Client(n_workers=threads, nthreads=1, direct_to_workers=True)

    from veerer import VeeringTriangulation
    from surface_dynamics import AbelianStratum

    # stratum_component = AbelianStratum(4).odd_component()
    # stratum_component = AbelianStratum(2, 2).hyperelliptic_component()
    stratum_component = AbelianStratum(4).hyperelliptic_component()
    # stratum_component = AbelianStratum(3, 1).unique_component()
    # stratum_component = AbelianStratum(2, 2).hyperelliptic_component()
    # stratum_component = AbelianStratum(2, 2).odd_component()
    # stratum_component = AbelianStratum(6).hyperelliptic_component()
    # stratum_component = AbelianStratum(6).odd_component()

    import os
    print('Computing geometric veering triangulations in %s' % stratum_component)
    vt0 = VeeringTriangulation.from_stratum(stratum_component).copy(mutable=True)
    vt0.set_canonical_labels()
    vt0.set_immutable()

    import dbm
    with dbm.open('/tmp/ruth.cache', 'n') as graph:
        import datetime
        t0 = datetime.datetime.now()
        nodes = explore(root=vt0, pool=pool, threads=threads, graph=graph)
        t1 = datetime.datetime.now()
        elapsed = t1 - t0
        print(f'{nodes} triangulations computed in {elapsed * threads} CPU time; {elapsed} wall time')


if __name__ == '__main__':
    main()
