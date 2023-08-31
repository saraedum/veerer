import sage.all
import pickle
import dask.distributed


def geometric_neighbors(vt):
    r"""
    Return the pair ``(vt, list_of_neighbors)`` that consists of the input
    triangulation ``vt`` together with the list ``list_of_neighbors`` of
    neighbors in the flip graph.
    """
    assert not hasattr(vt, "result")
    if isinstance(vt, bytes):
        vt = pickle.loads(vt)

    neighbors = []

    for edges, col in vt.geometric_flips(backend='ppl'):
        neighbor = vt.copy(mutable=True)
        for e in edges:
            neighbor.flip(e, col, check=False)
        neighbor.set_canonical_labels()
        neighbor.set_immutable()
        neighbors.append(neighbor)

    return neighbors


def md5(vt):
    assert not hasattr(vt, "result")
    if not isinstance(vt, bytes):
        vt = pickle.dumps(vt)

    import hashlib
    return hashlib.md5(vt).digest()


def md5s(vts):
    return [md5(vt) for vt in vts]


def select(vts, i):
    return vts[i]


def run_parallel(roots, pool, graph, threads):
    keys = md5s(roots)
    roots = pool.scatter(roots)

    unprocessed = list(zip(keys, roots))

    seen = set(keys)
    processing = {}

    from rich.progress import Progress, TextColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn
    with Progress(TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True, refresh_per_second=1) as progress:
        progress_bar = progress.add_task("completing graph", total=len(unprocessed))

        while processing or unprocessed:
            print(f"processing {len(processing)}, unprocessed {len(unprocessed)}, seen {len(seen)}")

            while unprocessed and len(processing) < 64 * threads:
                key, vt = unprocessed.pop()
                neighbors = pool.submit(geometric_neighbors, vt)
                processing[neighbors] = (geometric_neighbors, (vt, key))

            if processing:
                import concurrent.futures

                completed, _ = dask.distributed.wait(list(processing.keys()), return_when=concurrent.futures.FIRST_COMPLETED)
                print(f"completed {len(completed)}")

                for task in completed:
                    kind, args = processing.pop(task)
                    workers = pool.who_has(task)[task.key]

                    if kind == geometric_neighbors:
                        vt, key = args
                        vts = task

                        processing[pool.submit(md5s, vts, workers=workers, priority=2)] = (md5s, (vt, key, vts))

                    if kind == md5s:
                        vt, key, vts = args
                        vt = vt.result()

                        keys = task.result()

                        assert key not in graph

                        graph[key] = (vt, keys)

                        for i, key in enumerate(keys):
                            if key not in seen:
                                unprocessed.append((key, pool.submit(select, vts, i, workers=workers, priority=1)))
                                seen.add(key)

                        progress.update(progress_bar, total=len(seen))
                        progress.update(progress_bar, advance=1)

    return graph


def main():

    import os
    threads = os.cpu_count()

    pool = dask.distributed.Client(n_workers=threads, direct_to_workers=True)
    # pool = dask.distributed.Client("localhost:8786", direct_to_workers=False)
    # preload="dask_preload", serializers=["pickle"], deserializers=["pickle"], nworkers=8, nthreads=1)

    from veerer import VeeringTriangulation
    from surface_dynamics import AbelianStratum

    # Computation of the H(4)^hyperelliptic graph
    # For a larger computation, replace the line below with one of
    # stratum_component = AbelianStratum(4).odd_component()
    # stratum_component = AbelianStratum(2, 2).hyperelliptic_component()
    stratum_component = AbelianStratum(4).hyperelliptic_component()
    # stratum_component = AbelianStratum(3, 1).unique_component()
    # stratum_component = AbelianStratum(2, 2).hyperelliptic_component()
    # stratum_component = AbelianStratum(2, 2).odd_component()
    # stratum_component = AbelianStratum(6).hyperelliptic_component()
    # stratum_component = AbelianStratum(6).odd_component()

    print('Computing geometric veering triangulations in %s' % stratum_component)
    vt0 = VeeringTriangulation.from_stratum(stratum_component).copy(mutable=True)
    vt0.set_canonical_labels()
    vt0.set_immutable()

    import datetime
    t0 = datetime.datetime.now()
    graph = run_parallel([vt0], pool, graph={}, threads=threads)
    t1 = datetime.datetime.now()
    elapsed = t1 - t0
    print(f'{len(graph)} triangulations computed in {elapsed * threads} CPU time')


if __name__ == '__main__':
    main()
