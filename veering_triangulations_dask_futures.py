import sage.all
import compress_pickle

def dumps(*args, **kwargs):
    return compress_pickle.dumps(*args, compression="bz2", **kwargs)

def loads(*args, **kwargs):
    return compress_pickle.loads(*args, compression="bz2", **kwargs)

r"""
Tiings for the computation of the geometric veering triangulations
of H(4)^hyp


Usage

To run it sequentially

   $ sage -python veering_triangulations_multiprocessing.py s

To run it in parallel with 4 processes

   $ sage -python veering_triangulations_multiprocessing.py 4

Twice sequentially and then 3 and 4 processes

   $ sage -python veering_triangulations_multiprocessing.py s 3 4
"""
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


def run_parallel(root, pool, graph):
    r"""
    Compute the graph using a multiprocessing.Pool

    NOTE: the maxtaskperchild argument to Pool is randomly set to 128 so
    that workers are restarted to avoid leaking
    """
    vt0 = dumps(root)
    graph[vt0] = []

    from dask.distributed import as_completed
    jobs = as_completed([pool.submit(geometric_neighbors_batched, [vt0])], with_results=True)

    from rich.progress import Progress, TextColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn
    with Progress(TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True, refresh_per_second=1) as progress:
        task = progress.add_task("completing graph", total=len(graph))

        for completed in jobs:
            _, results = completed
            tasks = []
            for result in results:
                vt, vt_neighbors = result
                for vt2 in vt_neighbors:
                    if vt2 not in graph:
                        tasks.append(vt2)
                        graph[vt2] = None
                        progress.update(task, total=len(graph))
                graph[vt] = vt_neighbors
                progress.update(task, advance=1)
            if not tasks:
                continue
            if len(tasks) < 8:
                batches = [tasks]
            else:
                batches = [tasks[:len(tasks)//2], tasks[len(tasks)//2:]]
            for batch in batches:
                jobs.add(pool.submit(geometric_neighbors_batched, batch))


def main():
    import os
    threads = os.cpu_count()

    import dask.distributed
    pool = dask.distributed.Client(n_workers=threads)
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

    import os
    print('Computing geometric veering triangulations in %s' % stratum_component)
    vt0 = VeeringTriangulation.from_stratum(stratum_component).copy(mutable=True)
    vt0.set_canonical_labels()
    vt0.set_immutable()

    import datetime
    t0 = datetime.datetime.now()
    graph = {}
    run_parallel(root=vt0, pool=pool, graph=graph)
    t1 = datetime.datetime.now()
    elapsed = t1 - t0
    print(f'{len(graph)} triangulations computed in {elapsed * threads} CPU time')


if __name__ == '__main__':
    main()
