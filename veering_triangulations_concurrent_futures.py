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

def geometric_neighbors(vt):
    r"""
    Return the pair ``(vt, list_of_neighbors)`` that consists of the input
    triangulation ``vt`` together with the list ``list_of_neighbors`` of
    neighbors in the flip graph.
    """
    ans = []
    for edges, col in vt.geometric_flips(backend='ppl'):
        new_vt = vt.copy(mutable=True)
        for e in edges:
            new_vt.flip(e, col, check=False)
        new_vt.set_canonical_labels()
        new_vt.set_immutable()
        ans.append(new_vt)
    return vt, ans


def run_sequential(vt0):
    r"""
    Compute the graph sequentially
    """
    todo = [vt0]
    new_todo = []
    graph = {vt0: []}

    while todo:
        vt = todo.pop()
        _, vt_neighbors = geometric_neighbors(vt)
        for vt2 in vt_neighbors:
            if vt2 not in graph:
                todo.append(vt2)
                graph[vt2] = None
        graph[vt] = vt_neighbors

    return graph

def run_parallel(vt0, start_method, processes=None):
    r"""
    Compute the graph using a multiprocessing.Pool

    NOTE: the maxtaskperchild argument to Pool is randomly set to 128 so
    that workers are restarted to avoid leaking
    """
    import multiprocessing as mp

    ctx = mp.get_context(start_method)

    todo = [vt0]
    pending = []
    graph = {vt0: []}

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes, mp_context=ctx) as pool: # max_tasks_per_child= needs Python 3.11
        while todo or pending:
            if todo:
                pending.extend(pool.submit(geometric_neighbors, task) for task in todo)
                todo = []
            else:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                pending = list(pending)
                for task in done:
                    vt, vt_neighbors = task.result()
                    for vt2 in vt_neighbors:
                        if vt2 not in graph:
                            todo.append(vt2)
                            graph[vt2] = None
                    graph[vt] = vt_neighbors

    return graph


if __name__ == '__main__':
    import os, sys
    from time import sleep, time

    args = []
    for arg in sys.argv[1:]:
        if arg == 's':
            args.append(-1)
        elif arg.isdigit():
            arg = int(arg)
            if arg <= 0:
                raise ValueError('number of processes must be non-negative')
            args.append(arg)

    import sage.all

    from veerer import VeeringTriangulation
    from surface_dynamics import AbelianStratum

    # Computation of the H(4)^hyperelliptic graph
    # For a larger computation, replace the line below with one of
    # stratum_component = AbelianStratum(4).odd_component()
    # stratum_component = AbelianStratum(2, 2).hyperelliptic_component()
    stratum_component = AbelianStratum(4).hyperelliptic_component()

    print('Computing geometric veering triangulations in %s' % stratum_component)
    vt0 = VeeringTriangulation.from_stratum(stratum_component).copy(mutable=True)
    vt0.set_canonical_labels()
    vt0.set_immutable()

    sleep(1)

    for arg in args:
        if arg == -1:
            t0 = time()
            graph = run_sequential(vt0)
            t1 = time()
            print('sequential: %d triangulations computed in %f' % (len(graph), t1 - t0))
        else:
            t0 = time()
            # NOTE: pari seems unhappy with 'fork'
            graph = run_parallel(vt0, 'forkserver', arg)
            t1 = time()
            print('parallel with %d processes: %d triangulations computed in %f' % (arg, len(graph), t1 - t0))
