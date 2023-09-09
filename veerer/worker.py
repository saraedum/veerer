r"""
Setup for a functioning SageMath worker when spawned with

dask-worker --nworkers 1 --nthreads 1 --no-nanny --preload veerer.worker

The main problem that we are working around here is the following:

PARI in SageMath as shipped by conda-forge segfaults when sage.all has been
imported on one thread and then PARI is invoked on another thread. However,
dask runs the actual worker on a separate thread so we would have to postpone
loading sage.all until the worker thread has started and not preload sage.all.
However, cysignals refuses to setup signals when not imported on the main
thread :boom:

To work around, we create a worker with a forkserver in a clean process so dask
never gets to see sage.all. However, we cannot fork if we are a daemon so we
need to make sure that we are not the child process of a nanny (so, --no-nanny
see invocation above.)
"""

import multiprocessing

forkserver = multiprocessing.get_context("forkserver")
multiprocessing.set_forkserver_preload(["sage.all"])


class Batched:
    def __init__(self, callable, **kwargs):
        self._callable = callable
        self._kwargs = kwargs

    def __call__(self, batch):
        return [work(self._callable, *item, **self._kwargs) for item in batch]


class Worker:
    def __init__(self):
        self._worker = None

    def _ensure(self):
        if self._worker is None:
            self._work_queue = forkserver.Queue()
            self._result_queue = forkserver.Queue()
            self._worker = forkserver.Process(target=Worker._work, args=(self,), daemon=True)
            self._worker.start()

    @staticmethod
    def _work(self):
        while True:
            try:
                message = self._work_queue.get()
            except ValueError:
                break

            callable, args, kwargs = message
            self._result_queue.put(callable(*args, **kwargs))

    def __call__(self, callable, *args, **kwargs):
        self._ensure()
        self._work_queue.put((callable, args, kwargs))
        return self._result_queue.get()


work = Worker()
