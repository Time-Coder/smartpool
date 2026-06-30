from __future__ import annotations
from typing import TYPE_CHECKING
from ..worker import Worker

if TYPE_CHECKING:
    import threading
    from .infer_session_task import InferSessionTask
    from .infer_session_pool import InferSessionPool


class InferSessionWorker(Worker):

    def __init__(self, infer_session_pool:InferSessionPool):
        from queue import SimpleQueue

        Worker.__init__(
            self, infer_session_pool,
            task_queue_cls=SimpleQueue,
            task_queue_args=(),
            task_queue_kwargs={},
        )

    @property
    def thread(self)->threading.Thread:
        return self.executor

    @property
    def infer_session_pool(self)->InferSessionPool:
        return self.pool

    def add_task(self, task: InferSessionTask)->None:
        if task.use_io_binding:
            self.start()

        self.active_task = task
        infer_session_pool: InferSessionPool = self.infer_session_pool

        if infer_session_pool.print_info:
            print("infer with provider", task.provider, "in session", id(task.session.session))

        infer_session_pool._add_provider_running_device(task.provider)
        task.future.set_running_or_notify_cancel()
        if not task.use_io_binding:
            task.run_async()
        else:
            self.task_queue.put(task)

    @property
    def memory(self)->int:
        from ..resource import DataSize
        return (27 * DataSize.KB if self.executor is not None else 0)

    def start(self)->None:
        if self.executor is not None:
            return

        import threading
        self.executor = threading.Thread(
            target=self.run,
            name=f"{self.infer_session_pool._thread_name_prefix}{self.index}",
            daemon=True
        )
        self.executor.start()
        self._take_worker_memory()

    def join(self)->None:
        if self.executor is not None:
            self.executor.join()

        self._clear()

    def run(self):
        infer_session_pool: InferSessionPool = self.infer_session_pool
        while True:
            task: InferSessionTask = self.task_queue.get()
            if task is None:
                break

            success, result = task.exec()
            infer_session_pool._remove_provider_running_device(task.provider)
            infer_session_pool._on_task_done(task.id, success, result)
    