from __future__ import annotations
from typing import TYPE_CHECKING
from ..worker import Worker

if TYPE_CHECKING:
    import numpy as np
    import threading
    from .infersessiontask import InfersessionTask
    from .infersessionpool import InferSessionPool


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

    def add_task(self, task: InfersessionTask)->None:
        if task.use_io_binding:
            self.start()

        self.active_task = task
        infer_session_pool: InferSessionPool = self.infer_session_pool

        if infer_session_pool.print_info:
            print("infer with provider", task.provider, "in session", id(task.session_info.session))

        provider = task.provider
        infer_session_pool._add_provider_running_device(provider)

        task.future.set_running_or_notify_cancel()

        if not task.use_io_binding:
            try:
                task.session_info.run_async(task.output_names, input_feed=task.kwargs, callback=self.callback, user_data=None, run_options=task.run_options)
            except Exception as e:
                self.is_working = False
                infer_session_pool._remove_provider_running_device(provider)
                infer_session_pool._on_task_done(task.id, False, e)
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

    def callback(self, results:np.ndarray, user_data: None, error_str: str) -> None:
        task_id = self.active_task.id
        if error_str:
            success = False
            result = RuntimeError(error_str)
        else:
            success = True
            result = results

        infer_session_pool: InferSessionPool = self.infer_session_pool
        task: InfersessionTask = infer_session_pool._tasks[task_id]
        infer_session_pool._remove_provider_running_device(task.provider)
        infer_session_pool._on_task_done(task_id, success, result)

    def run(self):
        infer_session_pool: InferSessionPool = self.infer_session_pool
        while True:
            task: InfersessionTask = self.task_queue.get()
            if task is None:
                break

            success, result = task.exec()
            infer_session_pool._remove_provider_running_device(task.provider)
            infer_session_pool._on_task_done(task.id, success, result)
    