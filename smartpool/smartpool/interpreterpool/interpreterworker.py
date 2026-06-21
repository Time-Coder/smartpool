from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..worker import Worker

if TYPE_CHECKING:
    import threading
    from concurrent.interpreters import Queue

    from .interpreterpool import InterpreterPool


class InterpreterWorker(Worker):

    def __init__(self, interpreter_pool:InterpreterPool):
        import concurrent.interpreters as interpreters

        Worker.__init__(
            self, interpreter_pool,
            task_queue_cls=interpreters.create_queue,
            task_queue_args=(),
            task_queue_kwargs={}
        )

        if interpreter_pool._torch_gpu_available:
            self.change_device_cmd_queue:Optional[Queue[Optional[str]]] = interpreters.create_queue()
        else:
            self.change_device_cmd_queue:Optional[Queue[Optional[str]]] = None

    @property
    def thread(self)->threading.Thread:
        return self.executor

    @property
    def interpreter_pool(self)->InterpreterPool:
        return self.pool

    def change_device(self, device:str)->None:
        if self.change_device_cmd_queue is not None:
            self.change_device_cmd_queue.put(device)

    def _clear(self)->None:
        Worker._clear()
        self.interp = None
        self.imported_modules.clear()

    def start(self)->None:
        if self.executor is not None:
            return

        import concurrent.interpreters as interpreters
        from concurrent.interpreters import Interpreter

        interpreter_pool:InterpreterPool = self.interpreter_pool
        self.interp:Interpreter = interpreters.create()
        self.executor = self.interp.call_in_thread(
            InterpreterWorker.run,
            task_queue=self.task_queue,
            result_queue=interpreter_pool._result_queue,
            change_device_cmd_queue=self.change_device_cmd_queue,
            initializer=interpreter_pool._initializer,
            initargs=interpreter_pool._initargs,
            initkwargs=interpreter_pool._initkwargs
        )
        self._take_worker_memory()

    def join(self)->None:
        if self.executor is not None:
            self.executor.join()

        if self.interp is not None:
            self.interp.close()

        self._clear()
