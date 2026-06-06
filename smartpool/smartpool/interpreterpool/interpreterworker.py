from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable, Any, Dict, Tuple

from ..worker import Worker

if TYPE_CHECKING:
    from concurrent.interpreters import Queue


class InterpreterWorker(Worker):

    def __init__(
        self, index:int,
        result_queue:Queue[Optional[Tuple[str, bool, Any, int]]],
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]],
        torch_gpu_available:bool
    ):
        import concurrent.interpreters as interpreters

        Worker.__init__(
            self, index,
            result_queue=result_queue,
            task_queue_cls=interpreters.create_queue,
            task_queue_args=(),
            task_queue_kwargs={},
            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs
        )

        if torch_gpu_available:
            self.change_device_cmd_queue:Optional[Queue[Optional[str]]] = interpreters.create_queue()
        else:
            self.change_device_cmd_queue:Optional[Queue[Optional[str]]] = None

    def change_device(self, device:str)->None:
        if self.change_device_cmd_queue is not None:
            self.change_device_cmd_queue.put(device)

    def _clear(self)->None:
        self.process_or_thread = None
        self.interp = None
        self._is_working = False
        self.imported_modules.clear()

    def start(self)->None:
        if self.process_or_thread is not None:
            return

        import concurrent.interpreters as interpreters
        from concurrent.interpreters import Interpreter

        self.interp:Interpreter = interpreters.create()
        self.process_or_thread = self.interp.call_in_thread(
            InterpreterWorker.run,
            self.task_queue, self.result_queue, self.change_device_cmd_queue,
            initializer=self.initializer,
            initargs=self.initargs,
            initkwargs=self.initkwargs
        )

    def join(self)->None:
        self.process_or_thread.join()
        self.interp.close()
        self._clear()
