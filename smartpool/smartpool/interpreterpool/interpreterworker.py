from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable, Any, Dict, Tuple, Set

from ..worker import Worker

if TYPE_CHECKING:
    from concurrent.interpreters import Queue
    from ..task import Task


class InterpreterWorker(Worker):

    def __init__(
        self, index:int,
        result_queue:Queue[Optional[Tuple[str, bool, Any, int]]],
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]],
        torch_cuda_available:bool
    ):
        Worker.__init__(self, index, initializer, initargs, initkwargs)

        import concurrent.interpreters as interpreters

        if torch_cuda_available:
            self.change_device_cmd_queue:Optional[Queue[Optional[str]]] = interpreters.create_queue()
        else:
            self.change_device_cmd_queue:Optional[Queue[Optional[str]]] = None

        self.result_queue:Queue[Optional[Tuple[str, bool, Any, int]]] = result_queue
        self.task_queue:Queue[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]] = interpreters.create_queue()

        self.start()

    def change_device(self, device:str)->None:
        if self.change_device_cmd_queue is not None:
            self.change_device_cmd_queue.put(device)

    def stop(self)->None:
        self.task_queue.put(None)

    def start(self):
        import concurrent.interpreters as interpreters
        from concurrent.interpreters import Interpreter

        self.interp:Interpreter = interpreters.create()
        self.thread = self.interp.call_in_thread(
            InterpreterWorker.run,
            self.task_queue, self.result_queue, self.change_device_cmd_queue,
            initializer=self.initializer,
            initargs=self.initargs,
            initkwargs=self.initkwargs
        )

    def restart(self)->None:
        self.task_queue.put(None)
        self.thread.join()
        self.interp.close()
        self.n_finished_tasks:int = 0
        self.imported_modules.clear()
        self.start()

    def join(self)->None:
        self.thread.join()
        self.interp.close()
