from __future__ import annotations
import threading
from queue import SimpleQueue
from typing import TYPE_CHECKING, Optional, Callable, Any, Dict, Tuple

from ..worker import Worker
from ..utils import _set_best_device


if TYPE_CHECKING:
    from ..task import Task, Result


class ThreadWorker(Worker):

    def __init__(
        self, index:int, name_prefix:str,
        result_queue:SimpleQueue[Result],
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]]
    ):
        Worker.__init__(self, index, initializer=initializer, initargs=initargs, initkwargs=initkwargs)
        self.name_prefix:str = name_prefix
        self.task_queue:SimpleQueue[Task] = SimpleQueue()
        self.result_queue:SimpleQueue[Result] = result_queue

        self.start()

    def change_device(self, device:str)->None:
        _set_best_device(device, self.thread.ident)

    def start(self)->None:
        self.thread = threading.Thread(
            target=ThreadWorker.run,
            name=f"{self.name_prefix}{self.index}",
            args=(self.task_queue, self.result_queue, self.initializer, self.initargs, self.initkwargs),
            daemon=True
        )
        self.thread.start()

    def stop(self)->None:
        self.task_queue.put(None)

    def join(self)->None:
        self.thread.join()

    @staticmethod
    def run(
        task_queue:SimpleQueue[Task],
        result_queue:SimpleQueue[Result],
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]]
    ):
        if initializer is not None:
            initializer(*initargs, **initkwargs)

        while True:
            task = task_queue.get()
            if task is None:
                break

            _set_best_device(task.device)
            result_queue.put(task.exec())