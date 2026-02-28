from __future__ import annotations
import threading
from queue import SimpleQueue
from typing import TYPE_CHECKING, Optional, Callable, Any, Dict, Tuple

from ..utils import _set_best_device


if TYPE_CHECKING:
    from ..task import Task, Result


class Worker(threading.Thread):

    def __init__(
        self, index:int, name_prefix:str,
        result_queue:SimpleQueue[Result],
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]]
    ):
        threading.Thread.__init__(self, name=f"{name_prefix}{index}", daemon=True)
        self.index:int = index
        self.task_queue:SimpleQueue[Task] = SimpleQueue()
        self.result_queue:SimpleQueue[Result] = result_queue
        self._is_working:bool = False
        self.initializer:Optional[Callable[..., Any]] = initializer
        self.initargs:Tuple[Any, ...] = initargs
        self.initkwargs:Optional[Dict[str, Any]] = initkwargs

        self.start()

    @property
    def is_working(self)->bool:
        return self._is_working

    @is_working.setter
    def is_working(self, is_working:bool)->None:
        if self._is_working != is_working:
            self._is_working = is_working
            self._is_rss_dirty = True

    def change_device(self, device:str)->None:
        _set_best_device(device, self.ident)

    def stop(self)->None:
        self.task_queue.put(None)

    def run(self):
        if self.initializer is not None:
            self.initializer(*self.initargs, **self.initkwargs)

        while True:
            task = self.task_queue.get()
            if task is None:
                break

            _set_best_device(task.device)
            self.result_queue.put(task.exec())