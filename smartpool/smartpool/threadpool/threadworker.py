from __future__ import annotations
import threading
from queue import SimpleQueue
from typing import TYPE_CHECKING, Optional, Callable, Any, Dict, Tuple

from ..worker import Worker
from ..utils import _set_best_device


if TYPE_CHECKING:
    from ..task import Task
    from .threadpool import ThreadPool


class ThreadWorker(Worker):

    def __init__(
        self, index:int, name_prefix:str,
        thread_pool:ThreadPool,
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]]
    ):
        Worker.__init__(
            self, index,
            result_queue=None,
            task_queue_cls=SimpleQueue,
            task_queue_args=(),
            task_queue_kwargs={},
            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs
        )
        self.name_prefix:str = name_prefix
        self.thread_pool:ThreadPool = thread_pool

    def add_task(self, task:Task)->None:
        self.start()
        self.task_queue.put(task)

    def change_device(self, device:str)->None:
        _set_best_device(device, self.process_or_thread.ident)

    def _clear(self)->None:
        self.process_or_thread = None
        self._is_working = False

    def start(self)->None:
        if self.process_or_thread is not None:
            return

        self.process_or_thread = threading.Thread(
            target=self.run,
            name=f"{self.name_prefix}{self.index}",
            daemon=True
        )
        self.process_or_thread.start()

    def join(self)->None:
        self.process_or_thread.join()
        self._clear()

    def run(self):
        if self.initializer is not None:
            self.initializer(*self.initargs, **self.initkwargs)

        while True:
            task:Task = self.task_queue.get()
            if task is None:
                break

            _set_best_device(task.device)
            success, result = task.exec()
            self.thread_pool._on_task_done(task.id, success, result)