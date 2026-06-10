from __future__ import annotations

import threading
from queue import SimpleQueue
from typing import TYPE_CHECKING, Dict, Optional

from ..utils import _set_best_device, _set_best_stream
from ..worker import Worker

if TYPE_CHECKING:
    from torch.cuda import Stream

    from ..task import Task
    from .threadpool import ThreadPool


class ThreadWorker(Worker):

    def __init__(self, thread_pool:ThreadPool):
        Worker.__init__(
            self, thread_pool,
            task_queue_cls=SimpleQueue,
            task_queue_args=(),
            task_queue_kwargs={},
        )
        self._active_task:Optional[Task] = None
        self._streams:Dict[str, Stream] = {}

    def add_task(self, task:Task)->None:
        self.start()
        self.task_queue.put(task)
        task.future.set_running_or_notify_cancel()

    def change_device(self, device:str)->None:
        _set_best_device(device, self.task_executor.ident)
        self._set_stream(device)

    def _set_stream(self, device:str):
        if not self._active_task.use_torch:
            return

        if device not in self._streams:
            if device.startswith("cuda") or device.startswith("hip"):
                from torch.cuda import Stream
                self._streams[device] = Stream(device=device)
            else:
                self._streams[device] = None

        _set_best_stream(self._streams[device], self.task_executor.ident)

    def _clear(self)->None:
        self.task_executor = None
        self._is_working = False

    def start(self)->None:
        if self.task_executor is not None:
            return

        thread_pool:ThreadPool = self.pool
        self.task_executor = threading.Thread(
            target=self.run,
            name=f"{thread_pool._thread_name_prefix}{self.index}",
            daemon=True
        )
        self.task_executor.start()

    def join(self)->None:
        if self.task_executor is not None:
            self.task_executor.join()

        self._clear()

    def run(self):
        thread_pool:ThreadPool = self.pool
        if thread_pool._initializer is not None:
            thread_pool._initializer(*thread_pool._initargs, **thread_pool._initkwargs)

        while True:
            task:Task = self.task_queue.get()
            self._active_task = task
            if task is None:
                break

            _set_best_device(task.device)
            self._set_stream(task.device)
            success, result = task.exec()

            thread_pool._on_task_done(task.id, success, result)
