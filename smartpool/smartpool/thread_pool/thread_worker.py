from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, Optional

from ..utils import _set_best_device, _set_best_stream
from ..worker import Worker

if TYPE_CHECKING:
    from torch.cuda import Stream

    from ..gpuinfo import Device
    from ..task import Task
    from .thread_pool import ThreadPool


class ThreadWorker(Worker):

    def __init__(self, thread_pool:ThreadPool):
        Worker.__init__(
            self, thread_pool,
            task_queue_cls=thread_pool._get_task_queue
        )
        self._streams: Dict[str, Stream] = {}
        self._active_task: Optional[Task] = None

    @property
    def thread(self)->threading.Thread:
        return self.executor

    @property
    def thread_pool(self)->ThreadPool:
        return self.pool

    def add_task(self, task:Task)->None:
        self.start()
        task.future.set_running_or_notify_cancel()
        self.task_queue.put(task)

    def change_device(self, device:Device)->None:
        _set_best_device(device, self.executor.ident)
        self._set_stream(device)

    def _set_stream(self, device:Device):
        if not self._active_task.use_torch:
            return

        if device not in self._streams:
            if device.torch_device.startswith("cuda") or device.torch_device.startswith("hip"):
                from torch.cuda import Stream
                self._streams[device] = Stream(device=device.torch_device)
            else:
                self._streams[device] = None

        _set_best_stream(self._streams[device], self.executor.ident)

    @property
    def memory(self)->int:
        from ..resource import DataSize
        return (27 * DataSize.KB if self.executor is not None else 0)

    def start(self)->None:
        if self.executor is not None:
            return

        self.executor = threading.Thread(
            target=self.run,
            name=f"{self.thread_pool._thread_name_prefix}{self.index}",
            daemon=True
        )
        self.executor.start()
        self._take_worker_memory()

    def join(self)->None:
        if self.executor is not None:
            self.executor.join()

        self._clear()

    def run(self):
        thread_pool:ThreadPool = self.thread_pool
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
            self._active_task = None
            task = None
            result = None
