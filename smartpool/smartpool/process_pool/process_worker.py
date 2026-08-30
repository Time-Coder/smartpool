from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import psutil

from ..worker import Worker

if TYPE_CHECKING:
    import multiprocessing as mp

    from .process_pool import ProcessPool


class ProcessWorker(Worker):

    def __init__(self, process_pool:ProcessPool):
        if process_pool._use_torch:
            from torch.multiprocessing.queue import Queue, SimpleQueue
        else:
            from multiprocessing.queues import Queue, SimpleQueue

        Worker.__init__(
            self, process_pool,
            task_queue_cls=Queue,
            task_queue_kwargs={"ctx": process_pool._ctx}
        )

        if process_pool._torch_gpu_available:
            self.change_device_cmd_queue:Optional[SimpleQueue[Optional[str]]] = SimpleQueue(ctx=process_pool._ctx)
        else:
            self.change_device_cmd_queue:Optional[SimpleQueue[Optional[str]]] = None

        self._is_rss_dirty:bool = True
        self._cached_rss:int = 0
        self.process_info:Optional[psutil.Process] = None

        self.start()

    @property
    def process(self)->mp.Process:
        return self.executor

    @property
    def process_pool(self)->ProcessPool:
        return self.pool

    @property
    def cached_rss(self)->int:
        if self._is_working:
            return self.rss

        if self._is_rss_dirty:
            self._cached_rss = self.rss
            self._is_rss_dirty = False

        return self._cached_rss

    @property
    def rss(self)->int:
        try:
            return self.process_info.memory_info().rss
        except Exception:
            return 0

    def _working_changed_hook(self):
        self._is_rss_dirty = True

    def change_device(self, device:str)->None:
        if self.change_device_cmd_queue is not None:
            self.change_device_cmd_queue.put(device)

    def _clear(self)->None:
        Worker._clear(self)
        self.process_info = None
        self.imported_modules.clear()
        self._is_rss_dirty = True
        self._cached_rss = 0

    @property
    def memory(self) -> int:
        return self.cached_rss

    def start(self)->None:
        if self.executor is not None:
            return

        process_pool:ProcessPool = self.process_pool
        self.executor:mp.Process = process_pool._ctx.Process(
            target=Worker.run,
            args=(self.task_queue, process_pool._result_queue, self.change_device_cmd_queue),
            kwargs={"initializer": process_pool._initializer, "initargs": process_pool._initargs, "initkwargs": process_pool._initkwargs},
            name=f"{process_pool._process_name_prefix}{self.index}",
            daemon=True
        )
        self.executor.start()
        self.process_info = psutil.Process(self.executor.pid)
        self._take_worker_memory()

    def join(self)->None:
        if self.executor is not None and self.executor._popen is not None:
            self.executor.join()
            self._dispose_task_queue()

        self._clear()

    def _dispose_task_queue(self)->None:
        if self._task_queue is None:
            return

        try:
            self._task_queue.cancel_join_thread()
        finally:
            try:
                self._task_queue.close()
            except Exception:
                pass
