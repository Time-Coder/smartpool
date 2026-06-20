from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import psutil

from ..worker import Worker

if TYPE_CHECKING:
    import multiprocessing as mp
    from .processpool import ProcessPool


class ProcessWorker(Worker):

    def __init__(self, process_pool:ProcessPool):
        if process_pool._use_torch:
            from torch.multiprocessing.queue import SimpleQueue
        else:
            from multiprocessing.queues import SimpleQueue

        Worker.__init__(
            self, process_pool,
            task_queue_cls=SimpleQueue,
            task_queue_args=(),
            task_queue_kwargs={"ctx": process_pool._ctx}
        )

        if process_pool._torch_gpu_available:
            self.change_device_cmd_queue:Optional[SimpleQueue[Optional[str]]] = SimpleQueue(ctx=process_pool._ctx)
        else:
            self.change_device_cmd_queue:Optional[SimpleQueue[Optional[str]]] = None

        self._is_rss_dirty:bool = True
        self._cached_rss:int = 0
        self.process_info:Optional[psutil.Process] = None

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

    @property
    def is_working(self)->bool:
        return self._is_working

    @is_working.setter
    def is_working(self, is_working:bool)->None:
        if self._is_working == is_working:
            return

        self._is_working = is_working
        self._is_rss_dirty = True

        if is_working:
            Worker._total_working_count += 1
        else:
            Worker._total_working_count -= 1

    def change_device(self, device:str)->None:
        if self.change_device_cmd_queue is not None:
            self.change_device_cmd_queue.put(device)

    def _clear(self)->None:
        Worker._clear(self)
        self.process_info = None
        self.imported_modules.clear()
        self._is_rss_dirty = True
        self._cached_rss = 0

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

    def join(self)->None:
        if self.executor is not None:
            self.executor.join()
            
        self._clear()
