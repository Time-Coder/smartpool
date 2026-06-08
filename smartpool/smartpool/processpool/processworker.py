from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..worker import Worker

if TYPE_CHECKING:
    import multiprocessing as mp

    import psutil

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
        self.process_or_thread = None
        self.process_info = None
        self._is_working = False
        self.imported_modules.clear()
        self._is_rss_dirty = True
        self._cached_rss = 0

    def start(self)->None:
        if self.process_or_thread is not None:
            return

        import psutil

        process_pool:ProcessPool = self.pool
        self.process_or_thread:mp.Process = process_pool._ctx.Process(
            target=Worker.run,
            args=(self.task_queue, process_pool._result_queue, self.change_device_cmd_queue),
            kwargs={"initializer": process_pool._initializer, "initargs": process_pool._initargs, "initkwargs": process_pool._initkwargs},
            name=f"{process_pool._process_name_prefix}{self.index}",
            daemon=True
        )
        self.process_or_thread.start()
        self.process_info = psutil.Process(self.process_or_thread.pid)

    def join(self)->None:
        self.process_or_thread.join()
        self._clear()
