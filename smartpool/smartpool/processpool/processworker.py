from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable, Any, Dict, Tuple

from ..worker import Worker

if TYPE_CHECKING:
    from ..utils import QueueLike
    from ..task import Task

    import multiprocessing as mp
    import psutil


class ProcessWorker(Worker):

    def __init__(
        self, index:int, name_prefix:str,
        result_queue:QueueLike[Optional[Tuple[str, bool, Any]]], ctx,
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]],
        use_torch:bool, torch_cuda_available:bool
    ):
        if use_torch:
            from torch.multiprocessing.queue import SimpleQueue
        else:
            from multiprocessing.queues import SimpleQueue

        Worker.__init__(
            self, index,
            result_queue=result_queue,
            task_queue_cls=SimpleQueue,
            task_queue_args=(),
            task_queue_kwargs={"ctx": ctx},
            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs
        )

        if torch_cuda_available:
            self.change_device_cmd_queue:Optional[SimpleQueue[Optional[str]]] = SimpleQueue(ctx=ctx)
        else:
            self.change_device_cmd_queue:Optional[SimpleQueue[Optional[str]]] = None

        self.ctx = ctx
        self.name_prefix:str = name_prefix
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
        except:
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

        self.process_or_thread:mp.Process = self.ctx.Process(
            target=ProcessWorker.run,
            args=(self.task_queue, self.result_queue, self.change_device_cmd_queue),
            kwargs={"initializer": self.initializer, "initargs": self.initargs, "initkwargs": self.initkwargs},
            name=f"{self.name_prefix}{self.index}",
            daemon=True
        )
        self.process_or_thread.start()
        self.process_info = psutil.Process(self.process_or_thread.pid)
        
    def terminate(self)->None:
        self.process_or_thread.terminate()
        self._clear()

    def join(self)->None:
        self.process_or_thread.join()
        self._clear()
