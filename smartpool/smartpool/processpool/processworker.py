from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable, Any, Dict, Tuple, Set

from ..worker import Worker

if TYPE_CHECKING:
    from multiprocessing.queues import SimpleQueue
    from ..task import Task


class ProcessWorker(Worker):

    def __init__(
        self, index:int, name_prefix:str,
        result_queue:SimpleQueue[Optional[Tuple[str, bool, Any]]], ctx,
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]],
        use_torch:bool, torch_cuda_available:bool
    ):
        Worker.__init__(
            self, index,
            initializer,
            initargs,
            initkwargs
        )
        if use_torch:
            from torch.multiprocessing.queue import SimpleQueue
        else:
            from multiprocessing.queues import SimpleQueue

        if torch_cuda_available:
            self.change_device_cmd_queue:Optional[SimpleQueue[Optional[str]]] = SimpleQueue(ctx=ctx)
        else:
            self.change_device_cmd_queue:Optional[SimpleQueue[Optional[str]]] = None

        self.ctx = ctx
        self.name_prefix:str = name_prefix
        self.result_queue:SimpleQueue[Optional[Tuple[str, bool, Any]]] = result_queue
        self.task_queue:SimpleQueue[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]] = SimpleQueue(ctx=ctx)
        self._is_rss_dirty:bool = True
        self._cached_rss:int = 0

        self.start()

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

    def stop(self)->None:
        self.task_queue.put(None)
        self.task_queue.close()

    def start(self):
        import multiprocessing as mp
        import psutil

        self.process:mp.Process = self.ctx.Process(
            target=ProcessWorker.run,
            args=(self.task_queue, self.result_queue, self.change_device_cmd_queue),
            kwargs={"initializer": self.initializer, "initargs": self.initargs, "initkwargs": self.initkwargs},
            name=f"{self.name_prefix}{self.index}",
            daemon=True
        )
        self.process.start()
        self.process_info = psutil.Process(self.process.pid)

    def restart(self)->None:
        self.task_queue.put(None)
        self.process.join()
        self.n_finished_tasks:int = 0
        self.imported_modules.clear()
        self.start()
        
    def terminate(self)->None:
        self.process.terminate()

    def join(self)->None:
        self.process.join()
