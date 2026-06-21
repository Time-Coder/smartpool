from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Tuple, Union, Type

if TYPE_CHECKING:
    import multiprocessing as mp

    from .pool import Pool
    from .task import Task
    from .utils import QueueLike


WORKER_DEFAULT_MEMORY: int = 52428800


class Worker(ABC):

    _total_working_count_lock:threading.Lock = threading.Lock()
    _total_working_count:int = 0

    def __init__(
        self, pool:Pool,
        task_queue_cls:Type[QueueLike]=None,
        task_queue_args:Optional[Tuple[Any, ...]]=None,
        task_queue_kwargs:Optional[Dict[str, Any]]=None
    ):
        self.index: int = len(pool._workers)
        self.pool: Pool = pool
        self._is_working: bool = False
        self.imported_modules: Set[str] = set()
        self.n_finished_tasks: int = 0
        self.task_queue_cls: Type[QueueLike] = task_queue_cls
        self.task_queue_args: Optional[Tuple[Any, ...]] = task_queue_args
        self.task_queue_kwargs: Optional[Dict[str, Any]] = task_queue_kwargs
        self._task_queue: Optional[QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]]] = None
        self.executor: Optional[Union[mp.Process, threading.Thread]] = None
        self.active_task: Optional[Task] = None
        self._memory_taken: bool = False

    @property
    def task_queue(self)->Optional[QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]]]:
        if self._task_queue is not None:
            return self._task_queue

        if self.task_queue_cls is not None:
            if self.task_queue_args is None:
                self.task_queue_args = ()

            if self.task_queue_kwargs is None:
                self.task_queue_kwargs = {}
                
            self._task_queue: Optional[QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]]] = self.task_queue_cls(*self.task_queue_args, **self.task_queue_kwargs)

        return self._task_queue

    def add_task(self, task: Task)->None:
        self.start()
        self.active_task = task
        task.future.set_running_or_notify_cancel()
        self.task_queue.put(task.info())

    @property
    def is_working(self)->bool:
        return self._is_working

    @is_working.setter
    def is_working(self, is_working:bool)->None:
        if self._is_working == is_working:
            return

        self._is_working = is_working

        if is_working:
            with Worker._total_working_count_lock:
                Worker._total_working_count += 1
                self.pool._workers_working_count += 1
        else:
            with Worker._total_working_count_lock:
                Worker._total_working_count -= 1
                self.pool._workers_working_count -= 1

    @staticmethod
    def total_working_count()->int:
        with Worker._total_working_count_lock:
            return Worker._total_working_count

    def _clear(self)->None:
        self.executor = None
        self._is_working = False

    def change_device(self, device:str)->None:
        pass

    @property
    def memory(self) -> int:
        return WORKER_DEFAULT_MEMORY

    def _take_worker_memory(self) -> None:
        if self._memory_taken:
            return
        self._memory_taken = True
        with self.pool._sys_info_lock:
            self.pool._sys_info.cpu_mem_free -= self.memory

    def _release_worker_memory(self) -> None:
        if not self._memory_taken:
            return
        self._memory_taken = False
        with self.pool._sys_info_lock:
            self.pool._sys_info.cpu_mem_free += self.memory

    @abstractmethod
    def start(self):
        pass

    def stop(self, wait:bool=True, clear:bool=False)->None:
        if self.executor is None:
            return

        self._release_worker_memory()
        self.task_queue.put(None)
        if wait:
            self.join()
        elif clear:
            self._clear()

    @abstractmethod
    def join(self)->None:
        pass

    def overlap_modules_ratio(self, task:Task)->float:
        if not self.imported_modules:
            return 0

        return len(self.imported_modules & task.module_deps) / len(self.imported_modules)

    @staticmethod
    def _changing_device(cmd_queue:QueueLike[Optional[str]], current_thread_id):
        from .utils import _set_best_device
        while True:
            device = cmd_queue.get()
            if device is None:
                break

            _set_best_device(device, current_thread_id)

    @staticmethod
    def run(
        task_queue:QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]],
        result_queue:QueueLike[Tuple[str, bool, Any]],
        change_device_cmd_queue:Optional[QueueLike[Optional[str]]],
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]]
    ):
        from .utils import _set_best_device


        if initializer is not None:
            if initkwargs is None:
                initkwargs = {}

            initializer(*initargs, **initkwargs)

        if change_device_cmd_queue is not None:
            import threading

            current_thread_id = threading.get_ident()
            change_device_thread = threading.Thread(target=Worker._changing_device, args=(change_device_cmd_queue, current_thread_id), name="changing_device")
            change_device_thread.start()

        while True:
            task = task_queue.get()
            if task is None:
                break

            task_id, task_device, func, args, kwargs = task
            _set_best_device(task_device)

            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = e
                success = False

            result_queue.put((task_id, success, result))

        if change_device_cmd_queue is not None and change_device_thread.is_alive():
            change_device_cmd_queue.put(None)
            change_device_thread.join()
