from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, Tuple, Set, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .task import Task
    from .utils import QueueLike

    import multiprocessing as mp
    import threading


class Worker(ABC):

    _total_working_count:int = 0

    def __init__(
        self, index:int,
        result_queue:QueueLike[Tuple[str, bool, Any]],
        task_queue_cls:type,
        task_queue_args:Tuple[Any, ...],
        task_queue_kwargs:Dict[str, Any],
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]]
    ):
        self.index:int = index
        self._is_working:bool = False
        self.initializer:Optional[Callable[..., Any]] = initializer
        self.initargs:Tuple[Any, ...] = initargs
        self.initkwargs:Optional[Dict[str, Any]] = initkwargs
        self.imported_modules:Set[str] = set()
        self.n_finished_tasks:int = 0
        self.result_queue:QueueLike[Tuple[str, bool, Any]] = result_queue
        self.task_queue:QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]] = task_queue_cls(*task_queue_args, **task_queue_kwargs)
        self.process_or_thread:Optional[Union[mp.Process, threading.Thread]] = None
    
    def add_task(self, task:Task)->None:
        self.start()
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
            Worker._total_working_count += 1
        else:
            Worker._total_working_count -= 1

    @staticmethod
    def total_working_count()->int:
        return Worker._total_working_count

    @abstractmethod
    def _clear(self)->None:
        pass

    @abstractmethod
    def change_device(self, device:str)->None:
        pass

    @abstractmethod
    def start(self):
        pass

    def stop(self, wait:bool=True, clear:bool=False)->None:
        if self.process_or_thread is None:
            return
        
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
            _set_best_device(device, current_thread_id)

    @staticmethod
    def run(
        task_queue:QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]],
        result_queue:QueueLike[Tuple[str, bool, Any]],
        change_device_cmd_queue:Optional[QueueLike[str]],
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
            change_device_thread = threading.Thread(target=Worker._changing_device, args=(change_device_cmd_queue, current_thread_id), daemon=True, name="changing_device")
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
            except BaseException as e:
                result = e
                success = False

            result_queue.put((task_id, success, result))