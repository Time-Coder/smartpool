from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from queue import Queue
    from .task import Task


class Worker(ABC):

    _total_working_count:int = 0

    def __init__(
        self, index:int,
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
    def change_device(self, device:str)->None:
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def join(self)->None:
        pass

    def overlap_modules_ratio(self, task:Task)->float:
        if not self.imported_modules:
            return 0
        
        return len(self.imported_modules & task.module_deps) / len(self.imported_modules)

    @staticmethod
    def _changing_device(cmd_queue:Queue[Optional[str]]):
        from .utils import _set_best_device
        while True:
            device = cmd_queue.get()
            _set_best_device(device)

    @staticmethod
    def run(
        task_queue:Queue[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]],
        result_queue:Queue[Optional[Tuple[str, bool, Any]]],
        change_device_cmd_queue:Optional[Queue[Optional[str]]],
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

            change_device_thread = threading.Thread(target=Worker._changing_device, args=(change_device_cmd_queue,), daemon=True, name="changing_device")
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