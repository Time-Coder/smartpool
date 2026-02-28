from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable, Any, Dict, Tuple, Set

if TYPE_CHECKING:
    from concurrent.interpreters import Queue
    from ..task import Task


class Worker:

    def __init__(
        self, index:int,
        result_queue:Queue[Optional[Tuple[str, bool, Any, int]]],
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]],
        torch_cuda_available:bool
    ):
        from concurrent.interpreters import Queue
        import concurrent.interpreters as interpreters

        if torch_cuda_available:
            self.change_device_cmd_queue:Optional[Queue[Optional[str]]] = interpreters.create_queue()
        else:
            self.change_device_cmd_queue:Optional[Queue[Optional[str]]] = None

        self.index:int = index
        self.result_queue:Queue[Optional[Tuple[str, bool, Any, int]]] = result_queue
        self.task_queue:Queue[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]] = interpreters.create_queue()
        self._is_working:bool = False
        self.imported_modules:Set[str] = set()
        self.n_finished_tasks:int = 0
        self.initializer:Optional[Callable[..., Any]] = initializer
        self.initargs:Tuple[Any, ...] = initargs
        self.initkwargs:Optional[Dict[str, Any]] = initkwargs

        self.start()

    def overlap_modules_ratio(self, task:Task)->float:
        if not self.imported_modules:
            return 0
        
        return len(self.imported_modules & task.module_deps) / len(self.imported_modules)

    @property
    def is_working(self)->bool:
        return self._is_working

    @is_working.setter
    def is_working(self, is_working:bool)->None:
        if self._is_working != is_working:
            self._is_working = is_working

    def change_device(self, device:str)->None:
        if self.change_device_cmd_queue is not None:
            self.change_device_cmd_queue.put(device)

    def stop(self)->None:
        self.task_queue.put(None)

    def start(self):
        import concurrent.interpreters as interpreters
        from concurrent.interpreters import Interpreter

        self.interp:Interpreter = interpreters.create()
        self.thread = self.interp.call_in_thread(
            Worker.run,
            self.task_queue, self.result_queue, self.change_device_cmd_queue,
            initializer=self.initializer,
            initargs=self.initargs,
            initkwargs=self.initkwargs
        )

    def restart(self)->None:
        self.task_queue.put(None)
        self.thread.join()
        self.interp.close()
        self.n_finished_tasks:int = 0
        self.imported_modules.clear()
        self.start()

    def join(self)->None:
        self.thread.join()
        self.interp.close()

    @staticmethod
    def _changing_device(cmd_queue:Queue[Optional[str]]):
        from ..utils import _set_best_device
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
        from ..utils import _set_best_device
        

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