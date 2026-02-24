from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable, Any, Dict, Tuple, Set

if TYPE_CHECKING:
    from multiprocessing import SimpleQueue
    from .task import Task


class Worker:

    def __init__(
        self, index:int, result_queue:SimpleQueue[Optional[Tuple[str, bool, Any]]], mp_context:str,
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]],
        torch:bool=False
    ):
        if torch:
            from torch.multiprocessing.queue import SimpleQueue
        else:
            from multiprocessing import SimpleQueue

        self.is_torch:bool = torch
        self.index:int = index
        self.mp_context:str = mp_context
        self.result_queue:SimpleQueue[Optional[Tuple[str, bool, Any]]] = result_queue
        self.task_queue:SimpleQueue[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]] = SimpleQueue()
        self.change_device_cmd_queue:SimpleQueue[Optional[str]] = SimpleQueue()
        self._is_working:bool = False
        self._is_rss_dirty:bool = True
        self._cached_rss:int = 0
        self.modules:Set[str] = set()
        self.n_finished_tasks:int = 0
        self.initializer:Optional[Callable[..., Any]] = initializer
        self.initargs:Tuple[Any, ...] = initargs
        self.initkwargs:Optional[Dict[str, Any]] = initkwargs

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

    def overlap_modules_size(self, task:Task)->int:
        result = 0
        if len(task.module_sizes) < len(self.modules):
            less_modules = task.module_sizes
            more_modules = self.modules
        else:
            less_modules = self.modules
            more_modules = task.module_sizes

        for module_name in less_modules:
            if module_name in more_modules:
                result += task.module_sizes[module_name]

        return result

    @property
    def is_working(self)->bool:
        return self._is_working

    @is_working.setter
    def is_working(self, is_working:bool)->None:
        if self._is_working != is_working:
            self._is_working = is_working
            self._is_rss_dirty = True

    def add_task(self, task:Task)->None:
        self.is_working = True
        self.modules.update(task.module_sizes)
        self.task_queue.put(task.info())

    def change_device(self, device:str)->None:
        self.change_device_cmd_queue.put(device)

    def stop(self)->None:
        self.task_queue.put(None)
        self.change_device_cmd_queue.put(None)

        self.task_queue.close()
        self.change_device_cmd_queue.close()

    def start(self):
        import multiprocessing as mp
        import psutil

        ctx = mp.get_context(self.mp_context)
        Process = ctx.Process

        self.process:mp.Process = Process(
            target=Worker.run,
            args=(self.task_queue, self.result_queue, self.change_device_cmd_queue),
            kwargs={"initializer": self.initializer, "initargs": self.initargs, "initkwargs": self.initkwargs, "torch": self.is_torch},
            name=f"SmartProcessPool.worker:{self.index}",
            daemon=True
        )
        self.process.start()
        self.process_info = psutil.Process(self.process.pid)

    @property
    def modules_total_size(self)->int:
        from .utils import asizeof
        return sum(asizeof(module) for module in self.modules)

    def restart(self)->None:
        self.task_queue.put(None)
        self.change_device_cmd_queue.put(None)
        self.process.join()
        self.n_finished_tasks:int = 0
        self.modules.clear()
        self.start()
        
    def terminate(self)->None:
        self.process.terminate()

    def join(self)->None:
        self.process.join()

    current_func = None
    func_device_lock = None
    change_device_thread = None

    @staticmethod
    def _changing_device(cmd_queue:SimpleQueue[Optional[str]]):
        while True:
            device = cmd_queue.get()
            if device is None:
                cmd_queue.close()
                break

            if Worker.current_func is None or Worker.current_func._device != "cpu":
                continue

            with Worker.func_device_lock:
                Worker.current_func._device = device

    @staticmethod
    def run(
        task_queue:SimpleQueue[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]], result_queue:SimpleQueue[Optional[Tuple[str, bool, Any]]], change_device_cmd_queue:SimpleQueue[Optional[str]],
        initializer:Optional[Callable[..., Any]], initargs:Tuple[Any, ...], initkwargs:Optional[Dict[str, Any]], torch:bool
    ):
        if initializer is not None:
            if initkwargs is None:
                initkwargs = {}

            initializer(*initargs, **initkwargs)
        
        if torch:
            import threading
            import types

            Worker.func_device_lock = threading.Lock()
            Worker.change_device_thread = threading.Thread(target=Worker._changing_device, args=(change_device_cmd_queue,), daemon=True, name="changing_device")
            Worker.change_device_thread.start()

            def device(self):
                with Worker.func_device_lock:
                    return self._device

        while True:
            task = task_queue.get()
            if task is None:
                task_queue.close()
                result_queue.close()
                break

            task_id, func, args, kwargs = task

            if torch:
                try:
                    with Worker.func_device_lock:
                        func._device = device
                        func.device = types.MethodType(device, func)
                        Worker.current_func = func
                except AttributeError:
                    pass

            try:
                result = func(*args, **kwargs)
                success = True
            except BaseException as e:
                result = e
                success = False

            result_queue.put((task_id, success, result))