from __future__ import annotations
import threading
import types
from typing import TYPE_CHECKING, Union, TypeAlias, Optional, Callable, Any, Dict, Tuple

import psutil

from .task import Task, Result

if TYPE_CHECKING:
    import multiprocessing
    try:
        import torch
        Queue:TypeAlias = Union[torch.multiprocessing.Queue, multiprocessing.Queue]
    except ImportError:
        Queue:TypeAlias = multiprocessing.Queue


class Worker:

    def __init__(
        self, index:int, result_queue:Queue[Result], mp_context:str,
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]],
        torch:bool=False
    ):
        if torch:
            import torch.multiprocessing as mp
        else:
            import multiprocessing as mp

        self.is_torch:bool = torch
        self.index:int = index
        self.mp_context:str = mp_context
        self.result_queue:Queue[Result] = result_queue
        self.task_queue:Queue[Task] = mp.Queue()
        self.change_device_cmd_queue:Queue = mp.Queue()
        self.is_working:bool = False
        self.module_sizes:Dict[str, int] = {}
        self.n_finished_tasks:int = 0
        self.initializer:Optional[Callable[..., Any]] = initializer
        self.initargs:Tuple[Any, ...] = initargs
        self.initkwargs:Optional[Dict[str, Any]] = initkwargs

        self.start()

    @property
    def rss(self)->int:
        p = psutil.Process(self.process.pid)
        return p.memory_info().rss

    def overlap_modules_size(self, task:Task)->int:
        result = 0
        if len(task.module_sizes) < len(self.module_sizes):
            less_module_sizes = task.module_sizes
            more_module_sizes = self.module_sizes
        else:
            less_module_sizes = self.module_sizes
            more_module_sizes = task.module_sizes

        for module_name, module_size in less_module_sizes.items():
            if module_name in more_module_sizes:
                result += module_size

        return result

    def add_task(self, task:Task)->None:
        self.is_working = True
        self.module_sizes.update(task.module_sizes)
        task.mem_before_enter = self.rss
        self.task_queue.put(task)

    def change_device(self, device:str)->None:
        self.change_device_cmd_queue.put(device)

    def stop(self)->None:
        self.task_queue.put(None)
        self.change_device_cmd_queue.put(None)

        self.task_queue.close()
        self.change_device_cmd_queue.close()

    def start(self):
        if self.is_torch:
            import torch.multiprocessing as mp
            self.process:mp.Process = mp.Process(
                target=Worker.run,
                args=(self.task_queue, self.result_queue, self.change_device_cmd_queue),
                kwargs={"initializer": self.initializer, "initargs": self.initargs, "initkwargs": self.initkwargs},
                name=f"SmartProcessPool.worker:{self.index}",
                daemon=True
            )
        else:
            import multiprocessing as mp
            ctx = mp.get_context(self.mp_context)
            self.process:mp.Process = ctx.Process(
                target=Worker.run,
                args=(self.task_queue, self.result_queue, self.change_device_cmd_queue),
                kwargs={"initializer": self.initializer, "initargs": self.initargs, "initkwargs": self.initkwargs},
                name=f"SmartProcessPool.worker:{self.index}",
                daemon=True
            )

        self.process.start()

    def restart(self)->None:
        self.task_queue.put(None)
        self.change_device_cmd_queue.put(None)
        self.process.join()
        self.n_finished_tasks:int = 0
        self.start()
        
    def terminate(self)->None:
        self.process.terminate()

    def join(self)->None:
        self.process.join()

    current_func = None
    func_device_lock = None
    change_device_thread = None

    @staticmethod
    def _changing_device(cmd_queue:Queue[Optional[str]]):
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
        task_queue:Queue[Task], result_queue:Queue[Result], change_device_cmd_queue:Queue[Optional[str]],
        initializer:Optional[Callable[..., Any]], initargs:Tuple[Any, ...], initkwargs:Optional[Dict[str, Any]]    
    ):
        if initializer is not None:
            if initkwargs is None:
                initkwargs = {}

            initializer(*initargs, **initkwargs)
        
        Worker.func_device_lock = threading.Lock()
        Worker.change_device_thread = threading.Thread(target=Worker._changing_device, args=(change_device_cmd_queue,), daemon=True)
        Worker.change_device_thread.start()

        def device(self):
            with Worker.func_device_lock:
                return self._device

        while True:
            task:Task = task_queue.get()
            if task is None:
                task_queue.close()
                result_queue.close()
                break

            try:
                with Worker.func_device_lock:
                    task.func._device = task.device
                    task.func.device = types.MethodType(device, task.func)
                    Worker.current_func = task.func
            except AttributeError:
                pass

            result_queue.put(task.exec())