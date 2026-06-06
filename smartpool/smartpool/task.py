from __future__ import annotations
import uuid
import sys
from concurrent.futures import Future
from typing import Tuple, Any, Dict, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .worker import Worker
    from .utils import Resource


class Task:

    def __init__(self, func:Callable[..., Any], args:Tuple[Any, ...], kwargs:Dict[str, Any],
                 cpu_mode_res: Resource, gpu_mode_res: Resource, calculate_module_deps:bool):
        self.id:str = str(uuid.uuid4())
        self.func:Callable[..., Any] = func
        self.args:Tuple[Any] = args
        self.kwargs:Dict[str, Any] = kwargs
        self.cpu_mode_res: Resource = cpu_mode_res
        self.gpu_mode_res: Resource = gpu_mode_res
        self.estimated_need_cpu_mem:float = 0.0
        self.modules_overlap_ratio:float = 0.0
        self.module_deps:Dict[str, int] = {}
        
        if calculate_module_deps:
            from .module_deps import module_deps
            self.module_deps:Dict[str, int] = module_deps(sys.modules[func.__module__])
        
        self.device:Optional[str] = None
        self.worker:Worker = None
        self.mem_before_enter:int = 0
        self.future = Future()

    @property
    def effective_res(self) -> Resource:
        if self.device and self.device != "cpu":
            return self.gpu_mode_res
        return self.cpu_mode_res

    def info(self):
        return self.id, self.device, self.func, self.args, self.kwargs

    @property
    def gpu_id(self)->int:
        if isinstance(self.device, str) and ":" in self.device:
            parts = self.device.split(":")
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except (ValueError, IndexError):
                    pass
        return -1

    def exec(self)->Tuple[bool, Any]:
        try:
            result = self.func(*self.args, **self.kwargs)
            success = True
        except BaseException as e:
            result = e
            success = False

        return success, result