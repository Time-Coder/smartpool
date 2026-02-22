import uuid
import sys
from typing import Tuple, Any, Dict, Optional

from .utils import get_module_sizes


class Result:

    def __init__(self, task_id:str):
        self.task_id:str = task_id
        self.success:bool = False
        self.result = None


class Task:

    def __init__(self, func, args, kwargs, need_cpu_cores, need_cpu_mem, need_gpu_cores, need_gpu_mem):
        self.id:str = str(uuid.uuid4())
        self.func = func
        self.args:Tuple[Any] = args
        self.kwargs:Dict[str, Any] = kwargs
        self.need_cpu_cores:int = need_cpu_cores
        self.need_cpu_mem:int = need_cpu_mem
        self.need_gpu_cores:int = need_gpu_cores
        self.need_gpu_mem:int = need_gpu_mem
        self.module_sizes:Dict[str, int] = get_module_sizes(sys.modules[func.__module__])
        self.device:Optional[str] = None
        self.worker_index:int = -1
        self.mem_before_enter:int = 0

    def __getstate__(self):
        return {
            "id": self.id,
            "func": self.func,
            "args": self.args,
            "kwargs": self.kwargs
        }

    @property
    def gpu_id(self)->int:
        if isinstance(self.device, str) and self.device.startswith("cuda:"):
            return int(self.device[len("cuda:"):])
        
        return -1

    def exec(self)->Result:
        result = Result(self.id)

        try:
            result.result = self.func(*self.args, **self.kwargs)
            result.success = True
        except BaseException as e:
            result.result = e
            result.success = False
            
        return result