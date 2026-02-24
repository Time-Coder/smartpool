import uuid
import sys
from typing import Tuple, Any, Dict, Optional

from .utils import get_module_sizes


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
        self.estimated_need_cpu_mem:float = 0.0
        self.modules_overlap_size:int = 0.0
        self.module_sizes:Dict[str, int] = get_module_sizes(sys.modules[func.__module__], need_cpu_mem > 0)
        self.device:Optional[str] = None
        self.worker_index:int = -1
        self.mem_before_enter:int = 0

    def info(self):
        return self.id, self.func, self.args, self.kwargs

    @property
    def gpu_id(self)->int:
        if isinstance(self.device, str) and self.device.startswith("cuda:"):
            return int(self.device[len("cuda:"):])
        
        return -1
