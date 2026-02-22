from __future__ import annotations
import sys
import gc
from .asizeof import asizeof
import warnings
import functools
from enum import IntEnum
import itertools
from multiprocessing.connection import _ConnectionBase
from types import ModuleType
from typing import Set, Dict, Iterable


class DataSize(IntEnum):
    B = 1
    KB = 1024
    MB = 1024 ** 2
    GB = 1024 ** 3
    TB = 1024 ** 4
    PB = 1024 ** 5
    

asizeof = functools.lru_cache(maxsize=None)(asizeof)
sys_size = asizeof(sys)

def _good_module_name(module_name:str, module_sizes:Dict[str, int])->bool:
    return (
        module_name not in module_sizes and
        not module_name.startswith('builtins') and
        module_name in sys.modules
    )

def _get_module_sizes(module:ModuleType, module_sizes:Dict[str, int])->Dict[str, int]:
    if module.__name__ in module_sizes:
        return module_sizes
    
    module_sizes[module.__name__] = asizeof(module)

    referents = gc.get_referents(module.__dict__)

    module_names:Set[str] = set()

    for obj in referents:
        if isinstance(obj, ModuleType) and _good_module_name(obj.__name__, module_sizes):
            module_names.add(obj.__name__)
        
        elif hasattr(obj, '__module__') and obj.__module__:
            if _good_module_name(obj.__module__, module_sizes):
                module_names.add(obj.__module__)
        
        elif hasattr(obj, '__class__'):
            cls = obj.__class__
            if hasattr(cls, '__module__') and cls.__module__:
                if _good_module_name(cls.__module__, module_sizes):
                    module_names.add(cls.__module__)

    for module_name in module_names:
        _get_module_sizes(sys.modules[module_name], module_sizes)

    return module_sizes

@functools.lru_cache(maxsize=None)
def get_module_sizes(module:ModuleType)->Dict[str, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _get_module_sizes(module, {"sys": sys_size})


def batched(iterable:Iterable, chunksize:int):
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, chunksize))
        if not batch:
            break
        yield batch

def comm_put(queue, item)->None:
    if isinstance(queue, _ConnectionBase):
        queue.send(item)
    else:
        queue.put(item)

def comm_get(queue)->Any:
    if isinstance(queue, _ConnectionBase):
        return queue.recv()
    else:
        return queue.get()