import sys
import warnings
import functools
import itertools
from types import ModuleType
from typing import Iterable, Set


def _good_module_name(module_name:str, module_names:Set[str])->bool:
    return (
        module_name and 
        module_name not in module_names and
        not module_name.startswith('builtins') and
        module_name in sys.modules
    )

def _get_module_deps(module:ModuleType)->Set[str]:
    module_names = {module.__name__}
    stack = [module]

    while stack:
        current_module = stack.pop()
        
        for obj in current_module.__dict__.values():
            candidate_name = ""
            if isinstance(obj, ModuleType) and hasattr(obj, '__name__'):
                candidate_name = obj.__name__
            
            elif hasattr(obj, '__module__'):
                candidate_name = obj.__module__
            
            elif hasattr(obj, '__class__') and hasattr(obj.__class__, '__module__'):
                candidate_name = obj.__class__.__module__

            if _good_module_name(candidate_name, module_names):
                sub_module = sys.modules[candidate_name]
                module_names.add(candidate_name)
                stack.append(sub_module)
    
    return module_names


@functools.lru_cache(maxsize=None)
def get_module_deps(module:ModuleType)->Set[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _get_module_deps(module)


def batched(iterable:Iterable, chunksize:int):
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, chunksize))
        if not batch:
            break
        yield batch
