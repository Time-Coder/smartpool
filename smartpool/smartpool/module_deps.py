import functools
import sys
import warnings
from types import ModuleType
from typing import Set


def _good_module_name(module_name:str, module_names:Set[str])->bool:
    package_name = module_name.split('.', 1)[0] if module_name else ''
    return (
        package_name and
        package_name not in module_names and
        package_name != 'builtins' and
        package_name not in getattr(sys, 'stdlib_module_names', ()) and
        module_name in sys.modules and
        package_name in sys.modules
    )

def _module_deps(module:ModuleType)->Set[str]:
    module_names = {module.__name__}
    for obj in module.__dict__.values():
        candidate_name = ""
        if isinstance(obj, ModuleType):
            candidate_name = obj.__name__
        elif hasattr(obj, '__module__'):
            candidate_name = obj.__module__
        elif hasattr(obj, '__class__') and hasattr(obj.__class__, '__module__'):
            candidate_name = obj.__class__.__module__

        if _good_module_name(candidate_name, module_names):
            module_names.add(candidate_name.split('.', 1)[0])

    return module_names


@functools.lru_cache(maxsize=None)
def module_deps(module:ModuleType)->Set[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _module_deps(module)
