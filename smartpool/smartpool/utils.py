from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, TypeVar
import os

if TYPE_CHECKING:
    from torch.cuda import Stream
    from .device import Device

TRT_CACHE_PATH: str = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/__trtcache__"

T = TypeVar('T')
class QueueLike(Protocol[T]):
    def put(task, item: T) -> None: ...
    def get(task) -> T: ...


_has_gil = None

def has_gil()->bool:
    global _has_gil
    if _has_gil is None:
        import sys
        try:
            _has_gil = sys._is_gil_enabled()
        except Exception:
            _has_gil = True

    return _has_gil

def limit_num_single_thread():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def move_state_dict_to(obj, device):
    import numpy as np
    import torch

    base_types = (int, float, complex, bool, str, bytes, type(None), np.ndarray, np.bool)

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, base_types):
        return obj
    elif isinstance(obj, list):
        for i in range(len(obj)):
            sub_obj = obj[i]
            new_sub_obj = move_state_dict_to(sub_obj, device)
            if sub_obj is not new_sub_obj:
                obj[i] = new_sub_obj

        return obj
    elif isinstance(obj, tuple):
        if not all(isinstance(x, base_types) for x in obj):
            return tuple(move_state_dict_to(list(obj), device))
        else:
            return obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            new_v = move_state_dict_to(v, device)
            if new_v is not v:
                obj[k] = new_v

        return obj
    else:
        raise TypeError(f"{type(obj)} is not supported")


def move_optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        move_state_dict_to(state, device)

    return optimizer


def batched(iterable, chunksize:int):
    import itertools
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, chunksize))
        if not batch:
            break
        yield batch


_best_device_lock = None
_best_device = {}
_best_stream = {}

def _set_best_device(device:Device, tid=None)->None:
    import threading

    global _best_device_lock
    if _best_device_lock is None:
        _best_device_lock = threading.Lock()

    if tid is None:
        tid = threading.get_ident()

    with _best_device_lock:
        _best_device[tid] = device

def best_device()->Device:
    import threading

    global _best_device_lock
    if _best_device_lock is None:
        _best_device_lock = threading.Lock()

    tid = threading.get_ident()
    with _best_device_lock:
        return _best_device[tid]

def _set_best_stream(stream:Optional[Stream], tid=None)->None:
    import threading

    global _best_device_lock
    if _best_device_lock is None:
        _best_device_lock = threading.Lock()

    if tid is None:
        tid = threading.get_ident()

    with _best_device_lock:
        _best_stream[tid] = stream

def best_stream()->Optional[Stream]:
    import threading

    global _best_device_lock
    if _best_device_lock is None:
        _best_device_lock = threading.Lock()

    tid = threading.get_ident()
    with _best_device_lock:
        return _best_stream.get(tid)
