from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch

def move_state_dict_to(obj, device):
    import torch
    import numpy as np

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

def move_optimizer_to(optimizer:torch.optim.Optimizer, device):
    for state in optimizer.state.values():
        move_state_dict_to(state, device)

    return optimizer
