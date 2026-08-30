"""Build cross-validation task templates (importable package API).

This mirrors the template construction done in ``__main__.py`` but is exposed
as an importable function so external drivers (e.g. the §6.2 benchmark) can use
it without relying on ``sys.path`` hacks.
"""

import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from . import models
from .data_utils import prepare_data


def build_task_templates():
    model_classes = [
        cls for cls in models.__dict__.values()
        if isinstance(cls, type) and issubclass(cls, nn.Module) and cls is not nn.Module
    ]
    dataset = prepare_data()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    templates = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        for model_class in model_classes:
            templates.append(
                (fold_idx, model_class, train_idx.copy(), val_idx.copy(), dataset)
            )
    return templates
