from collections import defaultdict
import time

import ray
import torch
from ray.util.queue import Queue
from .progress_info import ProgressInfo

from smartpool import DataSize


def train_with_ray(task_templates, max_workers, progress_info: ProgressInfo):
    from .model_utils import train_single_fold

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
        )

    progress_queue = Queue()
    num_gpus=0.15
    if torch.cuda.is_available():
        best_device = 'cuda'
        num_gpus=0.14
    else:
        best_device = 'cpu'
        num_gpus=0
        
    remote_decorator = ray.remote(num_cpus=1, memory=1.1 * DataSize.GB, num_gpus=num_gpus)
    remote_func = remote_decorator(train_single_fold)
    tasks = [(*t, progress_queue, best_device) for t in task_templates]

    futures = {}
    for task_args in tasks:
        fold_idx, model_class = task_args[0], task_args[1]
        task_key = f"{model_class.__name__}_fold_{fold_idx}"
        future = remote_func.remote(*task_args)
        futures[task_key] = future

    progress_info.track(progress_queue)

    model_results = defaultdict(list)
    for task_key, future in futures.items():
        result = ray.get(future)
        model_results[result.model_name].append(result.val_accuracy)

    ray.shutdown()
    return model_results