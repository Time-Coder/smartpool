from collections import defaultdict

import ray
import torch
from progress_info import ProgressInfo

from smartpool import DataSize


@ray.remote(num_cpus=1, memory=1.1 * DataSize.GB)
def train_fold_task(fold_idx, model_class, train_indices, val_indices, dataset, progress_queue):
    from model_utils import train_single_fold
    gpu_ids = ray.get_gpu_ids()
    device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
    return train_single_fold(fold_idx, model_class, train_indices, val_indices, dataset, progress_queue, device)


def train_with_ray(task_templates, max_workers, progress_info: ProgressInfo):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    has_gpu = torch.cuda.is_available()
    progress_queue = ray.util.queue.Queue()

    tasks = [(*t, progress_queue) for t in task_templates]

    futures = {}
    for task_args in tasks:
        fold_idx, model_class = task_args[0], task_args[1]
        fut = train_fold_task.options(
            num_gpus=0.2 if has_gpu else 0
        ).remote(*task_args)
        task_key = f"{model_class.__name__}_fold_{fold_idx}"
        futures[task_key] = fut

    progress_info.track(progress_queue)

    model_results = defaultdict(list)
    for future in futures.values():
        result = ray.get(future)
        model_results[result.model_name].append(result.val_accuracy)

    ray.shutdown()
    return model_results
