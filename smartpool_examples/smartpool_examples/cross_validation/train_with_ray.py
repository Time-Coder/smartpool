from collections import defaultdict

import torch
import ray
import ray.util.queue
from smartpool import DataSize

from model_utils import train_single_fold
from progress_info import ProgressInfo


def train_with_ray(task_templates, max_workers, progress_info: ProgressInfo):
    progress_queue = ray.util.queue.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tasks = [(*t, progress_queue) for t in task_templates]

    futures = []
    futures_map = {}
    for i, task_args in enumerate(tasks):
        future = ray.remote(
            num_cpus=1,
            num_gpus=(0.2 if i % max_workers < 5 else 0),
            memory=1.1 * DataSize.GB,
        )(train_single_fold).remote(*task_args, best_device if i % max_workers < 5 else 'cpu')
        fold_idx = task_args[0]
        model_class = task_args[1]
        task_key = f"{model_class.__name__}_fold_{fold_idx}"
        futures_map[task_key] = future
        futures.append(future)

    progress_info.track(progress_queue)

    ray_results = ray.get(futures)
    model_results = defaultdict(list)
    for result in ray_results:
        model_results[result.model_name].append(result.val_accuracy)

    return model_results
