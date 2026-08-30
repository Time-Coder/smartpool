from collections import defaultdict
import multiprocessing as mp
import torch
from .model_utils import train_single_fold
from .progress_info import ProgressInfo


def train_with_multiprocessing(task_templates, max_workers, progress_info: ProgressInfo):
    manager = mp.Manager()
    progress_queue = manager.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tasks = [(*t, progress_queue, best_device) for t in task_templates]

    pool = mp.Pool(processes=max_workers)

    futures_map = {}
    for task_args in tasks:
        fold_idx = task_args[0]
        model_class = task_args[1]
        task_key = f"{model_class.__name__}_fold_{fold_idx}"
        futures_map[task_key] = pool.apply_async(train_single_fold, args=task_args)

    progress_info.track(progress_queue)

    model_results = defaultdict(list)
    for task_key, future in futures_map.items():
        result = future.get()
        model_results[result.model_name].append(result.val_accuracy)

    pool.close()
    pool.join()
    manager.shutdown()
    return model_results