from collections import defaultdict

import torch

from model_utils import train_single_fold
from progress_info import ProgressInfo


def train_with_joblib(task_templates, max_workers, pool_type, progress_info: ProgressInfo):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tasks = [(*t, progress_queue) for t in task_templates]

    from joblib import Parallel, delayed

    backend = 'loky' if 'loky' in pool_type else 'threading'
    pool = Parallel(n_jobs=max_workers, backend=backend, return_as="generator")

    futures_map = {}
    futures = []
    for i, task_args in enumerate(tasks):
        future = delayed(train_single_fold)(*task_args, best_device if i % max_workers < 5 else 'cpu')
        fold_idx = task_args[0]
        model_class = task_args[1]
        task_key = f"{model_class.__name__}_fold_{fold_idx}"
        futures_map[task_key] = future
        futures.append(future)

    joblib_results = pool(futures)

    progress_info.track(progress_queue)

    model_results = defaultdict(list)
    for result in joblib_results:
        model_results[result.model_name].append(result.val_accuracy)

    return model_results
