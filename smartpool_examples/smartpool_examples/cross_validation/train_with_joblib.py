from collections import defaultdict

import torch
from model_utils import preprocess_fold, train_single_fold
from progress_info import ProgressInfo


def train_with_joblib(task_templates, max_workers, progress_info: ProgressInfo):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from joblib import Parallel, delayed

    pool = Parallel(n_jobs=max_workers, backend='loky', return_as="generator")

    preprocess_futures = []
    preprocess_keys = []
    for task_args in task_templates:
        fold_idx = task_args[0]
        model_class = task_args[1]
        task_key = f"{model_class.__name__}_fold_{fold_idx}"
        preprocess_futures.append(delayed(preprocess_fold)(*task_args))
        preprocess_keys.append(task_key)

    preprocessed = {}
    for task_key, result in zip(preprocess_keys, pool(preprocess_futures)):
        fold_idx, model_class, train_loader, val_loader = result
        preprocessed[task_key] = (fold_idx, model_class, train_loader, val_loader)

    pool = Parallel(n_jobs=max_workers, backend='loky', return_as="generator")

    futures_map = {}
    train_futures = []
    for i, (task_key, (fold_idx, model_class, train_loader, val_loader)) in enumerate(preprocessed.items()):
        future = delayed(train_single_fold)(fold_idx, model_class, train_loader, val_loader, progress_queue, best_device if i % max_workers < 5 else 'cpu')
        futures_map[task_key] = future
        train_futures.append(future)

    joblib_results = pool(train_futures)

    progress_info.track(progress_queue)

    model_results = defaultdict(list)
    for result in joblib_results:
        model_results[result.model_name].append(result.val_accuracy)

    return model_results