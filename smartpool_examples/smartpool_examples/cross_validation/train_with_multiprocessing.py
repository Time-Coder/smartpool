from collections import defaultdict

import torch
from model_utils import preprocess_fold, train_single_fold
from progress_info import ProgressInfo


def train_with_multiprocessing(task_templates, max_workers, progress_info: ProgressInfo):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pool = mp.Pool(processes=max_workers)

    preprocess_futures = {}
    for task_args in task_templates:
        fold_idx = task_args[0]
        model_class = task_args[1]
        future = pool.apply_async(preprocess_fold, args=task_args)
        task_key = f"{model_class.__name__}_fold_{fold_idx}"
        preprocess_futures[task_key] = future

    preprocessed = {}
    for task_key, future in preprocess_futures.items():
        fold_idx, model_class, train_loader, val_loader = future.get()
        preprocessed[task_key] = (fold_idx, model_class, train_loader, val_loader)

    futures_map = {}
    for i, (task_key, (fold_idx, model_class, train_loader, val_loader)) in enumerate(preprocessed.items()):
        future = pool.apply_async(train_single_fold, args=(fold_idx, model_class, train_loader, val_loader, progress_queue, best_device if i % max_workers < 5 else 'cpu'))
        futures_map[task_key] = future

    progress_info.track(progress_queue)

    model_results = defaultdict(list)
    for future in futures_map.values():
        result = future.get()
        model_results[result.model_name].append(result.val_accuracy)

    pool.close()
    pool.join()
    return model_results