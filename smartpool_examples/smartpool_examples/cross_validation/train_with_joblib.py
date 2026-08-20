from collections import defaultdict

import torch
from data_utils import preprocess_audio
from model_utils import train_single_fold
from progress_info import ProgressInfo


def train_with_joblib(meta_rows, train_splits, model_classes, progress_info: ProgressInfo, max_workers):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from joblib import Parallel, delayed

    # Phase 1: preprocess every audio file
    pool = Parallel(n_jobs=max_workers, backend='loky', return_as="generator")
    preprocess_futures = [delayed(preprocess_audio)(row['filename'], row['audio'], progress_queue) for row in meta_rows]
    for _ in pool(preprocess_futures):
        pass

    # Phase 2: train one model per fold
    pool = Parallel(n_jobs=max_workers, backend='loky', return_as="generator")
    futures_map = {}
    i = 0
    train_futures = []
    for fold_idx, train_meta, val_meta in train_splits:
        for model_class in model_classes:
            future = delayed(train_single_fold)(fold_idx, model_class, train_meta, val_meta, progress_queue, best_device if i % max_workers < 5 else 'cpu')
            futures_map[f"{model_class.__name__}_fold_{fold_idx}"] = future
            train_futures.append(future)
            i += 1

    progress_info.track(progress_queue)

    model_results = defaultdict(list)
    for result in pool(train_futures):
        model_results[result.model_name].append(result.val_accuracy)

    return model_results