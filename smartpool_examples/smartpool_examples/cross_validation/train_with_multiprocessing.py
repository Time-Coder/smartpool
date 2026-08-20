from collections import defaultdict

import torch
from data_utils import preprocess_audio
from model_utils import train_single_fold
from progress_info import ProgressInfo


def train_with_multiprocessing(meta_rows, train_splits, model_classes, progress_info: ProgressInfo, max_workers):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pool = mp.Pool(processes=max_workers)

    # Phase 1: preprocess every audio file
    preprocess_futures = [pool.apply_async(preprocess_audio, args=(row['filename'], row['audio'], progress_queue)) for row in meta_rows]
    for fut in preprocess_futures:
        fut.get()

    # Phase 2: train one model per fold
    futures_map = {}
    i = 0
    for fold_idx, train_meta, val_meta in train_splits:
        for model_class in model_classes:
            future = pool.apply_async(train_single_fold, args=(fold_idx, model_class, train_meta, val_meta, progress_queue, best_device if i % max_workers < 5 else 'cpu'))
            futures_map[f"{model_class.__name__}_fold_{fold_idx}"] = future
            i += 1

    progress_info.track(progress_queue)

    model_results = defaultdict(list)
    for future in futures_map.values():
        result = future.get()
        model_results[result.model_name].append(result.val_accuracy)

    pool.close()
    pool.join()
    return model_results