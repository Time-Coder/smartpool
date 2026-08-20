import queue
import threading
from collections import defaultdict

import torch
from data_utils import preprocess_audio
from model_utils import train_single_fold
from progress_info import ProgressInfo


def train_sequentially(meta_rows, train_splits, model_classes, progress_info: ProgressInfo, max_workers):
    progress_queue = queue.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tracker = threading.Thread(target=progress_info.track, args=(progress_queue,), daemon=True)
    tracker.start()

    # Phase 1: preprocess every audio file sequentially
    for row in meta_rows:
        preprocess_audio(row['filename'], row['audio'], progress_queue)

    # Phase 2: train one model per fold sequentially
    model_results = defaultdict(list)
    i = 0
    for fold_idx, train_meta, val_meta in train_splits:
        for model_class in model_classes:
            r = train_single_fold(fold_idx, model_class, train_meta, val_meta, progress_queue, best_device if i % max_workers < 5 else 'cpu')
            model_results[r.model_name].append(r.val_accuracy)
            i += 1

    tracker.join()
    return model_results