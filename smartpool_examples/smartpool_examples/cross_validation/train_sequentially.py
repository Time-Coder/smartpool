import queue
import threading
from collections import defaultdict

import torch
from model_utils import preprocess_fold, train_single_fold
from progress_info import ProgressInfo


def train_sequentially(task_templates, max_workers, progress_info: ProgressInfo):
    progress_queue = queue.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tracker = threading.Thread(target=progress_info.track, args=(progress_queue,), daemon=True)
    tracker.start()

    preprocessed = []
    for task_args in task_templates:
        preprocessed.append(preprocess_fold(*task_args))

    model_results = defaultdict(list)
    for i, (fold_idx, model_class, train_loader, val_loader) in enumerate(preprocessed):
        r = train_single_fold(fold_idx, model_class, train_loader, val_loader, progress_queue, best_device if i % max_workers < 5 else 'cpu')
        model_results[r.model_name].append(r.val_accuracy)

    tracker.join()
    return model_results