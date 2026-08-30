import queue
import threading
from collections import defaultdict

import torch
from .model_utils import train_single_fold
from .progress_info import ProgressInfo


def train_sequentially(task_templates, max_workers, progress_info: ProgressInfo):
    progress_queue = queue.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tasks = [(*t, progress_queue, best_device) for t in task_templates]

    tracker = threading.Thread(target=progress_info.track, args=(progress_queue,), daemon=True)
    tracker.start()

    model_results = defaultdict(list)
    for task_args in tasks:
        r = train_single_fold(*task_args)
        model_results[r.model_name].append(r.val_accuracy)

    tracker.join()
    return model_results