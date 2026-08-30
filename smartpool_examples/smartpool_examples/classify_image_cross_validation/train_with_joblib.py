from collections import defaultdict
import threading

import torch
from .model_utils import train_single_fold
from .progress_info import ProgressInfo


def _reap_children():
    """Synchronously kill any still-alive child processes so their
    (model/GPU) memory is released before the next pool runs in the same
    benchmark process."""
    import multiprocessing as mp

    for child in mp.active_children():
        child.terminate()
    for child in mp.active_children():
        child.join(timeout=5)


def train_with_joblib(task_templates, max_workers, progress_info: ProgressInfo):
    import multiprocessing as mp
    from joblib import Parallel, delayed

    manager = mp.Manager()
    progress_queue = manager.Queue()
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tasks = [(*t, progress_queue, best_device) for t in task_templates]
    delayed_func = delayed(train_single_fold)
    with Parallel(n_jobs=max_workers, backend="loky") as parallel:
        tracker = threading.Thread(
            target=progress_info.track, args=(progress_queue,), daemon=True
        )
        tracker.start()
        futures = []
        for task_args in tasks:
            futures.append(delayed_func(*task_args))

        joblib_results = parallel(futures)

        tracker.join()

        model_results = defaultdict(list)
        for result in joblib_results:
            model_results[result.model_name].append(result.val_accuracy)
        
    manager.shutdown()
    try:
        parallel._backend.terminate()
    except Exception:
        pass
    _reap_children()

    return model_results