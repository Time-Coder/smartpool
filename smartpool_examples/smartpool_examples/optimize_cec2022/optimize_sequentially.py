import queue
import threading
from collections import defaultdict

from optimizer import optimize_single
from progress_info import ProgressInfo


def optimize_sequentially(task_templates, progress_info: ProgressInfo):
    progress_queue = queue.Queue()
    tasks = [(*t, progress_queue) for t in task_templates]

    tracker = threading.Thread(target=progress_info.track, args=(progress_queue,), daemon=True)
    tracker.start()

    results = defaultdict(lambda: defaultdict(list))
    for task_args in tasks:
        r = optimize_single(*task_args)
        results[r.func_name][r.algo_name].append(r)

    tracker.join()
    return results
