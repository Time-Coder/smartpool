from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from optimizer import optimize_single
from progress_info import ProgressInfo


def optimize_with_concurrent(task_templates, max_workers, progress_info: ProgressInfo):
    manager = mp.Manager()
    progress_queue = manager.Queue()
    tasks = [(*t, progress_queue) for t in task_templates]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures_map = {}
        for task_args in tasks:
            future = pool.submit(optimize_single, *task_args)
            task_key = f"{task_args[0]}_{task_args[1]}_run_{task_args[2]}"
            futures_map[task_key] = future

        progress_info.track(progress_queue)

        results = defaultdict(lambda: defaultdict(list))
        for future in futures_map.values():
            r = future.result()
            results[r.func_name][r.algo_name].append(r)

    return results
