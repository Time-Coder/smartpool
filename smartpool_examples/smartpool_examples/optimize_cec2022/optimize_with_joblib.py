from collections import defaultdict

from joblib import Parallel, delayed
from optimizer import optimize_single
from progress_info import ProgressInfo


def optimize_with_joblib(task_templates, max_workers, progress_info: ProgressInfo):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()
    tasks = [(*t, progress_queue) for t in task_templates]

    pool = Parallel(n_jobs=max_workers, backend='loky', return_as="generator")

    futures_map = {}
    futures = []
    for task_args in tasks:
        future = delayed(optimize_single)(*task_args)
        task_key = f"{task_args[0]}_{task_args[1]}_run_{task_args[2]}"
        futures_map[task_key] = future
        futures.append(future)

    joblib_results = pool(futures)

    progress_info.track(progress_queue)

    results = defaultdict(lambda: defaultdict(list))
    for r in joblib_results:
        results[r.func_name][r.algo_name].append(r)

    return results
