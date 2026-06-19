from collections import defaultdict

import ray
import ray.util.queue
from smartpool import DataSize

from optimizer import optimize_single
from progress_info import ProgressInfo


def optimize_with_ray(task_templates, max_workers, progress_info: ProgressInfo):
    progress_queue = ray.util.queue.Queue()
    tasks = [(*t, progress_queue) for t in task_templates]

    futures = []
    futures_map = {}
    for task_args in tasks:
        future = ray.remote(
            num_cpus=1,
            memory=200 * DataSize.MB,
        )(optimize_single).remote(*task_args, 'cpu')
        task_key = f"{task_args[0]}_{task_args[1]}_run_{task_args[2]}"
        futures_map[task_key] = future
        futures.append(future)

    progress_info.track(progress_queue)

    ray_results = ray.get(futures)
    results = defaultdict(lambda: defaultdict(list))
    for r in ray_results:
        results[r.func_name][r.algo_name].append(r)

    return results
