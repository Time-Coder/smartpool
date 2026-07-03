from collections import defaultdict

from optimizer import optimize_single
from progress_info import ProgressInfo

from smartpool import DataSize, ProcessPool, Resource


def optimize_with_smartpool(task_templates, max_workers, progress_info: ProgressInfo):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()
    tasks = [(*t, progress_queue) for t in task_templates]

    with ProcessPool(max_workers=max_workers, use_torch=False) as pool:
        futures_map = {}
        for task_args in tasks:
            future = pool.submit(
                optimize_single,
                args=task_args,
                cpu_mode_res=Resource(cpu_cores=1, cpu_mem=122 * DataSize.MB),
            )
            task_key = f"{task_args[0]}_{task_args[1]}_run_{task_args[2]}"
            futures_map[task_key] = future

        progress_info.track(progress_queue)

        results = defaultdict(lambda: defaultdict(list))
        for future in futures_map.values():
            r = future.result()
            results[r.func_name][r.algo_name].append(r)

    return results
