from collections import defaultdict

from model_utils import train_single_fold
from progress_info import ProgressInfo

from smartpool import DataSize, ProcessPool, Resource


def train_with_smartpool(task_templates, max_workers, progress_info: ProgressInfo):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()

    tasks = [(*t, progress_queue) for t in task_templates]

    with ProcessPool(max_workers=max_workers, use_torch=True) as pool:
        futures_map = {}
        for task_args in tasks:
            future = pool.submit(
                train_single_fold,
                args=task_args,
                cpu_mode_res=Resource(cpu_cores=1, cpu_mem=500 * DataSize.MB),
                gpu_mode_res=Resource(cpu_cores=1, cpu_mem=1.1 * DataSize.GB, gpu_cores=1000, gpu_mem=0.2 * DataSize.GB),
                device_changeable=True
            )
            fold_idx = task_args[0]
            model_class = task_args[1]
            task_key = f"{model_class.__name__}_fold_{fold_idx}"
            futures_map[task_key] = future

        progress_info.track(progress_queue)

        model_results = defaultdict(list)
        for future in futures_map.values():
            result = future.result()
            model_results[result.model_name].append(result.val_accuracy)

    return model_results
