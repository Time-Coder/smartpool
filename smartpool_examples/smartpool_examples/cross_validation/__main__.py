if __name__ == "__main__":
    import os
    import sys
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)


from smartpool import (
    DataSize,
    ProcessPool,
    Resource,
    ThreadPool,
    limit_num_single_thread,
)

limit_num_single_thread()

from typing import Literal, TypeAlias

import typer

app = typer.Typer(help="Use smartpool to do 5-fold cross validatation for 7 deep learning models for handwritten digit recognition task.")

PoolChoice: TypeAlias = Literal[
    "smartpool.ProcessPool",
    "smartpool.ThreadPool",
    "multiprocessing.Pool",
    "concurrent.futures.ProcessPoolExecutor",
    "concurrent.futures.ThreadPoolExecutor",
    "joblib.Parallel(backend='loky')",
    "joblib.Parallel(backend='threading')",
    "ray"
]


@app.command()
def main(
    pool: PoolChoice = typer.Option(
        "smartpool.ProcessPool",
        "--pool",
        help="choose process pool implementations"
    ),
    max_workers: int = typer.Option(
        0,
        "--max_workers",
        help="max number of workers to use, 0 to use all available cores"
    )
):
    import importlib
    import os
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

    print(f"Use {pool} to do 5-fold cross validatation for 7 deep learning models for handwritten digit recognition task.")
    print("Use `python -m smartpool_examples.cross_validation --help` to see all options.")
    print(f"See source code at folder {os.path.dirname(os.path.abspath(__file__))}")
    print("\npreparing data...")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is not installed. Follow https://pytorch.org/ instructions to install PyTorch.")
        exit(1)

    if importlib.util.find_spec("torchvision") is None:
        print("torchvision is not installed. Use `pip install torchvision` to install torchvision.")
        exit(1)

    import multiprocessing as mp
    import os
    import queue
    import sys
    import time
    from collections import defaultdict
    from concurrent.futures import Future
    from typing import Dict, Union

    import numpy as np
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
    from sklearn.model_selection import KFold

    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    sys.path.append(self_folder)

    import models
    from config import EPOCHS
    from data_utils import prepare_data
    from model_utils import ErrorInfo, ProgressInfo, TrainingResult, train_single_fold
    from visualization import plot_results, print_results_table

    if max_workers == 0:
        max_workers = os.cpu_count()

    model_classes = [
        cls for cls in models.__dict__.values()
        if isinstance(cls, type) and issubclass(cls, nn.Module) and cls != nn.Module
    ]

    dataset = prepare_data()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    manager = mp.Manager()

    if pool != "ray":
        progress_queue:queue.Queue[Union[ProgressInfo, ErrorInfo]] = manager.Queue()
    else:
        try:
            import ray
            import ray.util.queue
        except ImportError:
            print("Ray is not installed. Use `pip install ray` to install Ray.")
            exit(1)

        progress_queue:queue.Queue[Union[ProgressInfo, ErrorInfo]] = ray.util.queue.Queue()

    tasks = []
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        for model_class in model_classes:
            tasks.append((fold_idx, model_class, train_indices.copy(), val_indices.copy(), dataset, progress_queue))

    task_progress_bars = {}
    best_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = time.perf_counter()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    ) as progress:

        active_tasks = {}

        if pool == "smartpool.ProcessPool":
            process_pool = ProcessPool(max_workers=max_workers, use_torch=True)
        elif pool == "smartpool.ThreadPool":
            process_pool = ThreadPool(max_workers=max_workers, use_torch=True)
        elif pool == "concurrent.futures.ProcessPoolExecutor":
            from concurrent.futures import ProcessPoolExecutor
            process_pool = ProcessPoolExecutor(max_workers=max_workers)
        elif pool == "concurrent.futures.ThreadPoolExecutor":
            from concurrent.futures import ThreadPoolExecutor
            process_pool = ThreadPoolExecutor(max_workers=max_workers)
        elif pool == "multiprocessing.Pool":
            import multiprocessing
            process_pool = multiprocessing.Pool(processes=max_workers)
        elif pool == "joblib.Parallel(backend='loky')":
            from joblib import Parallel, delayed
            process_pool = Parallel(n_jobs=max_workers, backend='loky', return_as="generator")
        elif pool == "joblib.Parallel(backend='threading')":
            from joblib import Parallel, delayed
            process_pool = Parallel(n_jobs=max_workers, backend='threading', return_as="generator")

        print("submitting training tasks...")
        futures_map:Dict[str, Future] = {}
        futures = []
        for i, task_args in enumerate(tasks):
            if pool.startswith("smartpool."):
                future = process_pool.submit(
                    train_single_fold,
                    args=task_args,
                    cpu_mode_res=Resource(
                        cpu_cores=1,
                        cpu_mem=1.1*DataSize.GB
                    ),
                    gpu_mode_res=Resource(
                        cpu_cores=1,
                        cpu_mem=1.1*DataSize.GB,
                        gpu_cores=1000,
                        gpu_mem=0.2*DataSize.GB
                    )
                )
            elif pool.startswith("concurrent.futures."):
                future = process_pool.submit(train_single_fold, *task_args, best_device if i % max_workers < 5 else 'cpu')
            elif pool == "multiprocessing.Pool":
                future = process_pool.apply_async(train_single_fold, args=(*task_args, best_device if i % max_workers < 5 else 'cpu'))
            elif pool.startswith("joblib"):
                future = delayed(train_single_fold)(*task_args, best_device if i % max_workers < 5 else 'cpu')
            elif pool == "ray":
                future = ray.remote(num_cpus=1, num_gpus=(0.2 if i % max_workers < 5 else 0), memory=1.1*DataSize.GB)(train_single_fold).remote(*task_args, best_device if i % max_workers < 5 else 'cpu')

            fold_idx = task_args[0]
            model_class = task_args[1]
            model_name = model_class.__name__
            task_key = f"{model_name}_fold_{fold_idx}"
            futures_map[task_key] = future
            futures.append(future)

        print(f"training all models in {pool} ...")
        if pool.startswith("joblib"):
            joblib_results = process_pool(futures)

        finished_tasks = set()
        while True:
            progress_info:Union[ProgressInfo, ErrorInfo] = progress_queue.get()
            if isinstance(progress_info, ErrorInfo):
                print(progress_info.traceback)
                break

            task_key = f"{progress_info.model_name}_fold_{progress_info.fold_idx}"

            if task_key not in task_progress_bars:
                initial_desc = f"train {progress_info.model_name} on {progress_info.device} "
                initial_desc += f"for fold {progress_info.fold_idx+1}/5"
                task_progress_bars[task_key] = progress.add_task(initial_desc, total=100)
                active_tasks[task_key] = True

            if task_key in task_progress_bars:
                epoch_progress = (progress_info.epoch - 1) / 5
                batch_progress = progress_info.batch / progress_info.total_batches
                total_progress = (epoch_progress + batch_progress / 5) * 100

                if progress_info.epoch == 5 and progress_info.batch == progress_info.total_batches:
                    total_progress = 100.0
                    finished_tasks.add(task_key)

                new_desc = f"train {progress_info.model_name} on {progress_info.device} "
                new_desc += f"for fold {progress_info.fold_idx+1}/5 - Epoch {progress_info.epoch}/{EPOCHS} "
                new_desc += f"Loss: {progress_info.avg_loss:.4f} "
                new_desc += f"Val Acc: {progress_info.val_accuracy*100:.2f}%"
                if progress_info.device.startswith("cuda"):
                    new_desc = "[bright_cyan]" + new_desc

                progress.update(
                    task_progress_bars[task_key],
                    completed=total_progress,
                    description=new_desc
                )
                if total_progress >= 100.0:
                    progress.update(task_progress_bars[task_key], visible=False)

            if len(finished_tasks) == len(futures_map):
                break

    model_results = defaultdict(list)
    if pool in ["smartpool.ProcessPool", "smartpool.ThreadPool", "concurrent.futures.ProcessPoolExecutor", "concurrent.futures.ThreadPoolExecutor", "multiprocessing.Pool"]:
        for future in futures_map.values():
            if pool == "multiprocessing.Pool":
                result:TrainingResult = future.get()
            else:
                result:TrainingResult = future.result()

            model_results[result.model_name].append(result.val_accuracy)
    elif pool.startswith("joblib"):
        for result in joblib_results:
            model_results[result.model_name].append(result.val_accuracy)
    elif pool == "ray":
        ray_results = ray.get(futures)
        for result in ray_results:
            model_results[result.model_name].append(result.val_accuracy)

    stop_time = time.perf_counter()
    print(f"train completed in {stop_time - start_time:.2f} seconds")

    print("analysing results...")

    stats = {}
    for model_name, accuracies in model_results.items():
        stats[model_name] = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'accuracies': accuracies
        }

    print_results_table(stats)
    plot_results(model_results, stats)


if __name__ == "__main__":
    app()
