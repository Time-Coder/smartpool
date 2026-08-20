if __name__ == "__main__":
    import os
    import sys
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)


from smartpool import limit_num_single_thread

limit_num_single_thread()

from typing import Literal, TypeAlias

import typer

app = typer.Typer(help="Use smartpool to do 5-fold cross validation for 4 deep learning models for ESC-50 audio classification task.")

PoolChoice: TypeAlias = Literal[
    "smartpool",
    "multiprocessing",
    "concurrent",
    "joblib",
    "ray",
    "sequentially"
]


@app.command()
def main(
    pool: PoolChoice = typer.Option(
        "smartpool",
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

    print(f"Use {pool} to do 5-fold cross validation for 4 deep learning models for ESC-50 audio classification task.")
    print("Use `python -m smartpool_examples.cross_validation --help` to see all options.")
    print(f"See source code at folder {os.path.dirname(os.path.abspath(__file__))}")
    print("\npreparing data...")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is not installed. Follow https://pytorch.org/ instructions to install PyTorch.")
        return

    if importlib.util.find_spec("pyarrow") is None:
        print("pyarrow is not installed. Use `pip install pyarrow` to read ESC-50 parquet data.")
        return

    import os
    import sys
    import time

    import numpy as np
    from config import N_FOLDS

    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    sys.path.append(self_folder)

    import models
    from data_utils import prepare_metadata
    from progress_info import ProgressInfo
    from visualization import plot_results, print_results_table

    if max_workers == 0:
        max_workers = os.cpu_count() // 2

    model_classes = [
        cls for cls in models.__dict__.values()
        if isinstance(cls, type) and issubclass(cls, nn.Module) and cls != nn.Module
    ]

    meta_rows = prepare_metadata()

    # ESC-50 official fold split: fold 1..5, each fold has 400 samples (50 classes x 8)
    train_splits = []
    for fold_idx in range(1, N_FOLDS + 1):
        train_meta = [row for row in meta_rows if row['fold'] != fold_idx]
        val_meta = [row for row in meta_rows if row['fold'] == fold_idx]
        train_splits.append((fold_idx, train_meta, val_meta))

    total_preprocess = len(meta_rows)
    total_tasks = len(train_splits) * len(model_classes)

    start_time = time.perf_counter()
    progress_info = ProgressInfo(total_tasks, total_preprocess_tasks=total_preprocess)

    with progress_info:
        if pool == "smartpool":
            from train_with_smartpool import train_with_smartpool
            model_results = train_with_smartpool(meta_rows, train_splits, model_classes, progress_info, max_workers)
        elif pool == "concurrent":
            from train_with_concurrent import train_with_concurrent
            model_results = train_with_concurrent(meta_rows, train_splits, model_classes, progress_info, max_workers)
        elif pool == "multiprocessing":
            from train_with_multiprocessing import train_with_multiprocessing
            model_results = train_with_multiprocessing(meta_rows, train_splits, model_classes, progress_info, max_workers)
        elif pool == "joblib":
            from train_with_joblib import train_with_joblib
            model_results = train_with_joblib(meta_rows, train_splits, model_classes, progress_info, max_workers)
        elif pool == "ray":
            from train_with_ray import train_with_ray
            model_results = train_with_ray(meta_rows, train_splits, model_classes, progress_info, max_workers)
        elif pool == "sequentially":
            from train_sequentially import train_sequentially
            model_results = train_sequentially(meta_rows, train_splits, model_classes, progress_info, max_workers)

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