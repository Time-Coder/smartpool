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

app = typer.Typer(help="Use smartpool to do 5-fold cross validatation for 7 deep learning models for handwritten digit recognition task.")

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

    print(f"Use {pool} to do 5-fold cross validatation for 7 deep learning models for handwritten digit recognition task.")
    print("Use `python -m smartpool_examples.cross_validation --help` to see all options.")
    print(f"See source code at folder {os.path.dirname(os.path.abspath(__file__))}")
    print("\npreparing data...")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is not installed. Follow https://pytorch.org/ instructions to install PyTorch.")
        return

    if importlib.util.find_spec("torchvision") is None:
        print("torchvision is not installed. Use `pip install torchvision` to install torchvision.")
        return

    import os
    import sys
    import time

    import numpy as np
    from sklearn.model_selection import KFold

    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    sys.path.append(self_folder)

    import models
    from data_utils import prepare_data
    from progress_info import ProgressInfo
    from visualization import plot_results, print_results_table

    if max_workers == 0:
        max_workers = os.cpu_count() // 2

    model_classes = [
        cls for cls in models.__dict__.values()
        if isinstance(cls, type) and issubclass(cls, nn.Module) and cls != nn.Module
    ]

    dataset = prepare_data()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    task_templates = []
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        for model_class in model_classes:
            task_templates.append((fold_idx, model_class, train_indices.copy(), val_indices.copy(), dataset))

    start_time = time.perf_counter()
    progress_info = ProgressInfo(len(task_templates))

    with progress_info:
        if pool == "smartpool":
            from train_with_smartpool import train_with_smartpool
            model_results = train_with_smartpool(task_templates, max_workers, progress_info)
        elif pool == "concurrent":
            from train_with_concurrent import train_with_concurrent
            model_results = train_with_concurrent(task_templates, max_workers, progress_info)
        elif pool == "multiprocessing":
            from train_with_multiprocessing import train_with_multiprocessing
            model_results = train_with_multiprocessing(task_templates, max_workers, progress_info)
        elif pool == "joblib":
            from train_with_joblib import train_with_joblib
            model_results = train_with_joblib(task_templates, max_workers, progress_info)
        elif pool == "ray":
            from train_with_ray import train_with_ray
            model_results = train_with_ray(task_templates, max_workers, progress_info)
        elif pool == "sequentially":
            from train_sequentially import train_sequentially
            model_results = train_sequentially(task_templates, max_workers, progress_info)

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
