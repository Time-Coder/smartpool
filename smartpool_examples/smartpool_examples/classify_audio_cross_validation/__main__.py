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

app = typer.Typer(help="Use smartpool to do 5-fold cross validation for audio classification on ESC-50.")

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

    print(f"Use {pool} to do 5-fold cross validation for 4 audio classification models on ESC-50.")
    print("Use `python -m smartpool_examples.classify_audio_cross_validation --help` to see all options.")
    print(f"See source code at folder {os.path.dirname(os.path.abspath(__file__))}")
    print("\npreparing data...")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is not installed. Follow https://pytorch.org/ instructions to install PyTorch.")
        return

    import os
    import sys
    import time

    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    sys.path.append(self_folder)

    import models
    from config import N_FOLDS
    from data_utils import prepare_metadata, extract_wavs_to_disk
    from preprocess_progress_info import PreprocessProgressInfo
    from train_progress_info import TrainProgressInfo
    from visualization import plot_results

    if max_workers == 0:
        max_workers = os.cpu_count() // 2

    model_classes = [
        cls for cls in models.__dict__.values()
        if isinstance(cls, type) and issubclass(cls, nn.Module) and cls != nn.Module
    ]

    meta_rows = prepare_metadata()

    print("extracting WAVs to disk...")
    extract_wavs_to_disk(meta_rows)

    fnames = [row['filename'] for row in meta_rows]

    total_train_tasks = len(model_classes) * N_FOLDS
    start_time = time.perf_counter()
    preprocess_info = PreprocessProgressInfo(len(fnames))
    train_info = TrainProgressInfo(total_train_tasks)

    with preprocess_info, train_info:
        if pool == "smartpool":
            from train_with_smartpool import train_with_smartpool
            model_results = train_with_smartpool(meta_rows, fnames, model_classes, max_workers, preprocess_info, train_info)
        elif pool == "concurrent":
            from train_with_concurrent import train_with_concurrent
            model_results = train_with_concurrent(meta_rows, fnames, model_classes, max_workers, preprocess_info, train_info)
        elif pool == "multiprocessing":
            from train_with_multiprocessing import train_with_multiprocessing
            model_results = train_with_multiprocessing(meta_rows, fnames, model_classes, max_workers, preprocess_info, train_info)
        elif pool == "joblib":
            from train_with_joblib import train_with_joblib
            model_results = train_with_joblib(meta_rows, fnames, model_classes, max_workers, preprocess_info, train_info)
        elif pool == "ray":
            from train_with_ray import train_with_ray
            model_results = train_with_ray(meta_rows, fnames, model_classes, max_workers, preprocess_info, train_info)
        elif pool == "sequentially":
            from train_sequentially import train_sequentially
            model_results = train_sequentially(meta_rows, fnames, model_classes, max_workers, preprocess_info, train_info)

    stop_time = time.perf_counter()
    print(f"train completed in {stop_time - start_time:.2f} seconds")

    print("analysing results...")

    import numpy as np
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


def print_results_table(stats):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Model Results")
    table.add_column("Model", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    table.add_column("Min", style="red")
    table.add_column("Max", style="blue")

    for model_name, model_stats in stats.items():
        table.add_row(
            model_name,
            f"{model_stats['mean']:.4f}",
            f"{model_stats['std']:.4f}",
            f"{model_stats['min']:.4f}",
            f"{model_stats['max']:.4f}"
        )

    console.print(table)


if __name__ == "__main__":
    app()