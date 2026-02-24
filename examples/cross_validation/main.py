import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import multiprocessing as mp

parent_folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
sys.path.append(parent_folder)

import models
from smartprocesspool import SmartProcessPool, DataSize
from config import MAX_WORKERS
from data_utils import prepare_data
from model_utils import train_single_fold
from visualization import plot_results, print_results_table


def main():
    model_classes = [
        cls for cls in models.__dict__.values() 
        if isinstance(cls, type) and issubclass(cls, nn.Module) and cls != nn.Module
    ]
    
    dataset = prepare_data()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    manager = mp.Manager()
    progress_queue = manager.Queue()

    tasks = []
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        for model_class in model_classes:
            tasks.append((fold_idx, model_class, train_indices.copy(), val_indices.copy(), dataset, progress_queue))
    
    task_progress_bars = {}
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    ) as progress:
        
        active_tasks = {}
        
        with SmartProcessPool(max_workers=MAX_WORKERS, use_torch=True) as pool:
            futures_map = {}
            for task_args in tasks:
                future = pool.submit(
                    train_single_fold,
                    args=task_args,
                    need_cpu_cores=1,
                    need_cpu_mem=1.1*DataSize.GB,
                    need_gpu_cores=1152,
                    need_gpu_mem=0.2*DataSize.GB
                )
                fold_idx = task_args[0]
                model_class = task_args[1]
                model_name = model_class.__name__
                task_key = f"{model_name}_fold_{fold_idx}"
                futures_map[task_key] = future
            
            finished_tasks = set()
            while True:
                progress_info = progress_queue.get()
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
                    new_desc += f"for fold {progress_info.fold_idx+1}/5 - Epoch {progress_info.epoch} "
                    new_desc += f"Loss: {progress_info.avg_loss:.4f} "
                    
                    display_accuracy = progress_info.val_accuracy if progress_info.val_accuracy > 0 else progress_info.last_val_accuracy
                    new_desc += f"Val Acc: {display_accuracy*100:.2f}%"
                    
                    progress.update(task_progress_bars[task_key], 
                                    completed=total_progress,
                                    description=new_desc)

                if len(finished_tasks) == len(futures_map):
                    break
    
    model_results = {model_class.__name__: [] for model_class in model_classes}
    for task_key, future in futures_map.items():
        result = future.result()
        model_results[result.model_name].append(result.val_accuracy)
    
    stats = {}
    successful_models = []
    
    for model_name, accuracies in model_results.items():
        stats[model_name] = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'accuracies': accuracies
        }
        successful_models.append(model_name)
    
    print_results_table(successful_models, stats)
    plot_results(successful_models, {k: model_results[k] for k in successful_models}, stats)


if __name__ == "__main__":
    main()