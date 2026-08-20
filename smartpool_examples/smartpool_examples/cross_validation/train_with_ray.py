from collections import defaultdict

import ray
import torch
from data_utils import preprocess_audio
from progress_info import ProgressInfo

from smartpool import DataSize


@ray.remote(num_cpus=1, memory=1.1 * DataSize.GB)
def preprocess_audio_task(filename, audio_bytes, progress_queue):
    return preprocess_audio(filename, audio_bytes, progress_queue)


@ray.remote(num_cpus=1, memory=1.1 * DataSize.GB)
def train_fold_task(fold_idx, model_class, train_meta, val_meta, progress_queue):
    from model_utils import train_single_fold
    gpu_ids = ray.get_gpu_ids()
    device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
    return train_single_fold(fold_idx, model_class, train_meta, val_meta, progress_queue, device)


def train_with_ray(meta_rows, train_splits, model_classes, progress_info: ProgressInfo, max_workers):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    has_gpu = torch.cuda.is_available()
    progress_queue = ray.util.queue.Queue()

    # Phase 1: preprocess every audio file
    preprocess_futures = [preprocess_audio_task.remote(row['filename'], row['audio'], progress_queue) for row in meta_rows]
    for fut in preprocess_futures:
        ray.get(fut)

    # Phase 2: train one model per fold
    futures = {}
    for fold_idx, train_meta, val_meta in train_splits:
        for model_class in model_classes:
            fut = train_fold_task.options(
                num_gpus=0.2 if has_gpu else 0
            ).remote(fold_idx, model_class, train_meta, val_meta, progress_queue)
            futures[f"{model_class.__name__}_fold_{fold_idx}"] = fut

    progress_info.track(progress_queue)

    model_results = defaultdict(list)
    for future in futures.values():
        result = ray.get(future)
        model_results[result.model_name].append(result.val_accuracy)

    ray.shutdown()
    return model_results