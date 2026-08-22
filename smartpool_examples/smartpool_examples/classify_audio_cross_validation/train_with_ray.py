from collections import defaultdict

import ray
from data_utils import load_features, preprocess_audio
from model_utils import train_single_fold
from preprocess_progress_info import PreprocessProgressInfo
from train_progress_info import TrainProgressInfo


def train_with_ray(meta_rows, fnames, model_classes, max_workers, preprocess_info: PreprocessProgressInfo, train_info: TrainProgressInfo):
    ray.init(ignore_reinit_error=True)

    import ray.util.queue
    progress_queue = ray.util.queue.Queue()

    @ray.remote(num_cpus=1)
    def preprocess_remote(fname):
        preprocess_audio(fname)

    preprocess_refs = []
    for fname in fnames:
        preprocess_info.advance_preprocess_start()
        ref = preprocess_remote.remote(fname)
        preprocess_refs.append(ref)

    ray.get(preprocess_refs)
    for _ in preprocess_refs:
        preprocess_info.advance_preprocess_done()

    from config import N_FOLDS
    from sklearn.model_selection import KFold

    print("loading features...")
    all_data, all_targets = load_features(meta_rows)

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    task_templates = []
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(meta_rows)):
        train_data = all_data[train_indices]
        train_targets = all_targets[train_indices]
        val_data = all_data[val_indices]
        val_targets = all_targets[val_indices]
        for model_class in model_classes:
            task_templates.append((fold_idx, model_class, train_data, train_targets, val_data, val_targets))

    print(f"training {len(task_templates)} tasks...")
    tasks = [(*t, progress_queue) for t in task_templates]
    futures_map = {}
    for i, task_args in enumerate(tasks):
        fold_idx = task_args[0]
        model_class = task_args[1]
        task_key = f"{model_class.__name__}_fold_{fold_idx}"

        future = ray.remote(num_cpus=1, num_gpus=(0.2 if i % max_workers < 5 else 0), memory=1.1 * 1024**3)(train_single_fold).remote(*task_args, 'cuda' if i % max_workers < 5 else 'cpu')
        futures_map[task_key] = future

    train_info.track(progress_queue)

    model_results = defaultdict(list)
    ray_results = ray.get(list(futures_map.values()))
    for result in ray_results:
        model_results[result.model_name].append(result.val_accuracy)

    ray.shutdown()

    return model_results