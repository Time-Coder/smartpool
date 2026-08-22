from collections import defaultdict

from data_utils import load_features, preprocess_audio
from model_utils import train_single_fold
from preprocess_progress_info import PreprocessProgressInfo
from train_progress_info import TrainProgressInfo

from smartpool import DataSize, ProcessPool, Resource


def train_with_smartpool(meta_rows, fnames, model_classes, max_workers, preprocess_info: PreprocessProgressInfo, train_info: TrainProgressInfo):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()

    with ProcessPool(max_workers=max_workers, use_torch=True) as pool:
        preprocess_futures = []
        for fname in fnames:
            future = pool.submit(
                preprocess_audio,
                args=(fname,),
                cpu_mode_res=Resource(cpu_cores=1, cpu_mem=0.5 * DataSize.GB)
            )
            future.add_start_callback(preprocess_info.advance_preprocess_start)
            future.add_done_callback(lambda f: preprocess_info.advance_preprocess_done())
            preprocess_futures.append(future)

        for future in preprocess_futures:
            future.result()

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
        for task_args in tasks:
            future = pool.submit(
                train_single_fold,
                args=task_args,
                cpu_mode_res=Resource(cpu_cores=1, cpu_mem=3 * DataSize.GB),
                gpu_mode_res=Resource(cpu_cores=1, cpu_mem=3 * DataSize.GB, gpu_cores=2000, gpu_mem=0.2 * DataSize.GB),
                device_changeable=True
            )
            fold_idx = task_args[0]
            model_class = task_args[1]
            task_key = f"{model_class.__name__}_fold_{fold_idx}"
            futures_map[task_key] = future

        train_info.track(progress_queue)

        model_results = defaultdict(list)
        for task_key, future in futures_map.items():
            try:
                result = future.result()
                model_results[result.model_name].append(result.val_accuracy)
            except Exception as e:
                print(f"Task {task_key} failed with error: {type(e).__name__}: {e}")

    return model_results