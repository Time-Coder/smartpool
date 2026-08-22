from collections import defaultdict

from data_utils import load_features, preprocess_audio
from model_utils import train_single_fold
from preprocess_progress_info import PreprocessProgressInfo
from train_progress_info import TrainProgressInfo


def train_sequentially(meta_rows, fnames, model_classes, max_workers, preprocess_info: PreprocessProgressInfo, train_info: TrainProgressInfo):
    for fname in fnames:
        preprocess_info.advance_preprocess_start()
        preprocess_audio(fname)
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
    model_results = defaultdict(list)
    for i, task_args in enumerate(task_templates):
        try:
            result = train_single_fold(*task_args, 'cpu')
            model_results[result.model_name].append(result.val_accuracy)
        except Exception as e:
            print(f"Error in task {i}: {e}")

    return model_results