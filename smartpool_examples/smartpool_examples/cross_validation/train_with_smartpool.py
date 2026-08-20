from collections import defaultdict

from data_utils import preprocess_audio
from model_utils import train_single_fold
from progress_info import ProgressInfo

from smartpool import DataSize, ProcessPool, Resource


def train_with_smartpool(meta_rows, train_splits, model_classes, progress_info: ProgressInfo, max_workers):
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()

    with ProcessPool(max_workers=max_workers, use_torch=True) as pool:
        pool._need_module_deps = False

        # Phase 1: one preprocess task per audio file (2000 tasks)
        preprocess_futures = []
        for row in meta_rows:
            fut = pool.submit(
                preprocess_audio,
                args=(row['filename'], row['audio'], progress_queue),
                cpu_mode_res=Resource(cpu_cores=1, cpu_mem=1 * DataSize.GB)
            )
            preprocess_futures.append(fut)

        for fut in preprocess_futures:
            fut.result()

        # Phase 2: one training task per fold per model (20 tasks)
        train_futures = []
        for fold_idx, train_meta, val_meta in train_splits:
            for model_class in model_classes:
                fut = pool.submit(
                    train_single_fold,
                    args=(fold_idx, model_class, train_meta, val_meta, progress_queue),
                    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=1 * DataSize.GB),
                    gpu_mode_res=Resource(cpu_cores=1, cpu_mem=1.1 * DataSize.GB, gpu_cores=1000, gpu_mem=0.2 * DataSize.GB),
                    device_changeable=True
                )
                train_futures.append(fut)

        progress_info.track(progress_queue)

        model_results = defaultdict(list)
        for fut in train_futures:
            result = fut.result()
            model_results[result.model_name].append(result.val_accuracy)

    return model_results