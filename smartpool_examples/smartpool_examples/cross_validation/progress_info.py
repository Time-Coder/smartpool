from typing import Dict, Set

from config import EPOCHS, N_FOLDS
from model_utils import ErrorInfo as FoldError, PreprocessInfo
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn


class ProgressInfo:
    def __init__(self, total_tasks: int):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        self.total_tasks = total_tasks
        self.task_progress_bars: Dict[str, int] = {}
        self.preprocessed_tasks: Set[str] = set()
        self.finished_tasks: Set[str] = set()

    def track(self, progress_queue):
        while True:
            info = progress_queue.get()
            if isinstance(info, FoldError):
                print(info.traceback)
                break

            task_key = f"{info.model_name}_fold_{info.fold_idx}"

            if isinstance(info, PreprocessInfo):
                if task_key not in self.task_progress_bars:
                    desc = f"preprocess {info.model_name} data for fold {info.fold_idx+1}/{N_FOLDS}"
                    self.task_progress_bars[task_key] = self.progress.add_task(desc, total=100)
                self.progress.update(self.task_progress_bars[task_key], completed=100)
                self.progress.update(self.task_progress_bars[task_key], visible=False)
                self.preprocessed_tasks.add(task_key)
                continue

            if task_key not in self.task_progress_bars:
                desc = f"train {info.model_name} on {info.device} in process {info.pid} for fold {info.fold_idx+1}/{N_FOLDS}"
                self.task_progress_bars[task_key] = self.progress.add_task(desc, total=100)
            else:
                self.progress.update(self.task_progress_bars[task_key], visible=True)

            if task_key in self.task_progress_bars:
                epoch_progress = (info.epoch - 1) / EPOCHS
                batch_progress = info.batch / info.total_batches
                total_progress = (epoch_progress + batch_progress / EPOCHS) * 100

                if info.epoch == EPOCHS and info.batch == info.total_batches:
                    total_progress = 100.0
                    self.finished_tasks.add(task_key)

                desc = (
                    f"train {info.model_name} on {info.device} "
                    f"in process {info.pid} for fold {info.fold_idx+1}/{N_FOLDS} - "
                    f"Epoch {info.epoch}/{EPOCHS} "
                    f"Loss: {info.avg_loss:.4f} Val Acc: {info.val_accuracy*100:.2f}%"
                )
                if info.device.startswith("cuda"):
                    desc = "[bright_cyan]" + desc

                self.progress.update(self.task_progress_bars[task_key], completed=total_progress, description=desc)
                if total_progress >= 100.0:
                    self.progress.update(self.task_progress_bars[task_key], visible=False)

            if len(self.finished_tasks) == self.total_tasks:
                break

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.progress.__exit__(exc_type, exc, tb)