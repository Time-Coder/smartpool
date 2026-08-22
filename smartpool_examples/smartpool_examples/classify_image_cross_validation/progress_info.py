from typing import Dict, Set

from config import EPOCHS
from model_utils import ErrorInfo as FoldError
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
        self.finished_tasks: Set[str] = set()

    def track(self, progress_queue):
        while True:
            info = progress_queue.get()
            if isinstance(info, FoldError):
                print(info.traceback)
                break

            task_key = f"{info.model_name}_fold_{info.fold_idx}"

            if task_key not in self.task_progress_bars:
                desc = f"train {info.model_name} on {info.device} in process {info.pid} for fold {info.fold_idx+1}/5"
                self.task_progress_bars[task_key] = self.progress.add_task(desc, total=100)

            if task_key in self.task_progress_bars:
                epoch_progress = (info.epoch - 1) / 5
                batch_progress = info.batch / info.total_batches
                total_progress = (epoch_progress + batch_progress / 5) * 100

                if info.epoch == 5 and info.batch == info.total_batches:
                    total_progress = 100.0
                    self.finished_tasks.add(task_key)

                desc = (
                    f"train {info.model_name} on {info.device} "
                    f"in process {info.pid} for fold {info.fold_idx+1}/5 - "
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
