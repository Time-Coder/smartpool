import queue
import time
from typing import Dict, Set
from collections import defaultdict

from .config import EPOCHS
from .model_utils import ErrorInfo, StartInfo, StopInfo, ProgressChangedInfo
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
        self.overall_task_id = self.progress.add_task(
            f"Total progress [progress.description]{0}/{total_tasks}",
            total=total_tasks,
        )
        # Per-task timing: dict mapping task_key -> duration_in_seconds
        self.task_time_span: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Per-task device timeline: dict mapping task_key -> [[timestamp, device], ...]
        # only device transition points (CPU <-> CUDA) are recorded; timestamps share
        # the same monotonic clock as task_time_span.
        self.device_timeline: Dict[str, list] = defaultdict(list)

    def track(self, progress_queue):
        """Consume the progress queue and drive the rich progress display.
        timeout: if set, ``queue.get(timeout=timeout)`` is used so a stalled
            source (e.g. a Ray worker that never delivers) cannot hang the run
            forever; an empty dequeue simply retries.
        """
        while True:
            info = progress_queue.get()
            task_key = f"{info.model_name}_fold_{info.fold_idx}"

            if isinstance(info, ErrorInfo):
                print(info.traceback)
                break
            elif isinstance(info, StartInfo):
                self.task_time_span[task_key]["start_time"] = time.perf_counter()
            elif isinstance(info, StopInfo):
                self.task_time_span[task_key]["stop_time"] = time.perf_counter()
                self.finished_tasks.add(task_key)
                self.progress.update(self.task_progress_bars[task_key], visible=False)
                self.progress.update(
                    self.overall_task_id,
                    completed=len(self.finished_tasks),
                    description=f"Total progress [progress.description]{len(self.finished_tasks)}/{self.total_tasks}",
                )
                if len(self.finished_tasks) == self.total_tasks:
                    break
            elif isinstance(info, ProgressChangedInfo):
                timeline = self.device_timeline[task_key]
                if not timeline or timeline[-1][1] != info.device:
                    timeline.append([time.perf_counter(), info.device])

                if task_key not in self.task_progress_bars:
                    desc = f"train {info.model_name} on {info.device} in process {info.pid} for fold {info.fold_idx+1}/5"
                    self.task_progress_bars[task_key] = self.progress.add_task(desc, total=100)

                epoch_progress = (info.epoch - 1) / 5
                batch_progress = info.batch / info.total_batches
                total_progress = (epoch_progress + batch_progress / 5) * 100

                if info.epoch == 5 and info.batch == info.total_batches:
                    total_progress = 100.0

                desc = (
                    f"train {info.model_name} on {info.device} "
                    f"in process {info.pid} for fold {info.fold_idx+1}/5 - "
                    f"Epoch {info.epoch}/{EPOCHS} "
                    f"Loss: {info.avg_loss:.4f} Val Acc: {info.val_accuracy*100:.2f}%"
                )
                if info.device.startswith("cuda"):
                    desc = "[bright_cyan]" + desc

                self.progress.update(self.task_progress_bars[task_key], completed=total_progress, description=desc)

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.progress.__exit__(exc_type, exc, tb)