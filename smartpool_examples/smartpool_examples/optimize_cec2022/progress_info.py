from typing import Dict, Set

from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from optimizer import ErrorInfo, ProgressInfo as RunProgress, OptimizationResult


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
            if isinstance(info, ErrorInfo):
                print(info.traceback)
                break

            task_key = f"{info.algo_name}_{info.func_name}_run_{info.run_idx}"

            if task_key not in self.task_progress_bars:
                desc = f"optimize {info.algo_name} on {info.func_name} run {info.run_idx}"
                self.task_progress_bars[task_key] = self.progress.add_task(desc, total=info.total_iterations)

            if task_key in self.task_progress_bars:
                best_str = f"{info.best_fitness:.6e}" if info.best_fitness != float("inf") else "-"
                desc = (
                    f"optimize {info.algo_name} on {info.func_name} run {info.run_idx} "
                    f"- iter {info.iteration}/{info.total_iterations} "
                    f"best: {best_str}"
                )
                self.progress.update(
                    self.task_progress_bars[task_key],
                    completed=info.iteration,
                    description=desc,
                )
                if info.iteration >= info.total_iterations:
                    self.finished_tasks.add(task_key)
                    self.progress.update(self.task_progress_bars[task_key], visible=False)

            if len(self.finished_tasks) == self.total_tasks:
                break

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.progress.__exit__(exc_type, exc, tb)
