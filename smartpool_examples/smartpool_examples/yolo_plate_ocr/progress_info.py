from typing import Dict
from functools import partial

from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn


class PipelineProgress:
    def __init__(self):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        self._tasks: Dict[str, TaskID] = {}
        self._tasks_total: Dict[str, int] = {}

    def set_total(self, task_name:str, total:int):
        if task_name not in self._tasks:
            self._tasks[task_name] = self.progress.add_task(task_name, total=total)
        else:
            self.progress.update(self._tasks[task_name], total=total)

        self._tasks_total[task_name] = total

    def increase_total(self, task_name:str, total:int):
        if task_name not in self._tasks:
            self._tasks[task_name] = self.progress.add_task(task_name, total=total)
        else:
            self.progress.update(self._tasks[task_name], total=self._tasks_total[task_name]+total)

    def advance(self, task_name:str, *args):
        self.progress.update(self._tasks[task_name], advance=1)

    def attach_callbacks(self, future, task_name):
        future.add_start_callback(partial(self.advance, task_name=f"{task_name} start"))
        future.add_done_callback(partial(self.advance, task_name=f"{task_name} done"))

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.progress.__exit__(exc_type, exc, tb)
