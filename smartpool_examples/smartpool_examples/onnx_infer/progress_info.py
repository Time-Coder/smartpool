from rich.progress import TaskID
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)


class ProgressInfo:

    def __init__(self, n_tasks: int):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        self.preprocess_start_task: TaskID = self.progress.add_task("preprocess start", total=n_tasks)
        self.preprocess_done_task: TaskID = self.progress.add_task("preprocess done", total=n_tasks)
        self.infer_start_task = self.progress.add_task("infer start", total=n_tasks)
        self.infer_done_task = self.progress.add_task("infer done", total=n_tasks)
        self.postprocess_start_task: TaskID = self.progress.add_task("postprocess start", total=n_tasks)
        self.postprocess_done_task: TaskID = self.progress.add_task("postprocess done", total=n_tasks)

    def start_one_preprocess(self, *args):
        self.progress.update(self.preprocess_start_task, advance=1)

    def finish_one_preprocess(self, *args):
        self.progress.update(self.preprocess_done_task, advance=1)

    def start_one_infer(self, *args):
        self.progress.update(self.infer_start_task, advance=1)

    def finish_one_infer(self, *args):
        self.progress.update(self.infer_done_task, advance=1)

    def start_one_postprocess(self, *args):
        self.progress.update(self.postprocess_start_task, advance=1)

    def finish_one_postprocess(self, *args):
        self.progress.update(self.postprocess_done_task, advance=1)

    def __enter__(self):
        self.progress.__enter__()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        return self.progress.__exit__(exc_type, exc, tb)