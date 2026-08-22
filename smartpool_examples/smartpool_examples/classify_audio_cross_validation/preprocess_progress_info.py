from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn


class PreprocessProgressInfo:
    def __init__(self, total_tasks: int):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        self.preprocess_start_task = self.progress.add_task("preprocess start", total=total_tasks)
        self.preprocess_done_task = self.progress.add_task("preprocess done", total=total_tasks)

    def advance_preprocess_start(self, *args):
        self.progress.update(self.preprocess_start_task, advance=1)

    def advance_preprocess_done(self, *args):
        self.progress.update(self.preprocess_done_task, advance=1)

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.progress.__exit__(exc_type, exc, tb)