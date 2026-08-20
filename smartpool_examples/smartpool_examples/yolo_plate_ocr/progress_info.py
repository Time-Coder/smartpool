from typing import Optional

from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn


class PipelineProgress:
    def __init__(self, total: Optional[int]):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        self.vehicle_start_task: TaskID = self.progress.add_task("vehicle start", total=total)
        self.vehicle_done_task: TaskID = self.progress.add_task("vehicle done", total=total)
        self.plate_start_task: TaskID = self.progress.add_task("plate start", total=total)
        self.plate_done_task: TaskID = self.progress.add_task("plate done", total=total)
        self.save_start_task: TaskID = self.progress.add_task("render start", total=total)
        self.save_done_task: TaskID = self.progress.add_task("render done", total=total)

    def start_one_vehicle(self):
        self.progress.update(self.vehicle_start_task, advance=1)

    def finish_one_vehicle(self):
        self.progress.update(self.vehicle_done_task, advance=1)

    def start_one_plate(self):
        self.progress.update(self.plate_start_task, advance=1)

    def finish_one_plate(self):
        self.progress.update(self.plate_done_task, advance=1)

    def start_one_save(self):
        self.progress.update(self.save_start_task, advance=1)

    def finish_one_save(self):
        self.progress.update(self.save_done_task, advance=1)

    def advance_vehicle(self):
        self.start_one_vehicle()
        self.finish_one_vehicle()

    def advance_plate(self):
        self.start_one_plate()
        self.finish_one_plate()

    def advance_save(self):
        self.start_one_save()
        self.finish_one_save()

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.progress.__exit__(exc_type, exc, tb)
