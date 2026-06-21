from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
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
        self.det_preprocess_start_task: TaskID = self.progress.add_task("det preprocess start", total=n_tasks)
        self.det_preprocess_done_task: TaskID = self.progress.add_task("det preprocess done", total=n_tasks)
        self.det_infer_start_task = self.progress.add_task("det infer start", total=n_tasks)
        self.det_infer_done_task = self.progress.add_task("det infer done", total=n_tasks)
        self.det_postprocess_start_task: TaskID = self.progress.add_task("det postprocess start", total=n_tasks)
        self.det_postprocess_done_task: TaskID = self.progress.add_task("det postprocess done", total=n_tasks)
        self.pose_preprocess_start_task: TaskID = self.progress.add_task("pose preprocess start", total=n_tasks)
        self.pose_preprocess_done_task: TaskID = self.progress.add_task("pose preprocess done", total=n_tasks)
        self.pose_infer_start_task: TaskID = self.progress.add_task("pose infer start", total=n_tasks)
        self.pose_infer_done_task: TaskID = self.progress.add_task("pose infer done", total=n_tasks)
        self.pose_postprocess_start_task: TaskID = self.progress.add_task("pose postprocess start", total=n_tasks)
        self.pose_postprocess_done_task: TaskID = self.progress.add_task("pose postprocess done", total=n_tasks)
        self.draw_start_task: TaskID = self.progress.add_task("draw start", total=n_tasks)
        self.draw_done_task: TaskID = self.progress.add_task("draw done", total=n_tasks)

    def start_one_det_preprocess(self, *args):
        self.progress.update(self.det_preprocess_start_task, advance=1)

    def finish_one_det_preprocess(self, *args):
        self.progress.update(self.det_preprocess_done_task, advance=1)

    def start_one_det_infer(self, *args):
        self.progress.update(self.det_infer_start_task, advance=1)

    def finish_one_det_infer(self, *args):
        self.progress.update(self.det_infer_done_task, advance=1)

    def start_one_det_postprocess(self, *args):
        self.progress.update(self.det_postprocess_start_task, advance=1)

    def finish_one_det_postprocess(self, *args):
        self.progress.update(self.det_postprocess_done_task, advance=1)

    def start_one_pose_preprocess(self, *args):
        self.progress.update(self.pose_preprocess_start_task, advance=1)

    def finish_one_pose_preprocess(self, *args):
        self.progress.update(self.pose_preprocess_done_task, advance=1)

    def start_one_pose_infer(self, *args):
        self.progress.update(self.pose_infer_start_task, advance=1)

    def finish_one_pose_infer(self, *args):
        self.progress.update(self.pose_infer_done_task, advance=1)

    def start_one_pose_postprocess(self, *args):
        self.progress.update(self.pose_postprocess_start_task, advance=1)

    def finish_one_pose_postprocess(self, *args):
        self.progress.update(self.pose_postprocess_done_task, advance=1)

    def start_one_draw(self, *args):
        self.progress.update(self.draw_start_task, advance=1)

    def finish_one_draw(self, *args):
        self.progress.update(self.draw_done_task, advance=1)

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.progress.__exit__(exc_type, exc, tb)
