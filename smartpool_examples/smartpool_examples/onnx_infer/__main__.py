if __name__ == "__main__":
    import os
    import sys
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)

import typer

app = typer.Typer(help="Use smartpool to do YOLOv8n ONNX inference on COCO val2017 with real-time progress.")

@app.command()
def main(
    thread_pool_max_workers: int = typer.Option(
        4,
        "--thread_pool_max_workers",
        help="max number of thread pool (for preprocess and postprocess) workers to use, 0 to use all available cores"
    ),
    session_pool_max_workers: int = typer.Option(
        0,
        "--session_pool_max_workers",
        help="max number of infer session pool workers to use, 0 to use all available cores"
    ),
):
    import os

    print("Use `python -m smartpool_examples.onnx_infer --help` to see all options.")
    print(f"See source code at folder {os.path.dirname(os.path.abspath(__file__))}")
    print()

    try:
        import onnxruntime
    except ImportError:
        print("ONNX Runtime is not installed. Follow https://onnxruntime.ai/docs/install/ instructions to install ONNX Runtime.")
        exit(1)

    import queue
    import time
    from functools import partial

    from rich.console import Group
    from rich.live import Live
    from rich.progress import TaskID
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    from smartpool import DataSize, GPUInfo, InferSessionPool, Resource, ThreadPool

    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    sys.path.append(self_folder)

    from concurrent.futures import as_completed

    from config import DATASET_DIR, MODEL_PATH, OUTPUT_DIR
    from data_utils import download_dataset, download_model
    from inference import postprocess, preprocess

    download_model()
    download_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATASET_DIR.glob("*.jpg"))
    model_path_str = str(MODEL_PATH.resolve())
    output_dir_str = str(OUTPUT_DIR.resolve())

    cpu_res = Resource(
        cpu_cores_in_python=1,
        cpu_cores_out_of_python=1,
        cpu_mem=81*DataSize.MB
    )
    gpu_res = Resource(
        cpu_cores_in_python=1,
        cpu_cores_out_of_python=1,
        cpu_mem=200*DataSize.MB,
        gpu_cores=100,
        gpu_mem=200*DataSize.MB
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress, Live(refresh_per_second=4) as live:
        n_tasks = len(image_paths)
        preprocess_start_progress_task: TaskID = progress.add_task("preprocess start", total=n_tasks)
        preprocess_done_progress_task: TaskID = progress.add_task("preprocess done", total=n_tasks)
        # infer_start_progress_task = progress.add_task("infer start", total=n_tasks)
        # infer_done_progress_task = progress.add_task("infer done", total=n_tasks)
        postprocess_start_progress_task: TaskID = progress.add_task("postprocess start", total=n_tasks)
        postprocess_done_progress_task: TaskID = progress.add_task("postprocess done", total=n_tasks)

        # def update_task_count():
        #     lines = []
        #     for provider in GPUInfo.supported_onnx_providers():
        #         lines.append(Text(f"task count infer with {provider}: {infer_session_pool.task_count_with_provider(provider)}"))
        #     live.update(Group(*lines))

        start_time = time.perf_counter()

        futures = []
        for image_path in image_paths:
            # update_task_count()
            image_path = str(image_path)
            img, blob, scale, pad = preprocess(image_path, progress, preprocess_start_progress_task, preprocess_done_progress_task).unpack(4)
            infer_future = InferSessionPool.run_async(model_path_str, args=(blob,), cpu_mode_res=cpu_res, gpu_mode_res=gpu_res)
            outputs = infer_future[0]
            output_path = output_dir_str + "/" + os.path.basename(image_path)
            future = postprocess(img, outputs, output_path, scale, pad, progress, postprocess_start_progress_task, postprocess_done_progress_task)
            futures.append(future)

        for future in as_completed(futures):
            future.result()

    elapsed = time.perf_counter() - start_time
    print(f"\ninference completed in {elapsed:.2f} seconds")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    app()
