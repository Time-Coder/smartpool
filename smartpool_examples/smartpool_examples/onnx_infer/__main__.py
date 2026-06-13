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
    max_workers: int = typer.Option(
        0,
        "--max_workers",
        help="max number of workers to use, 0 to use all available cores"
    ),
):
    import os
    import time
    from smartpool import InferSessionPool, Resource, ThreadPool
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )
    
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    sys.path.append(self_folder)

    from config import DATASET_DIR, MODEL_PATH, OUTPUT_DIR
    from data_utils import download_dataset, download_model
    from inference import infer_task
    from concurrent.futures import Future, as_completed

    print("Use `python -m smartpool_examples.onnx_infer --help` to see all options.")
    print(f"See source code at folder {os.path.dirname(os.path.abspath(__file__))}")
    print()

    download_model()
    # download_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATASET_DIR.glob("*.jpg"))
    cpu_res = Resource(cpu_cores_in_python=1, cpu_mem=2*1024**3)

    model_path_str = str(MODEL_PATH.resolve())
    output_dir_str = str(OUTPUT_DIR.resolve())
    n_workers = max_workers

    print(f"\nSubmit {len(image_paths)} tasks to ThreadPool({n_workers})...")
    thread_pool = ThreadPool(max_workers=n_workers)
    infer_session_pool = InferSessionPool(model_path_str, max_workers=n_workers)
    infer_session_pool.print_info = True

    cpu_res = Resource(
        cpu_cores_in_python=1,
        cpu_cores_out_of_python=1,
        cpu_mem=2*1024**3
    )
    gpu_res = Resource(
        cpu_cores_in_python=1,
        cpu_cores_out_of_python=1,
        cpu_mem=512*1024**2,
        gpu_cores=1000,
        gpu_mem=2*1024**3
    )
    
    all_futures = []
    for image_path in image_paths:
        image_path = str(image_path)
        future: Future = thread_pool.submit(
            infer_task,
            args=(image_path, output_dir_str, infer_session_pool, cpu_res, gpu_res),
            cpu_mode_res=cpu_res
        )
        all_futures.append(future)

    start_time = time.perf_counter()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        n_tasks = len(image_paths)
        task = progress.add_task("infer", total=n_tasks)

        for future in as_completed(all_futures):
            future.result()
            progress.update(task, advance=1)
            progress.update(task, description=f"infer [{progress.tasks[0].completed}/{progress.tasks[0].total}]")

    elapsed = time.perf_counter() - start_time
    print(f"\ninference completed in {elapsed:.2f} seconds")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    app()
