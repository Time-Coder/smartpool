if __name__ == "__main__":
    import os
    import sys
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)

import os
import time
from concurrent.futures import as_completed

import typer
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from smartpool import InferSessionPool, Resource, ThreadPool

app = typer.Typer(help="Use smartpool to do YOLOv8n ONNX inference on COCO val2017 with real-time progress.")


@app.command()
def main(
    max_workers: int = typer.Option(
        0,
        "--max_workers",
        help="max number of workers to use, 0 to use all available cores"
    ),
):
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    sys.path.append(self_folder)

    from config import DATASET_DIR, MODEL_PATH, OUTPUT_DIR
    from data_utils import download_dataset, download_model
    from inference import preprocess, postprocess
    from concurrent.futures import Future, wait, FIRST_COMPLETED
    from functools import partial

    print("Use `python -m smartpool_examples.onnx_infer --help` to see all options.")
    print(f"See source code at folder {os.path.dirname(os.path.abspath(__file__))}")
    print()

    download_model()
    download_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATASET_DIR.glob("*.jpg"))
    cpu_res = Resource(cpu_cores_in_python=1, cpu_mem=2*1024**3)

    model_path_str = str(MODEL_PATH.resolve())
    output_dir_str = str(OUTPUT_DIR.resolve())
    n_workers = max_workers

    print(f"\nSubmit {len(image_paths)} tasks to ThreadPool({n_workers})...")
    thread_pool = ThreadPool(max_workers=n_workers)
    infer_session_pool = InferSessionPool(max_workers=n_workers)
    infer_session_pool.print_info = True

    cpu_res = Resource(cpu_cores_in_python=2, cpu_mem=2*1024**3)
    gpu_res = Resource(
        cpu_cores_in_python=1, cpu_mem=512*1024**2,
        gpu_cores=1000, gpu_mem=2*1024**3
    )

    not_done_futures = set()

    def infer_done_callback(img, scale, pad, infer_future: Future):
        outputs = infer_future.result()[0]
        output_path = output_dir_str + "/" + os.path.basename(image_path)
        postprocess_future = thread_pool.submit(
            postprocess,
            args=(img, outputs, output_path, scale, pad),
            cpu_mode_res=cpu_res
        )
        not_done_futures.add(postprocess_future)

    def preprocess_done_callback(preprocess_future: Future):
        img, blob, scale, pad = preprocess_future.result()
        infer_future: Future = infer_session_pool.submit(model_path_str, args=(blob,), cpu_mode_res=cpu_res, gpu_mode_res=gpu_res)
        infer_future.add_done_callback(partial(infer_done_callback, img, scale, pad))
        not_done_futures.add(infer_future)

    
    for image_path in image_paths:
        preprocess_future: Future = thread_pool.submit(
            preprocess,
            args=(str(image_path),),
            cpu_mode_res=cpu_res
        )
        preprocess_future.add_done_callback(preprocess_done_callback)
        not_done_futures.add(preprocess_future)


    start_time = time.perf_counter()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("infer", total=len(not_done_futures))

        while not_done_futures:
            done_futures, not_done_futures = wait(not_done_futures, return_when=FIRST_COMPLETED)

            for f in done_futures:
                f.result()
                progress.update(task, advance=1/3)
                progress.update(task, description=f"infer [{progress.tasks[0].completed}/{progress.tasks[0].total}]")

    elapsed = time.perf_counter() - start_time
    print(f"\ninference completed in {elapsed:.2f} seconds")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    app()
