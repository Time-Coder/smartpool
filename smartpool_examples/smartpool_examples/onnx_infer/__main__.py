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
    import queue
    from functools import partial
    from smartpool import InferSessionPool, Resource, ThreadPool, DataSize
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )
    from inference import preprocess, postprocess
    
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
        cpu_mem=81*DataSize.MB
    )
    gpu_res = Resource(
        cpu_cores_in_python=1,
        cpu_cores_out_of_python=1,
        cpu_mem=200*DataSize.MB,
        gpu_cores=100,
        gpu_mem=200*DataSize.MB
    )

    result_queue = queue.Queue()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        n_tasks = len(image_paths)
        preprocess_submit_progress_task = progress.add_task("preprocess submit", total=n_tasks)
        preprocess_done_progress_task = progress.add_task("preprocess done", total=n_tasks)
        infer_submit_progress_task = progress.add_task("infer submit", total=n_tasks)
        infer_done_progress_task = progress.add_task("infer done", total=n_tasks)
        postprocess_submit_progress_task = progress.add_task("postprocess submit", total=n_tasks)
        postprocess_done_progress_task = progress.add_task("postprocess done", total=n_tasks)

        def postprocess_done_callback(postprocess_future: Future):
            postprocess_future.result()
            progress.update(postprocess_done_progress_task, advance=1)
            result_queue.put(1)

        def infer_done_callback(image_path, img, scale, pad, infer_future: Future):
            outputs = infer_future.result()[0]
            progress.update(infer_done_progress_task, advance=1)

            output_path = output_dir_str + "/" + os.path.basename(image_path)
            postprocess_future: Future = thread_pool.submit(
                postprocess,
                args=(img, outputs, output_path, scale, pad),
                cpu_mode_res=cpu_res
            )
            progress.update(postprocess_submit_progress_task, advance=1)
            postprocess_future.add_done_callback(postprocess_done_callback)

        def preprocess_done_callback(image_path, preprocess_future: Future):
            progress.update(preprocess_done_progress_task, advance=1)
            img, blob, scale, pad = preprocess_future.result()
            infer_future: Future = infer_session_pool.submit(args=(blob,), providers=["CPUExecutionProvider"], cpu_mode_res=cpu_res, gpu_mode_res=gpu_res)
            progress.update(infer_submit_progress_task, advance=1)
            infer_future.add_done_callback(partial(infer_done_callback, image_path, img, scale, pad))
        
        for image_path in image_paths:
            image_path = str(image_path)
            preprocess_future: Future = thread_pool.submit(
                preprocess,
                args=(image_path,),
                cpu_mode_res=cpu_res
            )
            progress.update(preprocess_submit_progress_task, advance=1)
            preprocess_future.add_done_callback(partial(preprocess_done_callback, image_path))

        start_time = time.perf_counter()

        for _ in range(n_tasks):
            result_queue.get()
            
    elapsed = time.perf_counter() - start_time
    print(f"\ninference completed in {elapsed:.2f} seconds")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    app()
