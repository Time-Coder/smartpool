if __name__ == "__main__":
    import os
    import sys
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)

import typer
from enum import Enum

app = typer.Typer(help="Use different methods to do YOLOv8n ONNX inference on COCO val2017 with real-time progress.")

class Method(Enum):
    SmartPool = "smartpool"
    Sequentially = "sequentially"
    Ultralytics = "ultralytics"
    Ray = "ray"

@app.command()
def main(
    method: Method = typer.Option(
        Method.SmartPool,
        "--method"
    )
):
    import os

    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

    print("Use `python -m smartpool_examples.onnx_infer --help` to see all options.")
    print(f"See source code at folder {self_folder}")
    print()

    try:
        import onnxruntime
    except ImportError:
        print("ONNX Runtime is not installed. Follow https://onnxruntime.ai/docs/install/ instructions to install ONNX Runtime.")
        return

    import time
    import glob

    sys.path.append(self_folder)

    from config import DATASET_FOLDER, MODEL_PATH, OUTPUT_FOLDER
    from data_utils import download_dataset, download_model
    from progress_info import ProgressInfo

    download_model()
    download_dataset()

    image_paths = glob.glob(DATASET_FOLDER + "/*.jpg")
    image_paths = image_paths[:4]
    n_tasks = len(image_paths)

    progress_info = ProgressInfo(n_tasks)

    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    with progress_info:
        start_time = time.perf_counter()
        if method == Method.SmartPool:
            from infer_with_smartpool import infer_with_smartpool
            infer_with_smartpool(MODEL_PATH, image_paths, OUTPUT_FOLDER, progress_info)
        elif method == Method.Sequentially:
            from infer_sequentially import infer_sequentially
            infer_sequentially(MODEL_PATH, image_paths, OUTPUT_FOLDER, progress_info)
        elif method == Method.Ultralytics:
            from infer_sequentially_with_ultralytics import infer_sequentially_with_ultralytics
            infer_sequentially_with_ultralytics(MODEL_PATH, image_paths, OUTPUT_FOLDER, progress_info)
        elif method == Method.Ray:
            from infer_with_ray import infer_with_ray
            infer_with_ray(MODEL_PATH, image_paths, OUTPUT_FOLDER, progress_info)

    elapsed = time.perf_counter() - start_time
    print(f"\ninference completed in {elapsed:.2f} seconds")
    print(f"Output: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    app()
