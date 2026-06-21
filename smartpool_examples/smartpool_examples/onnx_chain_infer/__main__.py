if __name__ == "__main__":
    import os
    import sys
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)

from enum import Enum

import typer

app = typer.Typer(help="Chain YOLOv8n detection + pose estimation on COCO val2017 with real-time progress.")

class Method(Enum):
    SmartPool = "smartpool"
    Sequentially = "sequentially"

@app.command()
def main(
    method: Method = typer.Option(
        Method.SmartPool,
        "--method"
    )
):
    import os

    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

    print("Use `python -m smartpool_examples.onnx_chain_infer --help` to see all options.")
    print(f"See source code at folder {self_folder}")
    print()

    try:
        import onnxruntime
    except ImportError:
        print("ONNX Runtime is not installed. Follow https://onnxruntime.ai/docs/install/ instructions to install ONNX Runtime.")
        return

    import glob
    import time

    sys.path.append(self_folder)

    from config import DATASET_FOLDER, DET_MODEL_PATH, OUTPUT_FOLDER, POSE_MODEL_PATH
    from data_utils import download_dataset, download_model
    from progress_info import ProgressInfo

    download_model()
    download_dataset()

    image_paths = glob.glob(DATASET_FOLDER + "/*.jpg")
    n_tasks = len(image_paths)

    progress_info = ProgressInfo(n_tasks)

    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    with progress_info:
        start_time = time.perf_counter()
        if method == Method.SmartPool:
            from infer_with_smartpool import infer_with_smartpool
            infer_with_smartpool(DET_MODEL_PATH, POSE_MODEL_PATH, image_paths, OUTPUT_FOLDER, progress_info)
        elif method == Method.Sequentially:
            from infer_sequentially import infer_sequentially
            infer_sequentially(DET_MODEL_PATH, POSE_MODEL_PATH, image_paths, OUTPUT_FOLDER, progress_info)

    elapsed = time.perf_counter() - start_time
    print(f"\ninference completed in {elapsed:.2f} seconds")
    print(f"Output: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    app()
