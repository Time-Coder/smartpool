from typing import List
import os
from ultralytics import YOLO

from progress_info import ProgressInfo


def infer_sequentially_with_ultralytics(model_path: str, image_paths: List[str], output_dir: str, progress_info: ProgressInfo):
    model = YOLO("yolov8n.pt")

    for image_path in image_paths:
        progress_info.start_one_preprocess()
        progress_info.finish_one_preprocess()

        progress_info.start_one_infer()
        results = model(image_path)
        progress_info.finish_one_infer()

        progress_info.start_one_postprocess()
        output_path = output_dir + "/" + os.path.basename(image_path)
        for r in results:
            r.save(filename=output_path)
        progress_info.finish_one_postprocess()
