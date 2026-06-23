from typing import List
import os
import cv2
from ultralytics import YOLO

from progress_info import ProgressInfo


def infer_sequentially_with_ultralytics(model_path: str, image_paths: List[str], output_dir: str, progress_info: ProgressInfo):
    model = YOLO(model_path)
    model.verbose = False

    _ = model(image_paths[0])
    predictor = model.predictor

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue

        progress_info.start_one_preprocess()
        im_tensor = predictor.preprocess([img])
        progress_info.finish_one_preprocess()

        progress_info.start_one_infer()
        preds = predictor.inference(im_tensor)
        progress_info.finish_one_infer()

        progress_info.start_one_postprocess()
        output_path = output_dir + "/" + os.path.basename(image_path)
        predictor.batch = ([image_path], [img], [""])
        results = predictor.postprocess(preds, im_tensor, [img])
        results[0].save(filename=output_path)
        progress_info.finish_one_postprocess()
