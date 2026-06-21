import os
from typing import List

import onnxruntime as ort
from inference import (
    draw_results,
    postprocess_det,
    postprocess_pose,
    preprocess_det,
    preprocess_pose,
)
from progress_info import ProgressInfo

from smartpool import TRT_CACHE_PATH


def infer_sequentially(det_model_path: str, pose_model_path: str, image_paths: List[str], output_dir: str, progress_info: ProgressInfo):
    providers = ort.get_available_providers()
    for i in range(len(providers)):
        if providers[i] == "TensorrtExecutionProvider":
            providers[i] = ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": TRT_CACHE_PATH
            })

    det_session = ort.InferenceSession(det_model_path, providers=providers)
    pose_session = ort.InferenceSession(pose_model_path, providers=providers)
    det_input_name = det_session.get_inputs()[0].name
    pose_input_name = pose_session.get_inputs()[0].name

    for image_path in image_paths:
        progress_info.start_one_det_preprocess()
        img, det_blob, scale, pad = preprocess_det(image_path)
        progress_info.finish_one_det_preprocess()

        progress_info.start_one_det_infer()
        raw_det = det_session.run(None, input_feed={det_input_name: det_blob})[0]
        progress_info.finish_one_det_infer()

        progress_info.start_one_det_postprocess()
        person_boxes = postprocess_det(raw_det, scale, pad, img.shape)
        progress_info.finish_one_det_postprocess()

        all_keypoints = []
        for box in person_boxes:
            progress_info.start_one_pose_preprocess()
            pose_blob, crop_info = preprocess_pose(img, box)
            progress_info.finish_one_pose_preprocess()

            progress_info.start_one_pose_infer()
            raw_pose = pose_session.run(None, input_feed={pose_input_name: pose_blob})[0]
            progress_info.finish_one_pose_infer()

            progress_info.start_one_pose_postprocess()
            kpts = postprocess_pose(raw_pose, crop_info)
            progress_info.finish_one_pose_postprocess()

            all_keypoints.append(kpts)

        output_path = output_dir + "/" + os.path.basename(image_path)

        progress_info.start_one_draw()
        draw_results(img, output_path, person_boxes, all_keypoints)
        progress_info.finish_one_draw()
