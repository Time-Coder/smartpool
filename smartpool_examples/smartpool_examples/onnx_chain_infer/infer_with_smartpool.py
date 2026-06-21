import os
from concurrent.futures import as_completed
from typing import List

from inference import (
    draw_results,
    postprocess_det,
    postprocess_pose,
    preprocess_det,
    preprocess_pose,
)
from progress_info import ProgressInfo

from smartpool import DataSize, InferSessionPool, Resource, ThreadPool

preprocess_det = ThreadPool.task(
    pool_name="pre_post_det",
    cpu_mode_res=Resource(
        cpu_cores=1,
        cpu_mem=15 * DataSize.MB
    )
)(preprocess_det)

postprocess_det = ThreadPool.task(
    pool_name="pre_post_det",
    cpu_mode_res=Resource(
        cpu_cores=1,
        cpu_mem=15 * DataSize.MB
    )
)(postprocess_det)

preprocess_pose = ThreadPool.task(
    pool_name="pre_post_pose",
    cpu_mode_res=Resource(
        cpu_cores=1,
        cpu_mem=15 * DataSize.MB
    )
)(preprocess_pose)

postprocess_pose = ThreadPool.task(
    pool_name="pre_post_pose",
    cpu_mode_res=Resource(
        cpu_cores=1,
        cpu_mem=15 * DataSize.MB
    )
)(postprocess_pose)

draw_results = ThreadPool.task(
    pool_name="pre_post_det",
    cpu_mode_res=Resource(
        cpu_cores=1,
        cpu_mem=81 * DataSize.MB
    )
)(draw_results)


def infer_with_smartpool(det_model_path: str, pose_model_path: str, image_paths: List[str], output_dir: str, progress_info: ProgressInfo):
    cpu_res = Resource(
        cpu_cores_in_python=0,
        cpu_cores_out_of_python=1,
        cpu_mem=81 * DataSize.MB
    )
    gpu_res = Resource(
        cpu_cores_in_python=0,
        cpu_cores_out_of_python=1,
        cpu_mem=128 * DataSize.MB,
        gpu_cores=100,
        gpu_mem=600 * DataSize.MB
    )

    infer_session_pool = InferSessionPool()

    all_result_futures = []
    for image_path in image_paths:
        det_pre_future = preprocess_det(image_path)
        det_pre_future.add_start_callback(progress_info.start_one_det_preprocess)
        det_pre_future.add_done_callback(progress_info.finish_one_det_preprocess)

        img, det_blob, scale, pad = det_pre_future.unpack(4)

        det_infer_future = infer_session_pool.submit(
            det_model_path, args=(det_blob,),
            cpu_mode_res=cpu_res, gpu_mode_res=gpu_res, use_io_binding=False
        )
        det_infer_future.add_start_callback(progress_info.start_one_det_infer)
        det_infer_future.add_done_callback(progress_info.finish_one_det_infer)

        raw_det = det_infer_future[0]

        det_post_future = postprocess_det(raw_det, scale, pad, img.shape)
        det_post_future.add_start_callback(progress_info.start_one_det_postprocess)
        det_post_future.add_done_callback(progress_info.finish_one_det_postprocess)

        person_boxes = det_post_future.result()

        pose_infer_futures = []
        for box in person_boxes:
            pose_pre_future = preprocess_pose(img, box)
            pose_pre_future.add_start_callback(progress_info.start_one_pose_preprocess)
            pose_pre_future.add_done_callback(progress_info.finish_one_pose_preprocess)

            pose_blob, crop_info = pose_pre_future.unpack(2)

            pose_infer_future = infer_session_pool.submit(
                pose_model_path, args=(pose_blob,),
                cpu_mode_res=cpu_res, gpu_mode_res=gpu_res, use_io_binding=False
            )
            pose_infer_future.add_start_callback(progress_info.start_one_pose_infer)
            pose_infer_future.add_done_callback(progress_info.finish_one_pose_infer)

            raw_pose = pose_infer_future[0]

            pose_post_future = postprocess_pose(raw_pose, crop_info)
            pose_post_future.add_start_callback(progress_info.start_one_pose_postprocess)
            pose_post_future.add_done_callback(progress_info.finish_one_pose_postprocess)

            pose_infer_futures.append(pose_post_future)

        all_keypoints = [f.result() for f in pose_infer_futures]

        output_path = output_dir + "/" + os.path.basename(image_path)
        draw_future = draw_results(img, output_path, person_boxes, all_keypoints)
        draw_future.add_start_callback(progress_info.start_one_draw)
        draw_future.add_done_callback(progress_info.finish_one_draw)

        all_result_futures.append(draw_future)

    for f in as_completed(all_result_futures):
        f.result()
