from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from smartpool import DataSize, InferSessionPool, Resource, ThreadPool

from .inference import (
    annotate,
    create_tracker,
    decode_yolo_detections,
    decode_ocr_output,
    decode_yolo_pose,
    expand_box,
    load_ocr_keys,
    preprocess_frame,
    preprocess_image,
    preprocess_ocr_image,
    save_warped_plates,
    track_vehicle_boxes,
    PlateResult,
    warp_plate,
)
from .progress_info import PipelineProgress

PREPROCESS_POOL = "yolo_plate_ocr_preprocess"
POSTPROCESS_POOL = "yolo_plate_ocr_postprocess"
PLATE_POOL = "yolo_plate_ocr_plate"
RENDER_POOL = "yolo_plate_ocr_render"


@ThreadPool.task(
    pool_name=PREPROCESS_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=24 * DataSize.MB, result_cpu_mem=8 * DataSize.MB),
    check_args=False,
)
def preprocess_task(frame_index, output_name, image):
    return preprocess_frame(frame_index, output_name, image)


@ThreadPool.task(
    pool_name=POSTPROCESS_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=64 * DataSize.MB, result_cpu_mem=8 * DataSize.MB),
    check_args=False,
)
def postprocess_vehicle_task(frame_index, output_name, image, output, ratio, pad, tracker, vehicle_conf):
    boxes = decode_yolo_detections(output, image.shape, ratio, pad, vehicle_conf, {2, 3, 5, 7})
    vehicles = track_vehicle_boxes(tracker, boxes, image)
    return frame_index, output_name, image, vehicles


@ThreadPool.task(
    pool_name=PLATE_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=128 * DataSize.MB, result_cpu_mem=16 * DataSize.MB),
    check_args=False,
)
def detect_and_ocr_plates_task(
    frame_index,
    output_name,
    image,
    vehicles,
    plate_infer_model,
    ocr_infer_model,
    ocr_keys,
    plate_conf,
    crop_padding,
    save_plates,
    output_dir,
):
    plates = []
    for vehicle in vehicles:
        height, width = image.shape[:2]
        crop_box = expand_box(vehicle.box, width, height, crop_padding)
        x1, y1, x2, y2 = crop_box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        blob, ratio, pad = preprocess_image(crop)
        infer_future = plate_infer_model(blob)
        output = infer_future[0].result()
        candidates = decode_yolo_pose(output, crop.shape, ratio, pad, plate_conf)
        for confidence, corners_crop in candidates:
            corners_image = corners_crop + np.asarray([x1, y1], dtype=np.float32)
            warped = warp_plate(crop, corners_crop)
            ocr_blob = preprocess_ocr_image(warped)
            ocr_future = ocr_infer_model(ocr_blob)
            text, _ = decode_ocr_output(ocr_future[0].result(), ocr_keys)
            plates.append(
                PlateResult(
                    vehicle=vehicle,
                    crop_box=crop_box,
                    corners_crop=corners_crop,
                    corners_image=corners_image,
                    warped=warped,
                    text=text,
                    confidence=confidence,
                )
            )

    if save_plates:
        save_warped_plates(output_dir, output_name.replace("/", "_"), plates)

    return frame_index, output_name, image, vehicles, plates


@ThreadPool.task(
    pool_name=RENDER_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=64 * DataSize.MB, result_cpu_mem=8 * DataSize.MB),
    check_args=False,
)
def render_task(frame_index, output_name, image, vehicles, plates, output_dir, write_image):
    annotated = annotate(image, vehicles, plates)
    if write_image:
        cv2.imwrite(str(output_dir / output_name), annotated)
    return frame_index, output_name, annotated, len(vehicles), len(plates)


def configure_pools(max_workers: int = 4):
    ThreadPool.config(pool_name=PREPROCESS_POOL, max_workers=max_workers)
    ThreadPool.config(pool_name=POSTPROCESS_POOL, max_workers=1)
    ThreadPool.config(pool_name=PLATE_POOL, max_workers=max_workers)
    ThreadPool.config(pool_name=RENDER_POOL, max_workers=max_workers)
    InferSessionPool.config(pool_name="yolo_plate_ocr_infer", max_workers=max_workers)


def create_infer_model(model_path: Path, gpu_mem: int = 64 * DataSize.MB):
    cpu_res = Resource(
        cpu_cores_in_python=0,
        cpu_cores_out_of_python=1,
        cpu_mem=128 * DataSize.MB,
        result_cpu_mem=8 * DataSize.MB,
    )
    gpu_res = Resource(
        cpu_cores_in_python=0,
        cpu_cores_out_of_python=1,
        cpu_mem=128 * DataSize.MB,
        gpu_cores=100,
        gpu_mem=gpu_mem,
        result_cpu_mem=8 * DataSize.MB,
        result_gpu_mem=8 * DataSize.MB,
    )
    return InferSessionPool.model(
        str(model_path),
        pool_name="yolo_plate_ocr_infer",
        cpu_mode_res=cpu_res,
        gpu_mode_res=gpu_res,
        check_args=False,
    )


def submit_frame(
    frame_index,
    output_name,
    image,
    vehicle_infer_model,
    plate_infer_model,
    ocr_infer_model,
    ocr_keys,
    tracker,
    vehicle_conf,
    plate_conf,
    crop_padding,
    output_dir,
    save_plates,
    write_image,
):
    preprocess_future = preprocess_task(frame_index, output_name, image)
    frame_index_f, output_name_f, image_f, blob_f, ratio_f, pad_f = preprocess_future.unpack(6)
    vehicle_infer_future = vehicle_infer_model(blob_f)
    vehicle_output_f = vehicle_infer_future[0]
    vehicle_future = postprocess_vehicle_task(
        frame_index_f,
        output_name_f,
        image_f,
        vehicle_output_f,
        ratio_f,
        pad_f,
        tracker,
        vehicle_conf,
    )
    plate_future = detect_and_ocr_plates_task(
        *vehicle_future.unpack(4),
        plate_infer_model,
        ocr_infer_model,
        ocr_keys,
        plate_conf,
        crop_padding,
        save_plates,
        output_dir,
    )
    render_future = render_task(
        *plate_future.unpack(5),
        output_dir,
        write_image,
    )
    return preprocess_future, vehicle_infer_future, vehicle_future, plate_future, render_future


def attach_callbacks(futures, progress: PipelineProgress):
    preprocess_future, vehicle_infer_future, vehicle_future, plate_future, render_future = futures
    vehicle_infer_future.add_start_callback(progress.start_one_vehicle)
    vehicle_future.add_done_callback(lambda _: progress.finish_one_vehicle())
    plate_future.add_start_callback(progress.start_one_plate)
    plate_future.add_done_callback(lambda _: progress.finish_one_plate())
    render_future.add_start_callback(progress.start_one_save)
    render_future.add_done_callback(lambda _: progress.finish_one_save())


def infer_with_smartpool(
    source: Path,
    output_dir: Path,
    vehicle_model_path: Path,
    plate_model_path: Path,
    ocr_model_path: Path,
    ocr_keys_path: Path,
    vehicle_conf: float,
    plate_conf: float,
    crop_padding: float,
    save_plates: bool,
    progress: PipelineProgress,
    video_output: Optional[Path] = None,
):
    from .data_utils import is_video_file, iter_images

    configure_pools()
    output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_infer_model = create_infer_model(vehicle_model_path)
    plate_infer_model = create_infer_model(plate_model_path)
    ocr_infer_model = create_infer_model(ocr_model_path, gpu_mem=16 * DataSize.MB)
    ocr_keys = load_ocr_keys(ocr_keys_path)

    if is_video_file(source):
        return infer_video_with_smartpool(
            source,
            output_dir,
            vehicle_infer_model,
            plate_infer_model,
            ocr_infer_model,
            ocr_keys,
            vehicle_conf,
            plate_conf,
            crop_padding,
            save_plates,
            progress,
            video_output,
        )

    return infer_images_with_smartpool(
        iter_images(source),
        output_dir,
        vehicle_infer_model,
        plate_infer_model,
        ocr_infer_model,
        ocr_keys,
        vehicle_conf,
        plate_conf,
        crop_padding,
        save_plates,
        progress,
    )


def infer_images_with_smartpool(
    image_paths,
    output_dir,
    vehicle_infer_model,
    plate_infer_model,
    ocr_infer_model,
    ocr_keys,
    vehicle_conf,
    plate_conf,
    crop_padding,
    save_plates,
    progress,
):
    tracker = create_tracker()
    final_futures = []
    with progress:
        for frame_index, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            futures = submit_frame(
                frame_index,
                image_path.name,
                image,
                vehicle_infer_model,
                plate_infer_model,
                ocr_infer_model,
                ocr_keys,
                tracker,
                vehicle_conf,
                plate_conf,
                crop_padding,
                output_dir,
                save_plates,
                True,
            )
            attach_callbacks(futures, progress)
            final_futures.append(futures[-1])

        total_vehicles = 0
        total_plates = 0
        for future in final_futures:
            _, _, _, vehicle_count, plate_count = future.result()
            total_vehicles += vehicle_count
            total_plates += plate_count

    return total_vehicles, total_plates


def infer_video_with_smartpool(
    source,
    output_dir,
    vehicle_infer_model,
    plate_infer_model,
    ocr_infer_model,
    ocr_keys,
    vehicle_conf,
    plate_conf,
    crop_padding,
    save_plates,
    progress,
    video_output,
):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = create_tracker(int(fps) or 30)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_output is None:
        video_output = output_dir / f"{source.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open video writer: {video_output}")

    final_futures = []
    frame_index = 0
    total_vehicles = 0
    total_plates = 0
    with progress:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            futures = submit_frame(
                frame_index,
                f"{source.stem}_frame{frame_index:06d}.jpg",
                image,
                vehicle_infer_model,
                plate_infer_model,
                ocr_infer_model,
                ocr_keys,
                tracker,
                vehicle_conf,
                plate_conf,
                crop_padding,
                output_dir,
                save_plates,
                False,
            )
            attach_callbacks(futures, progress)
            final_futures.append(futures[-1])
            frame_index += 1

        for future in final_futures:
            _, _, annotated, vehicle_count, plate_count = future.result()
            writer.write(annotated)
            total_vehicles += vehicle_count
            total_plates += plate_count

    capture.release()
    writer.release()
    return len(final_futures), total_vehicles, total_plates, video_output
