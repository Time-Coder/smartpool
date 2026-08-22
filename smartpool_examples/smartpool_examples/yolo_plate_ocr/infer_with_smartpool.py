from pathlib import Path
from functools import partial
from typing import Optional, Sequence

import cv2
import numpy as np

from smartpool import DataSize, InferSessionPool, Resource, ThreadPool

from inference import (
    annotate,
    create_tracker,
    decode_yolo_detections,
    decode_ocr_output,
    decode_yolo_pose,
    expand_box,
    load_ocr_keys,
    preprocess_image,
    preprocess_ocr_image,
    save_warped_plates,
    track_vehicle_boxes,
    PlateResult,
    warp_plate,
)
from .progress_info import PipelineProgress

CALCULATION_POOL = "calculation"
IO_POOL = "IO"


preprocess_image = ThreadPool.task(
    pool_name=CALCULATION_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=24 * DataSize.MB, result_cpu_mem=8 * DataSize.MB)
)(preprocess_image)

async_imwrite = ThreadPool.task(pool_name=IO_POOL)(cv2.imwrite)

@ThreadPool.task(
    pool_name=CALCULATION_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=64 * DataSize.MB, result_cpu_mem=8 * DataSize.MB)
)
def postprocess_vehicle(image, output, ratio, pad, tracker, vehicle_conf):
    boxes = decode_yolo_detections(output, image.shape, ratio, pad, vehicle_conf, {2, 3, 5, 7})
    vehicles = track_vehicle_boxes(tracker, boxes, image)
    return vehicles

@ThreadPool.task(
    pool_name=CALCULATION_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=64 * DataSize.MB, result_cpu_mem=8 * DataSize.MB)
)
def preprocess_ocr_candidate(crop_image, corners_crop):
    warped = warp_plate(crop_image, corners_crop)
    ocr_blob = preprocess_ocr_image(warped)
    return ocr_blob


@ThreadPool.task(
    pool_name=CALCULATION_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=64 * DataSize.MB, result_cpu_mem=8 * DataSize.MB),
)
def process_yolo_pose(
    plate_output,
    crop_image,
    crop_box,
    ratio,
    pad,
    plate_conf,
    ocr_infer_model,
    ocr_keys,
    vehicle,
    save_plate,
    progress
):
    x1, y1, x2, y2 = crop_box
    plates = []
    candidates = decode_yolo_pose(plate_output, crop_image.shape, ratio, pad, plate_conf)
    for idx, (confidence, corners_crop) in enumerate(candidates, start=1):
        corners_image = corners_crop + np.asarray([x1, y1], dtype=np.float32)
        ocr_blob_future = preprocess_ocr_candidate(crop_image, corners_crop)
        ocr_output = ocr_infer_model(ocr_blob_future)[0]
        text, _ = decode_ocr_output(ocr_output, ocr_keys)
        plate = PlateResult(
            vehicle=vehicle,
            crop_box=crop_box,
            corners_crop=corners_crop,
            corners_image=corners_image,
            warped=warped,
            text=text,
            confidence=confidence,
        )
        plates.append(plate)

        if save_plate:
            text = plate.text or "unknown"
            filename = f"{image_stem}_track{plate.vehicle.track_id}_plate{idx}_{text}.jpg"
            write_future = async_imwrite(str(plate_dir / filename), plate.warped)

    return plates

@ThreadPool.task(
    pool_name=CALCULATION_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=128 * DataSize.MB, result_cpu_mem=16 * DataSize.MB),
)
def save_warped_plates(output_dir: Path, image_stem: str, plates: Sequence[PlateResult]):
    plate_dir = output_dir / "plates"
    plate_dir.mkdir(parents=True, exist_ok=True)
    write_futures = []
    for idx, plate in enumerate(plates, start=1):
        text = plate.text or "unknown"
        filename = f"{image_stem}_track{plate.vehicle.track_id}_plate{idx}_{text}.jpg"
        write_future = async_imwrite(str(plate_dir / filename), plate.warped)
        write_futures.append(write_future)

    return write_futures

@ThreadPool.task(
    pool_name=CALCULATION_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=128 * DataSize.MB, result_cpu_mem=16 * DataSize.MB),
)
def detect_and_ocr_plate(
    output_name,
    image,
    vehicle,
    plate_infer_model,
    ocr_infer_model,
    ocr_keys,
    plate_conf,
    crop_padding,
    save_plates,
    output_dir,
    progress
):
    height, width = image.shape[:2]
    crop_box = expand_box(vehicle.box, width, height, crop_padding)
    x1, y1, x2, y2 = crop_box
    crop_image = image[y1:y2, x1:x2]
    if crop_image.size == 0:
        return

    preprocess_future = preprocess_image(crop_image)
    progress.attach_callbacks(preprocess_future, "preprocess vehicle")
    
    blob, ratio, pad = preprocess_future.unpack(3)
    plate_infer_future = plate_infer_model(blob)
    progress.attach_callbacks(plate_infer_future, "infer plate")

    plate_output = plate_infer_future[0]

    if save_plates:
        plate_dir = output_dir / "plates"
        plate_dir.mkdir(parents=True, exist_ok=True)
    
    plates_future = process_yolo_pose(
        plate_output,
        crop_image,
        crop_box,
        ratio,
        pad,
        plate_conf,
        ocr_infer_model,
        ocr_keys,
        vehicle,
        save_plates,
        progress
    )

    return plates_future

@ThreadPool.task(
    pool_name=CALCULATION_POOL,
    cpu_mode_res=Resource(cpu_cores=1, cpu_mem=128 * DataSize.MB, result_cpu_mem=16 * DataSize.MB),
)
def detect_and_ocr_plates_task(
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
    progress
):
    plates_futures = []
    write_futures = []
    progress.increase_total("preprocess vehicle start", len(vehicles))
    progress.increase_total("preprocess vehicle done", len(vehicles))
    progress.increase_total("infer plate start", len(vehicles))
    progress.increase_total("infer plate done", len(vehicles))
    for vehicle in vehicles:
        height, width = image.shape[:2]
        crop_box = expand_box(vehicle.box, width, height, crop_padding)
        x1, y1, x2, y2 = crop_box
        crop_image = image[y1:y2, x1:x2]
        if crop_image.size == 0:
            continue

        preprocess_future = preprocess_image(crop_image)
        progress.attach_callbacks(preprocess_future, "preprocess vehicle")
        
        blob, ratio, pad = preprocess_future.unpack(3)
        plate_infer_future = plate_infer_model(blob)
        progress.attach_callbacks(plate_infer_future, "infer plate")

        plate_output = plate_infer_future[0]
        plates_future = process_yolo_pose(
            plate_output,
            crop_image,
            crop_box,
            ratio,
            pad,
            plate_conf,
            ocr_infer_model,
            ocr_keys,
            vehicle
        )

        plates_futures.append(plates_future)
        if save_plates:
            write_future = save_warped_plates(output_dir, output_name.replace("/", "_"), plates_future)
            write_futures.append(write_future)

    return plates_futures, write_futures


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
    )

def submit_frame(
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
    progress
):
    preprocess_future = preprocess_image(image)
    progress.attach_callbacks(preprocess_future, "preprocess hole image")

    blob, ratio, pad = preprocess_future.unpack(3)
    vehicle_infer_future = vehicle_infer_model(blob)
    progress.attach_callbacks(vehicle_infer_future, "infer vehicles")

    vehicle_output = vehicle_infer_future[0]
    vehicles_future = postprocess_vehicle(
        image,
        vehicle_output,
        ratio,
        pad,
        tracker,
        vehicle_conf
    )
    progress.attach_callbacks(vehicles_future, "extract vehicles")

    plates_future = detect_and_ocr_plates_task(
        output_name,
        image,
        vehicles_future,
        plate_infer_model,
        ocr_infer_model,
        ocr_keys,
        plate_conf,
        crop_padding,
        save_plates,
        output_dir
    )
    annotated_future = annotate(image, vehicles_future, plates_future)

    if write_image:
        imwrite_future = async_imwrite(str(output_dir / output_name), annotated_future)
        return imwrite_future

    return annotated_future


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
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            future = submit_frame(
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
                progress
            )
            final_futures.append(future)

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
