from pathlib import Path
from typing import Optional

import cv2
import onnxruntime as ort

from .data_utils import is_video_file, iter_images
from .inference import (
    annotate,
    create_tracker,
    decode_yolo_detections,
    detect_plates_for_vehicle_onnx,
    load_ocr_keys,
    preprocess_image,
    save_warped_plates,
    track_vehicle_boxes,
)
from .progress_info import PipelineProgress


def process_frame_with_ort(
    image,
    vehicle_session,
    plate_session,
    ocr_session,
    ocr_keys,
    tracker,
    vehicle_conf: float,
    plate_conf: float,
    crop_padding: float,
):
    vehicle_input_name = vehicle_session.get_inputs()[0].name
    blob, ratio, pad = preprocess_image(image)
    vehicle_outputs = vehicle_session.run(None, {vehicle_input_name: blob})
    boxes = decode_yolo_detections(vehicle_outputs[0], image.shape, ratio, pad, vehicle_conf, {2, 3, 5, 7})
    vehicles = track_vehicle_boxes(tracker, boxes, image)

    plates = []
    for vehicle in vehicles:
        plates.extend(
            detect_plates_for_vehicle_onnx(
                plate_session,
                ocr_session,
                ocr_keys,
                image,
                vehicle,
                plate_conf,
                crop_padding,
            )
        )
    annotated = annotate(image, vehicles, plates)
    return annotated, vehicles, plates


def infer_sequentially(
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
    progress: Optional[PipelineProgress] = None,
    video_output: Optional[Path] = None,
):
    vehicle_session = ort.InferenceSession(str(vehicle_model_path), providers=["CPUExecutionProvider"])
    plate_session = ort.InferenceSession(str(plate_model_path), providers=["CPUExecutionProvider"])
    ocr_session = ort.InferenceSession(str(ocr_model_path), providers=["CPUExecutionProvider"])
    ocr_keys = load_ocr_keys(ocr_keys_path)

    if is_video_file(source):
        return infer_video_sequentially(
            source,
            output_dir,
            vehicle_session,
            plate_session,
            ocr_session,
            ocr_keys,
            vehicle_conf,
            plate_conf,
            crop_padding,
            save_plates,
            progress,
            video_output,
        )

    return infer_images_sequentially(
        iter_images(source),
        output_dir,
        vehicle_session,
        plate_session,
        ocr_session,
        ocr_keys,
        vehicle_conf,
        plate_conf,
        crop_padding,
        save_plates,
        progress,
    )


def infer_images_sequentially(
    image_paths,
    output_dir,
    vehicle_session,
    plate_session,
    ocr_session,
    ocr_keys,
    vehicle_conf,
    plate_conf,
    crop_padding,
    save_plates,
    progress,
):
    tracker = create_tracker()
    total_vehicles = 0
    total_plates = 0
    owns_progress = progress is None
    if progress is None:
        progress = PipelineProgress(len(image_paths))

    with progress if owns_progress else _null_context(progress):
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            progress.start_one_vehicle()
            annotated, vehicles, plates = process_frame_with_ort(
                image,
                vehicle_session,
                plate_session,
                ocr_session,
                ocr_keys,
                tracker,
                vehicle_conf,
                plate_conf,
                crop_padding,
            )
            progress.finish_one_vehicle()
            progress.start_one_plate()
            progress.finish_one_plate()
            total_vehicles += len(vehicles)
            total_plates += len(plates)

            progress.start_one_save()
            cv2.imwrite(str(output_dir / image_path.name), annotated)
            if save_plates:
                save_warped_plates(output_dir, image_path.stem, plates)
            progress.finish_one_save()

    return total_vehicles, total_plates


def infer_video_sequentially(
    source,
    output_dir,
    vehicle_session,
    plate_session,
    ocr_session,
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

    tracker = create_tracker(int(capture.get(cv2.CAP_PROP_FPS)) or 30)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_output is None:
        video_output = output_dir / f"{source.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open video writer: {video_output}")

    frame_index = 0
    total_vehicles = 0
    total_plates = 0
    owns_progress = progress is None
    if progress is None:
        progress = PipelineProgress(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None)

    with progress if owns_progress else _null_context(progress):
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            progress.start_one_vehicle()
            annotated, vehicles, plates = process_frame_with_ort(
                frame,
                vehicle_session,
                plate_session,
                ocr_session,
                ocr_keys,
                tracker,
                vehicle_conf,
                plate_conf,
                crop_padding,
            )
            progress.finish_one_vehicle()
            progress.start_one_plate()
            progress.finish_one_plate()
            total_vehicles += len(vehicles)
            total_plates += len(plates)

            progress.start_one_save()
            writer.write(annotated)
            if save_plates:
                save_warped_plates(output_dir, f"{source.stem}_frame{frame_index:06d}", plates)
            progress.finish_one_save()
            frame_index += 1

    capture.release()
    writer.release()
    return frame_index, total_vehicles, total_plates, video_output


class _null_context:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, tb):
        return False
