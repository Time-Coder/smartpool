import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from .config import VEHICLE_CLASS_IDS


@dataclass
class VehicleTrack:
    box: Tuple[int, int, int, int]
    track_id: int
    confidence: float
    cls_id: int


@dataclass
class PlateResult:
    vehicle: VehicleTrack
    crop_box: Tuple[int, int, int, int]
    corners_crop: "np.ndarray"
    corners_image: "np.ndarray"
    warped: "np.ndarray"
    text: str
    confidence: float


class DetectionBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    @property
    def xywh(self):
        boxes = self.xyxy
        xywh = np.empty_like(boxes, dtype=np.float32)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        return xywh

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, item):
        return DetectionBoxes(self.xyxy[item], self.conf[item], self.cls[item])


class TrackerArgs:
    tracker_type = "bytetrack"
    track_high_thresh = 0.25
    track_low_thresh = 0.1
    new_track_thresh = 0.25
    track_buffer = 30
    match_thresh = 0.8
    fuse_score = True


def create_tracker(frame_rate: int = 30):
    _ = frame_rate
    return BYTETracker(TrackerArgs())


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, ratio, (left, top)


def preprocess_image(image: np.ndarray, image_size: int = 640):
    blob, ratio, pad = letterbox(image, (image_size, image_size))
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
    blob = blob.astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])
    return blob, ratio, pad


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def decode_yolo_detections(
    output,
    image_shape,
    ratio: float,
    pad,
    conf_threshold: float,
    class_ids=None,
    iou_threshold: float = 0.5,
):
    pred = np.squeeze(output)
    if pred.ndim != 2:
        return DetectionBoxes([], [], [])
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes = pred[:, :4]
    class_scores = pred[:, 4:]
    scores = class_scores.max(axis=1)
    classes = class_scores.argmax(axis=1)
    mask = scores >= conf_threshold
    if class_ids is not None:
        mask &= np.isin(classes, list(class_ids))
    if not mask.any():
        return DetectionBoxes([], [], [])

    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    boxes_xyxy = np.empty_like(boxes, dtype=np.float32)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(boxes_xyxy, scores, iou_threshold)
    if not keep:
        return DetectionBoxes([], [], [])

    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    classes = classes[keep]

    left, top = pad
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - left) / ratio
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - top) / ratio
    height, width = image_shape[:2]
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, width)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, height)
    return DetectionBoxes(boxes_xyxy, scores, classes)


def decode_yolo_pose(output, crop_shape, ratio: float, pad, conf_threshold: float, iou_threshold: float = 0.5):
    pred = np.squeeze(output)
    if pred.ndim != 2:
        return []
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    box_data = pred[:, :4]
    scores = pred[:, 4]
    keypoint_data = pred[:, 5:]
    mask = scores >= conf_threshold
    if not mask.any():
        return []

    box_data = box_data[mask]
    scores = scores[mask]
    keypoint_data = keypoint_data[mask]
    boxes_xyxy = np.empty_like(box_data, dtype=np.float32)
    boxes_xyxy[:, 0] = box_data[:, 0] - box_data[:, 2] / 2
    boxes_xyxy[:, 1] = box_data[:, 1] - box_data[:, 3] / 2
    boxes_xyxy[:, 2] = box_data[:, 0] + box_data[:, 2] / 2
    boxes_xyxy[:, 3] = box_data[:, 1] + box_data[:, 3] / 2
    keep = nms(boxes_xyxy, scores, iou_threshold)

    candidates = []
    left, top = pad
    for idx in keep:
        points = keypoint_data[idx].reshape(-1, 3)[:4, :2].astype(np.float32)
        points[:, 0] = (points[:, 0] - left) / ratio
        points[:, 1] = (points[:, 1] - top) / ratio
        height, width = crop_shape[:2]
        points[:, 0] = np.clip(points[:, 0], 0, width)
        points[:, 1] = np.clip(points[:, 1], 0, height)
        candidates.append((float(scores[idx]), points))
    return candidates


def track_vehicle_boxes(tracker, boxes: DetectionBoxes, image) -> List[VehicleTrack]:
    if len(boxes) == 0:
        return []

    tracks = tracker.update(boxes, image)
    vehicles = []
    for track in tracks:
        if len(track) < 7:
            continue
        x1, y1, x2, y2, track_id, score, cls_id = track[:7]
        vehicles.append(
            VehicleTrack(
                box=clamp_box((x1, y1, x2, y2), image.shape[1], image.shape[0]),
                track_id=int(track_id),
                confidence=float(score),
                cls_id=int(cls_id),
            )
        )
    return vehicles


def preprocess_frame(frame_index, output_name, image):
    blob, ratio, pad = preprocess_image(image)
    return frame_index, output_name, image, blob, ratio, pad


def postprocess_vehicle_onnx(frame_index, output_name, image, output, ratio, pad, tracker, vehicle_conf):
    boxes = decode_yolo_detections(output, image.shape, ratio, pad, vehicle_conf, VEHICLE_CLASS_IDS)
    vehicles = track_vehicle_boxes(tracker, boxes, image)
    return frame_index, output_name, image, vehicles


def run_plate_onnx_session(session, crop, plate_conf):
    blob, ratio, pad = preprocess_image(crop)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    return decode_yolo_pose(outputs[0], crop.shape, ratio, pad, plate_conf)


def load_ocr_keys(keys_path: Path):
    chars = [line.strip("\n\r") for line in Path(keys_path).read_text(encoding="utf-8").splitlines()]
    if " " not in chars:
        chars.append(" ")
    return ["blank"] + chars


def preprocess_ocr_image(image: np.ndarray, image_shape=(3, 48, 320)):
    img_c, img_h, img_w = image_shape
    height, width = image.shape[:2]
    ratio = width / float(height)
    resized_w = min(img_w, max(1, int(math.ceil(img_h * ratio))))
    resized = cv2.resize(image, (resized_w, img_h))
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - 0.5) / 0.5
    resized = resized.transpose(2, 0, 1)
    padded = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded[np.newaxis, :]


def decode_ocr_output(output, keys):
    pred = np.asarray(output)
    if pred.ndim == 3:
        pred = pred[0]
    indices = pred.argmax(axis=1)
    scores = pred.max(axis=1)
    chars = []
    confs = []
    last_idx = None
    for idx, score in zip(indices, scores):
        idx = int(idx)
        if idx != 0 and idx != last_idx and idx < len(keys):
            chars.append(keys[idx])
            confs.append(float(score))
        last_idx = idx
    text = "".join(chars).upper()
    text = re.sub(r"[^0-9A-Z\u4e00-\u9fff]", "", text)
    confidence = sum(confs) / len(confs) if confs else 0.0
    return text, confidence


def run_ocr_onnx_session(session, image, keys):
    blob = preprocess_ocr_image(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    text, _ = decode_ocr_output(outputs[0], keys)
    return text


def clamp_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(math.floor(x1))))
    y1 = max(0, min(height - 1, int(math.floor(y1))))
    x2 = max(0, min(width, int(math.ceil(x2))))
    y2 = max(0, min(height, int(math.ceil(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def expand_box(box: Sequence[float], width: int, height: int, ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * ratio
    pad_y = bh * ratio
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def detect_vehicles(result, image_shape) -> List[VehicleTrack]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    height, width = image_shape[:2]
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    vehicles = []
    for idx, cls_id in enumerate(clss):
        if int(cls_id) not in VEHICLE_CLASS_IDS:
            continue
        track_id = int(ids[idx]) if ids is not None else idx + 1
        vehicles.append(
            VehicleTrack(
                box=clamp_box(xyxy[idx], width, height),
                track_id=track_id,
                confidence=float(confs[idx]),
                cls_id=int(cls_id),
            )
        )
    return vehicles


def extract_plate_candidates(result, min_conf: float):
    if result is None or result.keypoints is None or result.boxes is None:
        return []

    boxes = result.boxes
    keypoints = result.keypoints
    points = keypoints.xy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    candidates = []
    for idx, corners in enumerate(points):
        if corners.shape[0] < 4:
            continue
        confidence = float(confs[idx])
        if confidence < min_conf:
            continue
        corners = np.asarray(corners[:4], dtype=np.float32)
        if np.any(~np.isfinite(corners)):
            continue
        candidates.append((confidence, corners))
    return candidates


def order_corners(corners: "np.ndarray") -> "np.ndarray":
    corners = np.asarray(corners, dtype=np.float32)
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = corners[np.argmin(sums)]
    ordered[2] = corners[np.argmax(sums)]
    ordered[1] = corners[np.argmin(diffs)]
    ordered[3] = corners[np.argmax(diffs)]
    return ordered


def warp_plate(image: "np.ndarray", corners: "np.ndarray") -> "np.ndarray":
    corners = order_corners(corners)
    tl, tr, br, bl = corners
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    width = max(80, int(max(width_a, width_b)))
    height = max(24, int(max(height_a, height_b)))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def build_plate_results_from_candidates_onnx(candidates, image, vehicle: VehicleTrack, crop_box, crop, ocr_session, ocr_keys):
    plate_results = []
    x1, y1, _, _ = crop_box
    for confidence, corners_crop in candidates:
        corners_image = corners_crop + np.asarray([x1, y1], dtype=np.float32)
        warped = warp_plate(crop, corners_crop)
        text = run_ocr_onnx_session(ocr_session, warped, ocr_keys)
        plate_results.append(
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
    return plate_results


def detect_plates_for_vehicle_ultralytics_onnx_ocr(
    plate_model,
    ocr_session,
    ocr_keys,
    image,
    vehicle: VehicleTrack,
    plate_conf: float,
    crop_padding: float,
) -> List[PlateResult]:
    height, width = image.shape[:2]
    crop_box = expand_box(vehicle.box, width, height, crop_padding)
    x1, y1, x2, y2 = crop_box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    results = plate_model.predict(crop, conf=plate_conf, verbose=False)
    candidates = extract_plate_candidates(results[0] if results else None, plate_conf)
    return build_plate_results_from_candidates_onnx(candidates, image, vehicle, crop_box, crop, ocr_session, ocr_keys)


def detect_plates_for_vehicle_onnx(
    plate_session,
    ocr_session,
    ocr_keys,
    image,
    vehicle: VehicleTrack,
    plate_conf: float,
    crop_padding: float,
) -> List[PlateResult]:
    height, width = image.shape[:2]
    crop_box = expand_box(vehicle.box, width, height, crop_padding)
    x1, y1, x2, y2 = crop_box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    candidates = run_plate_onnx_session(plate_session, crop, plate_conf)
    return build_plate_results_from_candidates_onnx(candidates, image, vehicle, crop_box, crop, ocr_session, ocr_keys)


def draw_label(image, text: str, left: int, top: int, color):
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + baseline + 4, top)
    cv2.rectangle(image, (left, y - th - baseline - 6), (left + tw + 8, y + 2), color, -1)
    cv2.putText(image, text, (left + 4, y - baseline - 2), font, scale, (255, 255, 255), thickness)


def annotate(image, vehicles: Iterable[VehicleTrack], plates: Iterable[PlateResult]):
    output = image.copy()
    plate_by_track = {}
    for plate in plates:
        plate_by_track.setdefault(plate.vehicle.track_id, []).append(plate)

    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle.box
        color = (0, 170, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        plates_for_vehicle = plate_by_track.get(vehicle.track_id, [])
        plate_text = plates_for_vehicle[0].text if plates_for_vehicle else ""
        label = f"ID {vehicle.track_id}"
        if plate_text:
            label += f" | {plate_text}"
        draw_label(output, label, x1, max(0, y1 - 6), color)

    for plate in plates:
        pts = order_corners(plate.corners_image).astype(np.int32)
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        if plate.text:
            px, py = pts[0]
            draw_label(output, plate.text, int(px), max(0, int(py) - 6), (0, 140, 0))
    return output


def process_frame_ultralytics_with_onnx_ocr(
    image,
    vehicle_model,
    plate_model,
    ocr_session,
    ocr_keys,
    vehicle_conf: float,
    plate_conf: float,
    crop_padding: float,
):
    vehicle_results = vehicle_model.track(
        image,
        conf=vehicle_conf,
        classes=sorted(VEHICLE_CLASS_IDS),
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )
    vehicles = detect_vehicles(vehicle_results[0], image.shape) if vehicle_results else []

    plates: List[PlateResult] = []
    for vehicle in vehicles:
        plates.extend(
            detect_plates_for_vehicle_ultralytics_onnx_ocr(
                plate_model,
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


def save_warped_plates(output_dir: Path, image_stem: str, plates: Sequence[PlateResult]):
    plate_dir = output_dir / "plates"
    plate_dir.mkdir(parents=True, exist_ok=True)
    for idx, plate in enumerate(plates, start=1):
        text = plate.text or "unknown"
        filename = f"{image_stem}_track{plate.vehicle.track_id}_plate{idx}_{text}.jpg"
        cv2.imwrite(str(plate_dir / filename), plate.warped)
