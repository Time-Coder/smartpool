from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from smartpool_examples.onnx_chain_infer.config import (
    SKELETON,
)


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess_det(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"failed to read image: {image_path}")

    blob, scale, pad = letterbox(img)
    blob = blob.astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])
    return img, blob, scale, pad


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
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess_det(
    raw_output: np.ndarray, scale: float, pad: Tuple[int, int], img_shape: Tuple[int, int],
    conf_thresh: float = 0.5, iou_thresh: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    predictions = np.squeeze(raw_output).T
    box_data = predictions[:, :4]
    class_scores = predictions[:, 4:]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    person_mask = class_ids == 0
    mask = (scores > conf_thresh) & person_mask
    if not mask.any():
        return []

    boxes = box_data[mask]
    scores = scores[mask]

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(boxes_xyxy, scores, iou_thresh)
    if not keep:
        return []

    boxes_xyxy = boxes_xyxy[keep]
    left, top = pad
    boxes_xyxy = (boxes_xyxy - np.array([left, top, left, top])) / scale
    orig_h, orig_w = img_shape[:2]
    boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_w, orig_h, orig_w, orig_h])

    person_boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes_xyxy]
    return person_boxes


def preprocess_pose(src_img: np.ndarray, person_box: Tuple[int, int, int, int], input_size=(640, 640)):
    x1, y1, x2, y2 = person_box
    h, w = src_img.shape[:2]
    margin = 0.2
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))

    crop = src_img[y1:y2, x1:x2]
    crop_blob, scale, pad = letterbox(crop, input_size)
    crop_blob = crop_blob.astype(np.float32) / 255.0
    crop_blob = np.ascontiguousarray(crop_blob.transpose(2, 0, 1)[np.newaxis, ...])

    crop_info = {
        "crop_x1": x1, "crop_y1": y1,
        "scale": scale, "pad": pad,
    }
    return crop_blob, crop_info


def postprocess_pose(
    raw_output: np.ndarray, crop_info: dict,
    conf_thresh: float = 0.5, iou_thresh: float = 0.5
):
    predictions = np.squeeze(raw_output).T
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    kpts_flat = predictions[:, 5:]

    mask = scores > conf_thresh
    if not mask.any():
        return None

    boxes = boxes[mask]
    scores = scores[mask]
    kpts_flat = kpts_flat[mask]

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(boxes_xyxy, scores, iou_thresh)
    if not keep:
        return None

    best_idx = keep[scores[keep].argmax()]
    kpts = kpts_flat[best_idx].reshape(-1, 3)

    left, top = crop_info["pad"]
    scale = crop_info["scale"]
    kpts[:, 0] = (kpts[:, 0] - left) / scale
    kpts[:, 1] = (kpts[:, 1] - top) / scale
    kpts[:, 0] += crop_info["crop_x1"]
    kpts[:, 1] += crop_info["crop_y1"]

    return kpts


def draw_results(
    src_img: np.ndarray, output_path: str,
    person_boxes: List[Tuple[int, int, int, int]],
    all_keypoints: List[np.ndarray],
):
    img = src_img.copy()
    box_color = (56, 56, 255)
    kpt_color = (0, 255, 0)
    skeleton_color = (0, 255, 255)

    for box, kpts in zip(person_boxes, all_keypoints):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        label = "person"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), box_color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if kpts is not None:
            for (x, y, conf) in kpts[:, :3]:
                if conf > 0.5:
                    cv2.circle(img, (int(x), int(y)), 3, kpt_color, -1)

            for i, j in SKELETON:
                if kpts[i, 2] > 0.5 and kpts[j, 2] > 0.5:
                    pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                    pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                    cv2.line(img, pt1, pt2, skeleton_color, 2)

    cv2.imwrite(output_path, img)
