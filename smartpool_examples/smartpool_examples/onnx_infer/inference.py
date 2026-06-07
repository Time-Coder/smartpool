import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from config import COCO_CLASSES

from smartpool import InferSessionPool, Resource


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


def preprocess(img: np.ndarray):
    img, scale, pad = letterbox(img)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return img, scale, pad


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


def postprocess(output: np.ndarray, orig_shape: Tuple[int, int], scale: float,
                pad: Tuple[int, int], conf_thresh: float = 0.5, iou_thresh: float = 0.5):
    predictions = np.squeeze(output).T
    box_data = predictions[:, :4]
    class_scores = predictions[:, 4:]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    mask = scores > conf_thresh
    if not mask.any():
        return []
    boxes = box_data[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(boxes_xyxy, scores, iou_thresh)
    if not keep:
        return []

    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    left, top = pad
    boxes_xyxy = (boxes_xyxy - np.array([left, top, left, top])) / scale
    boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_shape[1], orig_shape[0], orig_shape[1], orig_shape[0]])

    return [
        {"bbox": boxes_xyxy[i].tolist(), "score": float(scores[i]), "class_id": int(class_ids[i])}
        for i in range(len(boxes_xyxy))
    ]


def draw_detections(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    colors = [
        (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
        (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
        (52, 147, 26), (187, 135, 1), (132, 145, 137), (168, 229, 11),
    ]
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_id = det["class_id"]
        score = det["score"]
        color = colors[cls_id % len(colors)]
        label = f"{COCO_CLASSES[cls_id]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def infer_task(image_path: str, model_path_str: str, output_dir_str: str, infer_session_pool: InferSessionPool) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    blob, scale, pad = preprocess(img)
    gpu_res = Resource(cpu_cores_in_python=1, cpu_mem=512*1024**2,
                       gpu_cores=1000, gpu_mem=2*1024**3)
    cpu_res = Resource(cpu_cores_in_python=2, cpu_mem=2*1024**3)
    future = infer_session_pool.submit(model_path_str, args=(blob,), cpu_mode_res=cpu_res, gpu_mode_res=gpu_res)
    outputs = future.result()[0]
    detections = postprocess(outputs, (h, w), scale, pad)
    result_img = draw_detections(img, detections)

    out_path = os.path.join(output_dir_str, os.path.basename(image_path))
    cv2.imwrite(out_path, result_img)
