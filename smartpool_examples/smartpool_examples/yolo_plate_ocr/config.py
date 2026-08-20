import os
from pathlib import Path

self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROOT = Path(self_folder)
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
DATASET_DIR = DATA_DIR / "dataset"
OUTPUT_DIR = DATA_DIR / "output"

VEHICLE_MODEL_NAME = "yolov8n.pt"
VEHICLE_ONNX_MODEL_NAME = "yolov8n.onnx"
PLATE_MODEL_NAME = "license_plate_keypoint.pt"
PLATE_ONNX_MODEL_NAME = "license_plate_keypoint.onnx"
PLATE_MODEL_URL = (
    "https://huggingface.co/PrasannaBAImodel/license-plate-keypoint-detection/"
    "resolve/main/license_plate_keypoint.pt"
)
OCR_ONNX_MODEL_NAME = "ch_PP-OCRv4_rec_infer.onnx"
OCR_KEYS_NAME = "ppocr_keys_v1.txt"
OCR_ONNX_MODEL_URL = "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx"
OCR_KEYS_URL = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt"
OCR_KEYS_URLS = (
    OCR_KEYS_URL,
    "https://cdn.jsdelivr.net/gh/PaddlePaddle/PaddleOCR@release/2.7/ppocr/utils/ppocr_keys_v1.txt",
)

SAMPLE_IMAGES = {
    "car10.jpg": (
        "https://raw.githubusercontent.com/femioladeji/"
        "License-Plate-Recognition-Nigerian-vehicles/master/test_images/car10.jpg"
    ),
    "car6.jpg": (
        "https://raw.githubusercontent.com/femioladeji/"
        "License-Plate-Recognition-Nigerian-vehicles/master/test_images/car6.jpg"
    ),
    "test_image_1.png": (
        "https://raw.githubusercontent.com/georgia-tech-db/"
        "license-plate-recognition/main/test_image_1.png"
    ),
    "test_image_2.png": (
        "https://raw.githubusercontent.com/georgia-tech-db/"
        "license-plate-recognition/main/test_image_2.png"
    ),
}

VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
