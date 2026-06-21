import os

self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
DATA_FOLDER = self_folder + "/data"
MODEL_FOLDER = DATA_FOLDER + "/models"
DATASET_FOLDER = DATA_FOLDER + "/dataset"
OUTPUT_FOLDER = DATA_FOLDER + "/output"

DET_MODEL_URL = "https://hf-mirror.com/Kalray/yolov8/resolve/main/yolov8n.onnx"
DET_MODEL_PATH = MODEL_FOLDER + "/yolov8n.onnx"

POSE_MODEL_URL = "https://hf-mirror.com/Xenova/yolov8-pose-onnx/resolve/main/yolov8n-pose.onnx"
POSE_MODEL_PATH = MODEL_FOLDER + "/yolov8n-pose.onnx"

COCO_ZIP_URLS = [
    "http://images.cocodataset.org/zips/val2017.zip",
]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

DET_INPUT_SIZE = (640, 640)
POSE_INPUT_SIZE = (640, 640)
