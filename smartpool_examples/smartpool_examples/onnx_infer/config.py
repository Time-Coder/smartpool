import os


self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
DATA_FOLDER = self_folder + "/data"
MODEL_FOLDER = DATA_FOLDER + "/models"
DATASET_FOLDER = DATA_FOLDER + "/dataset"
OUTPUT_FOLDER = DATA_FOLDER + "/output"

MODEL_URL = "https://hf-mirror.com/Kalray/yolov8/resolve/main/yolov8n.onnx"
MODEL_PATH = MODEL_FOLDER + "/yolov8n.onnx"

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
