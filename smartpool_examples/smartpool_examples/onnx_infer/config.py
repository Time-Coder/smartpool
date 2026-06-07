from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PACKAGE_DIR / "data" / "models"
DATASET_DIR = PACKAGE_DIR / "data" / "dataset"
OUTPUT_DIR = PACKAGE_DIR / "data" / "output"

MODEL_URL = "https://hf-mirror.com/Kalray/yolov8/resolve/main/yolov8n.onnx"
MODEL_PATH = MODEL_DIR / "yolov8n.onnx"

COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{}.jpg"
COCO_IMAGE_IDS = [
    "000000000139", "000000000285", "000000000632", "000000000724", "000000000776",
    "000000000785", "000000000802", "000000000872", "000000001000", "000000001268",
    "000000001296", "000000001353", "000000001490", "000000001503", "000000001532",
    "000000001584", "000000001675", "000000001818", "000000096001", "000000202001",
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
