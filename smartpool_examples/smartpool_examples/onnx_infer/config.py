from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PACKAGE_DIR / "data" / "models"
DATASET_DIR = PACKAGE_DIR / "data" / "dataset"
OUTPUT_DIR = PACKAGE_DIR / "data" / "output"

MODEL_URL = "https://hf-mirror.com/Kalray/yolov8/resolve/main/yolov8n.onnx"
MODEL_PATH = MODEL_DIR / "yolov8n.onnx"

# Mirror URLs for val2017.zip (tried in order, first successful wins)
# 国内已知直链镜像较少，TUNA/USTC 已停止服务，pjreddie 只有 2014 版。
# 如你有内网或自建镜像，在前面加一条即可优先使用，例如：
#   "https://your-mirror.example.com/val2017.zip",
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
