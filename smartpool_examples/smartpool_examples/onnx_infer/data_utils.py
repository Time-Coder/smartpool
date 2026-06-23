import os
import glob
import shutil

from config import DATASET_FOLDER, MODEL_FOLDER, MODEL_PATH


def _count_images():
    return len(glob.glob(DATASET_FOLDER + "/*.jpg"))


def _flatten_images():
    for image_filename in list(glob.glob(DATASET_FOLDER + "/**/*.jpg", recursive=True)):
        image_folder = os.path.dirname(os.path.abspath(image_filename)).replace("\\", "/")
        image_basename = os.path.basename(image_filename)
        if image_folder != DATASET_FOLDER:
            shutil.move(image_filename, DATASET_FOLDER + "/" + image_basename)


def download_dataset():
    if not os.path.isdir(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)

    _flatten_images()
    if _count_images() > 4000:
        print(f"[SKIP DOWNLOAD] Dataset exists ({_count_images()} images)")
        return

    print("[DOWNLOAD] val2017.zip (~1GB) ...")
    from ultralytics.utils.downloads import safe_download
    safe_download(
        url="http://images.cocodataset.org/zips/val2017.zip",
        dir=DATASET_FOLDER,
        unzip=True,
        delete=True,
    )

    _flatten_images()
    n = _count_images()
    print(f"[DONE] {n} images ready")


def download_model():
    if os.path.isfile(MODEL_PATH):
        print(f"[SKIP] Model {os.path.basename(MODEL_PATH)} exists")
        return

    if not os.path.isdir(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    
    print("[DOWNLOAD] YOLOv8n via ultralytics...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    exported_path = model.export(format="onnx")
    
    exported_path = exported_path.replace("\\", "/")
    if os.path.normpath(exported_path) != os.path.normpath(MODEL_PATH):
        shutil.copy2(exported_path, MODEL_PATH)
    
    print(f"[DONE] Model ready at {MODEL_PATH}")
