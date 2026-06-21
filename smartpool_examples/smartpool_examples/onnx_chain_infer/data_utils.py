import glob
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile

from config import (
    COCO_ZIP_URLS,
    DATASET_FOLDER,
    DET_MODEL_PATH,
    DET_MODEL_URL,
    POSE_MODEL_PATH,
    POSE_MODEL_URL,
)
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def download_file(url: str, dest_zip_filename: str, desc: str = "", max_retries: int = 3):
    dest_folder: str = os.path.dirname(dest_zip_filename)
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    tmp_filename = dest_zip_filename + ".tmp"
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=120)
            total = int(resp.headers.get("Content-Length", 0))
            chunk_size = 8192
            with Progress(
                TextColumn(f"[cyan]{desc}" if desc else ""),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                transient=False,
            ) as pbar:
                task = pbar.add_task("", total=total)
                with open(tmp_filename, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(task, advance=len(chunk))

            os.rename(tmp_filename, dest_zip_filename)
            return
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last_err = e
            if attempt < max_retries:
                delay = 2 ** attempt
                time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < max_retries:
                delay = 2 ** attempt
                time.sleep(delay)

    raise last_err


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
    zip_path = DATASET_FOLDER + "/val2017.zip"

    for url in COCO_ZIP_URLS:
        try:
            download_file(url, zip_path, "COCO val2017")
            break
        except Exception:
            print(f"  mirror failed: {url}")
            continue
    else:
        raise RuntimeError("all mirrors failed")

    print(f"[EXTRACT] val2017.zip -> {DATASET_FOLDER} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATASET_FOLDER)

    _flatten_images()
    n = _count_images()
    print(f"[DONE] {n} images ready")


def download_model():
    if os.path.isfile(DET_MODEL_PATH):
        print(f"[SKIP DOWNLOAD] Detection model {os.path.basename(DET_MODEL_PATH)} exists")
    else:
        print("[DOWNLOAD] YOLOv8n ONNX (detection model)...")
        download_file(DET_MODEL_URL, DET_MODEL_PATH, "YOLOv8n")

    if os.path.isfile(POSE_MODEL_PATH):
        print(f"[SKIP DOWNLOAD] Pose model {os.path.basename(POSE_MODEL_PATH)} exists")
    else:
        print("[DOWNLOAD] YOLOv8n-pose ONNX (pose estimation model)...")
        download_file(POSE_MODEL_URL, POSE_MODEL_PATH, "YOLOv8n-pose")
