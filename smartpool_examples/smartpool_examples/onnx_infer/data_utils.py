import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from config import COCO_ZIP_URLS, DATASET_DIR, MODEL_PATH, MODEL_URL
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def download_file(url: str, dest: Path, desc: str = "", max_retries: int = 3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
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
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(task, advance=len(chunk))
            tmp.rename(dest)
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


def download_dataset():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    existing = len(list(DATASET_DIR.glob("*.jpg")))
    if existing > 4000:
        print(f"[SKIP DOWNLOAD] Dataset exists ({existing} images)")
        return

    print("[DOWNLOAD] val2017.zip (~1GB) ...")
    zip_path = DATASET_DIR / "val2017.zip"

    for url in COCO_ZIP_URLS:
        try:
            download_file(url, zip_path, "COCO val2017")
            break
        except Exception:
            print(f"  mirror failed: {url}")
            continue
    else:
        raise RuntimeError("all mirrors failed")

    print(f"[EXTRACT] val2017.zip -> {DATASET_DIR} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATASET_DIR)

    zip_path.unlink()
    n = len(list(DATASET_DIR.glob("*.jpg")))
    print(f"[DONE] {n} images ready")


def download_model():
    if MODEL_PATH.exists():
        print(f"[SKIP DOWNLOAD] Model {MODEL_PATH.name} exists")
        return
    print("[DOWNLOAD] YOLOv8n ONNX...")
    download_file(MODEL_URL, MODEL_PATH, "YOLOv8n")
