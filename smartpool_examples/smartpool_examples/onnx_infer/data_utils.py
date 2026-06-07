import time
import urllib.error
import urllib.request
from pathlib import Path

from config import (
    COCO_IMAGE_IDS,
    COCO_IMAGE_URL,
    DATASET_DIR,
    MODEL_PATH,
    MODEL_URL,
)
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
            resp = urllib.request.urlopen(req, timeout=60)
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
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < max_retries:
                delay = 2 ** attempt
                time.sleep(delay)
    raise last_err


def download_model():
    if MODEL_PATH.exists():
        print(f"[SKIP DOWNLOAD] Model {MODEL_PATH.name} exists")
        return
    print("[DOWNLOAD] YOLOv8n ONNX...")
    download_file(MODEL_URL, MODEL_PATH, "YOLOv8n")


def download_dataset():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in DATASET_DIR.glob("*.jpg")}
    to_dl = [iid for iid in COCO_IMAGE_IDS if iid not in existing]
    if not to_dl:
        print(f"[SKIP DOWNLOAD] Dataset exists ({len(COCO_IMAGE_IDS)} images)")
        return
    print(f"[DOWNLOAD] {len(to_dl)} COCO images...")
    for iid in to_dl:
        url = COCO_IMAGE_URL.format(iid)
        dest = DATASET_DIR / f"{iid}.jpg"
        download_file(url, dest, f"COCO {iid}")
