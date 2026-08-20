import ssl
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import typer
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from ultralytics import YOLO

from .config import (
    DATASET_DIR,
    MODEL_DIR,
    OCR_KEYS_NAME,
    OCR_KEYS_URLS,
    OCR_ONNX_MODEL_NAME,
    OCR_ONNX_MODEL_URL,
    PLATE_MODEL_NAME,
    PLATE_MODEL_URL,
    PLATE_ONNX_MODEL_NAME,
    SAMPLE_IMAGES,
    VEHICLE_MODEL_NAME,
    VEHICLE_ONNX_MODEL_NAME,
    VIDEO_EXTENSIONS,
)


def _open_url(url: str):
    request = Request(url, headers={"User-Agent": "smartpool-examples/0.1"})
    try:
        return urlopen(request)
    except Exception:
        return urlopen(request, context=ssl._create_unverified_context())


def download_file(urls, path: Path, description: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        typer.echo(f"[SKIP] {description} exists: {path}")
        return

    if isinstance(urls, str):
        urls = (urls,)

    tmp_path = path.with_suffix(path.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()

    last_error = None
    for index, url in enumerate(urls, start=1):
        if len(urls) > 1:
            typer.echo(f"[DOWNLOAD] {description} mirror {index}/{len(urls)}")
        try:
            response = _open_url(url)
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            )
            with response:
                total = int(response.headers.get("Content-Length") or 0)
                with progress:
                    task = progress.add_task(f"download {description}", total=total or None)
                    with tmp_path.open("wb") as fp:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            fp.write(chunk)
                            progress.update(task, advance=len(chunk))

            tmp_path.replace(path)
            return
        except Exception as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            typer.echo(f"[WARN] Failed to download {description} from {url}: {exc}")

    raise RuntimeError(f"failed to download {description}") from last_error


def prepare_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vehicle_model = MODEL_DIR / VEHICLE_MODEL_NAME
    vehicle_onnx_model = MODEL_DIR / VEHICLE_ONNX_MODEL_NAME
    if vehicle_model.is_file():
        typer.echo(f"[SKIP] Vehicle model exists: {vehicle_model}")
    else:
        typer.echo("[DOWNLOAD] YOLOv8n via ultralytics...")
        YOLO(str(vehicle_model))
        typer.echo(f"[DONE] Vehicle model ready: {vehicle_model}")

    plate_model = MODEL_DIR / PLATE_MODEL_NAME
    plate_onnx_model = MODEL_DIR / PLATE_ONNX_MODEL_NAME
    if plate_model.is_file():
        typer.echo(f"[SKIP] Plate model exists: {plate_model}")
    else:
        typer.echo("[DOWNLOAD] license plate keypoint model via ultralytics...")
        from ultralytics.utils.downloads import safe_download

        safe_download(
            url=PLATE_MODEL_URL,
            file=plate_model,
            unzip=False,
            delete=False,
            exist_ok=True,
            progress=True,
        )
        typer.echo(f"[DONE] Plate model ready: {plate_model}")

    export_onnx(vehicle_model, vehicle_onnx_model, "vehicle detector")
    export_onnx(plate_model, plate_onnx_model, "license plate keypoint detector")

    ocr_onnx_model, ocr_keys = prepare_ocr_model()
    return vehicle_model, plate_model, vehicle_onnx_model, plate_onnx_model, ocr_onnx_model, ocr_keys


def export_onnx(pt_path: Path, onnx_path: Path, description: str):
    if onnx_path.is_file() and onnx_path.stat().st_size > 0:
        typer.echo(f"[SKIP] {description} ONNX exists: {onnx_path}")
        return

    typer.echo(f"[EXPORT] {description} to ONNX...")
    exported_path = Path(YOLO(str(pt_path)).export(format="onnx", dynamic=True))
    if exported_path.resolve() != onnx_path.resolve():
        shutil.move(str(exported_path), str(onnx_path))
    typer.echo(f"[DONE] {description} ONNX ready: {onnx_path}")


def prepare_ocr_model():
    onnx_path = MODEL_DIR / OCR_ONNX_MODEL_NAME
    keys_path = MODEL_DIR / OCR_KEYS_NAME

    download_file(OCR_ONNX_MODEL_URL, onnx_path, "PaddleOCR rec ONNX model")
    download_file(OCR_KEYS_URLS, keys_path, "PaddleOCR character dictionary")
    return onnx_path, keys_path


def prepare_dataset():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLE_IMAGES.items():
        download_file(url, DATASET_DIR / name, name)
    return DATASET_DIR


def iter_images(source: Path):
    if source.is_file():
        return [source]

    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(source.glob(pattern))
    return sorted(image_paths)


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
