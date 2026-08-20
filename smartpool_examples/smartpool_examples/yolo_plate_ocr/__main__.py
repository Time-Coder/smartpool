if __name__ == "__main__":
    import os
    import sys

    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)


from typing import Optional
from enum import Enum

from pathlib import Path
import typer

from .config import OUTPUT_DIR, ROOT


app = typer.Typer(
    help=(
        "Run a multi-stage YOLOv8 vehicle tracking -> plate keypoint detection "
        "-> perspective transform -> PaddleOCR ONNX recognition pipeline."
    )
)

class Method(Enum):
    SmartPool = "smartpool"
    Sequentially = "sequentially"
    Ultralytics = "ultralytics"

@app.command()
def main(
    method: Method = typer.Option(
        Method.SmartPool,
        "--method",
        help="Run with smartpool ONNX pipeline, ONNX Runtime loop, or Ultralytics loop.",
    ),
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        help="Image file, image folder, or video file. Defaults to the auto-downloaded sample dataset.",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="Processing mode: auto, image, or video.",
    ),
    output_dir: Path = typer.Option(OUTPUT_DIR, "--output-dir", help="Where annotated images are saved."),
    video_output: Optional[Path] = typer.Option(
        None,
        "--video-output",
        help="Output video path when processing a video source.",
    ),
    vehicle_conf: float = typer.Option(0.25, "--vehicle-conf", min=0.0, max=1.0),
    plate_conf: float = typer.Option(0.25, "--plate-conf", min=0.0, max=1.0),
    crop_padding: float = typer.Option(0.05, "--crop-padding", min=0.0, max=0.5),
    save_plates: bool = typer.Option(True, "--save-plates/--no-save-plates"),
):
    typer.echo("Use `python -m smartpool_examples.yolo_plate_ocr --help` to see all options.")
    typer.echo(f"See source code at folder {ROOT}")
    typer.echo()

    import time
    import cv2

    from .data_utils import is_video_file, iter_images, prepare_dataset, prepare_models
    from .progress_info import PipelineProgress

    (
        vehicle_model_path,
        plate_model_path,
        vehicle_onnx_path,
        plate_onnx_path,
        ocr_onnx_path,
        ocr_keys_path,
    ) = prepare_models()
    if source is None:
        source = prepare_dataset()
    else:
        source = source.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode not in {"auto", "image", "video"}:
        typer.echo("--mode must be one of: auto, image, video")
        raise typer.Exit(code=1)

    if mode == "auto":
        mode = "video" if is_video_file(source) else "image"

    if mode == "video" and not source.is_file():
        typer.echo("Video mode requires --source to be a video file.")
        raise typer.Exit(code=1)

    start = time.perf_counter()
    progress_total = None
    if mode == "image":
        image_paths = iter_images(source)
        if not image_paths:
            typer.echo(f"No images found in {source}")
            raise typer.Exit(code=1)
        progress_total = len(image_paths)
    elif mode == "video":
        capture = cv2.VideoCapture(str(source))
        progress_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        capture.release()

    progress = PipelineProgress(progress_total)
    if method == Method.SmartPool:
        from .infer_with_smartpool import infer_with_smartpool

        result = infer_with_smartpool(
            source,
            output_dir,
            vehicle_onnx_path,
            plate_onnx_path,
            ocr_onnx_path,
            ocr_keys_path,
            vehicle_conf,
            plate_conf,
            crop_padding,
            save_plates,
            progress,
            video_output,
        )
    elif method == Method.Sequentially:
        from .infer_sequentially import infer_sequentially

        result = infer_sequentially(
            source,
            output_dir,
            vehicle_onnx_path,
            plate_onnx_path,
            ocr_onnx_path,
            ocr_keys_path,
            vehicle_conf,
            plate_conf,
            crop_padding,
            save_plates,
            progress,
            video_output,
        )
    else:
        from ultralytics import YOLO

        from .infer_sequentially_with_ultralytics import infer_sequentially_with_ultralytics

        result = infer_sequentially_with_ultralytics(
            source,
            output_dir,
            YOLO(str(vehicle_model_path)),
            YOLO(str(plate_model_path)),
            ocr_onnx_path,
            ocr_keys_path,
            vehicle_conf,
            plate_conf,
            crop_padding,
            save_plates,
            progress,
            video_output,
        )

    if mode == "video":
        frames, total_vehicles, total_plates, resolved_video_output = result
        typer.echo()
        typer.echo(f"{method.value} video completed in {time.perf_counter() - start:.2f} seconds")
        typer.echo(f"Frames: {frames}, vehicles: {total_vehicles}, plates: {total_plates}")
        typer.echo(f"Output video: {resolved_video_output}")
        typer.echo(f"Plates: {output_dir / 'plates'}")
        return

    total_vehicles, total_plates = result

    elapsed = time.perf_counter() - start
    typer.echo()
    typer.echo(f"{method.value} pipeline completed in {elapsed:.2f} seconds")
    typer.echo(f"Images: {progress_total}, vehicles: {total_vehicles}, plates: {total_plates}")
    typer.echo(f"Output: {output_dir}")


if __name__ == "__main__":
    app()
