# YOLO Plate OCR

This example demonstrates a multi-stage inference pipeline:

1. YOLOv8 vehicle detection with ByteTrack tracking to get vehicle boxes and track IDs.
2. Vehicle crops from the original image.
3. YOLOv8 Pose license plate detection on each vehicle crop, producing four plate corners.
4. Perspective transform to rectify each plate.
5. PaddleOCR ONNX text extraction on the rectified plate image.
6. Draw vehicle boxes, plate quadrangles, track IDs, and recognized plate text, then save annotated images.

Run it with:

```bash
python -m smartpool_examples.yolo_plate_ocr
```

The default method is `smartpool`, which exports the YOLO models to ONNX,
downloads a PaddleOCR recognition ONNX model, then uses `InferSessionPool.model()` for
vehicle detection, plate pose detection, and OCR recognition. You can compare
it with a plain ONNX Runtime loop and an Ultralytics YOLO + ONNX OCR baseline:

```bash
python -m smartpool_examples.yolo_plate_ocr --method smartpool
python -m smartpool_examples.yolo_plate_ocr --method sequentially
python -m smartpool_examples.yolo_plate_ocr --method ultralytics
```

The example downloads sample images, YOLO PyTorch models, a PaddleOCR ONNX
recognition model, and the PaddleOCR character dictionary into `data/` on first
run. It exports YOLO ONNX models for the SmartPool and ONNX Runtime paths, and
skips files that already exist. Annotated images are written to `data/output/`;
rectified plate crops are written to `data/output/plates/`.

Useful options:

```bash
python -m smartpool_examples.yolo_plate_ocr --source path/to/images
python -m smartpool_examples.yolo_plate_ocr --method smartpool --mode video --source path/to/video.mp4 --video-output path/to/out.mp4
python -m smartpool_examples.yolo_plate_ocr --vehicle-conf 0.35 --plate-conf 0.4
```
