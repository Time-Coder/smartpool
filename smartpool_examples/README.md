# SmartPool Examples

This package contains practical examples demonstrating the capabilities of SmartPool for various computational tasks.

## Examples Overview

### 1. Prime Number Counting (`count_prime`)

Count the number of prime numbers below 10000 using smartpool.ProcessPool.
Demonstrates basic usage of smartpool.ProcessPool.

#### Running the Example

```bash
python -m smartpool_examples.count_prime
```

### 2. Cross-Validation for Deep Learning models (`cross_validation`)

Demonstrates SmartPool's capabilities for machine learning workloads with GPU resource management. Runs 5-fold cross-validation of four lightweight CNN models (AudioCNN-Small, AudioCNN-Medium, AudioCNN-Large, AudioCNN-Res) for ESC-50 environmental sound classification. The preprocessing stage submits one task per audio file (2,000 tasks total) to decode the WAV and extract a 64×64 log-mel spectrogram saved to disk; the training stage then runs one task per fold per model (20 tasks) reading the preprocessed features from disk.

#### Running the Example

```bash
# Using ProcessPool  
python -m smartpool_examples.cross_validation --pool smartpool.ProcessPool

# Using ThreadPool
python -m smartpool_examples.cross_validation --pool smartpool.ThreadPool

# Using multiprocessing.Pool
python -m smartpool_examples.cross_validation --pool multiprocessing.Pool

# Using concurrent.futures.ProcessPoolExecutor
python -m smartpool_examples.cross_validation --pool concurrent.futures.ProcessPoolExecutor

# Using concurrent.futures.ThreadPoolExecutor
python -m smartpool_examples.cross_validation --pool concurrent.futures.ThreadPoolExecutor

# Using joblib.Parallel(backend='loky')
python -m smartpool_examples.cross_validation --pool joblib.Parallel(backend='loky')

# Using joblib.Parallel(backend='threading')
python -m smartpool_examples.cross_validation --pool joblib.Parallel(backend='threading')

# Using Ray
python -m smartpool_examples.cross_validation --pool ray
```

#### What it Demonstrates

- GPU memory management and core allocation
- Automatic device selection (CPU vs GPU)
- Per-sample audio preprocessing pipeline (decoding + log-mel feature extraction)
- Cross-validation pipeline parallelization
- Resource monitoring during training
- Performance comparison with external frameworks


### 3. ONNX Inference (`onnx_infer`)

Runs batched ONNX model inference using `InferSessionPool` for concurrent GPU/CPU execution.
Automatically manages inference sessions across worker threads.

#### Running the Example

```bash
python -m smartpool_examples.onnx_infer --max-workers 4
```

#### What it Demonstrates

- `InferSessionPool` creation and session lifecycle management
- Multi-threaded inference with automatic device placement
- COCO-format image preprocessing (resize, normalize, letterbox)
- Softmax + top-5 postprocessing
- Progress bars for downloads and inference steps


### 4. YOLO License Plate OCR (`yolo_plate_ocr`)

Runs a multi-stage vehicle and license plate recognition pipeline:
YOLOv8 vehicle detection + ByteTrack tracking, vehicle cropping, YOLOv8 Pose
plate corner detection, perspective rectification, PaddleOCR ONNX recognition, and final image
annotation.

#### Running the Example

```bash
python -m smartpool_examples.yolo_plate_ocr --method smartpool
python -m smartpool_examples.yolo_plate_ocr --method sequentially
python -m smartpool_examples.yolo_plate_ocr --method ultralytics
```

#### What it Demonstrates

- Multi-stage inference pipeline orchestration
- SmartPool `InferSessionPool.model` ONNX inference for YOLO and PaddleOCR recognition
- ONNX Runtime sequential and Ultralytics YOLO + ONNX OCR sequential baselines
- Vehicle Track ID propagation across downstream plate detection and OCR
- Four-corner plate keypoint handling and perspective transform
- Automatic sample/model download with cache checks
- Rich progress bars for tracking, OCR, and result saving


## License

MIT License - see main smartpool repository for details
