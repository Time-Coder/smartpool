# SmartPool

SmartPool is a Python library that provides intelligent resource-aware pooling mechanisms for parallel computing. It automatically manages CPU and GPU resources to optimize performance while preventing resource exhaustion.

## Features

- **Multiple Pool Types**: ProcessPool, ThreadPool, and InterpreterPool for different use cases
- **Intuitive API Design**: Almost the same usage as `concurrent.futures` pools
- **Automatic Resource Management**: Monitors and manages CPU cores, memory, and GPU resources
- **Hardware-Aware Scheduling**: Automatically detects system resources and schedules tasks accordingly
- **PyTorch Integration**: Support for PyTorch multiprocessing with tensor sharing to avoid serialization
- **Training Hot Migration**: Automatically moves CPU training tasks to GPU when `best_device()` changes
- **InferSessionPool**: Thread-safe ONNX Runtime session pool for concurrent inference
- **ONNX Inference Batching**: `InferSessionPool` transparently batches inference calls across the batch dimension
- **InferSessionPool IO Binding**: Zero-copy ONNX inference with `IOBinding` support for GPU-resident data
- **`@Pool.task` Decorator**: Write concurrent code like synchronous code using the task decorator pattern
- **Future Dependency Resolution**: Submit tasks with `Future` objects as arguments; automatically waits for dependencies to complete before execution
- **Enhanced Future API**: Tuple unpacking (`future.unpack(n)`) and index/key access (`future[key]`) for easy pipeline composition
- **Chunked Task Batching**: Aggregate multiple `submit()` calls with the same function into a single batch execution for better throughput

## Installation

```bash
pip install pysmartpool
```

## Examples

### Basic Usage

```python
from smartpool import ProcessPool


if __name__ == "__main__":
    # Create a process pool that automatically manages system resources
    with ProcessPool() as pool:
        # Submit tasks with proper argument passing
        futures = [pool.submit(expensive_computation, args=(arg,)) for arg in arguments]
        
        # Get results
        results = [future.result() for future in futures]
```

### Resource-Aware Task Scheduling

```python
from smartpool import ProcessPool, DataSize, Resource


# Tasks can specify their resource requirements
def memory_intensive_task(data):
    # Your computation here
    return processed_data


if __name__ == "__main__":
    with ProcessPool(use_torch=True) as pool:
        # Pool automatically schedules tasks based on available memory
        future = pool.submit(
            memory_intensive_task, 
            args=(large_dataset,),
            cpu_mode_res=Resource(
                cpu_cores_in_python=2,
                cpu_mem=1*DataSize.GB,
            ),
            gpu_mode_res=Resource(
                cpu_cores_in_python=1,
                cpu_mem=500*DataSize.MB,
                gpu_cores=1024,
                gpu_mem=1*DataSize.GB,
            ),
        )
```

### PyTorch Training Hot Migration from CPU to GPU

SmartPool automatically migrates training tasks from CPU to GPU when better devices become available:

```python
# Complete setup for training with optimizer migration
from smartpool import (
    limit_num_single_thread, 
    best_device, 
    move_optimizer_to,
    ProcessPool
)
# Critical: Call before importing torch/numpy
limit_num_single_thread()

import torch

def training_task():
    device = best_device().torch_device # <-- get best suitable device at init time
    old_device = device

    for epoch in range(epochs):
        for x, y in data_loader:
            device = best_device().torch_device # <-- get best suitable device at each batch
            x, y = x.to(device), y.to(device)

            if old_device != device:
                model.to(device) # move model to new device
                move_optimizer_to(optimizer, device) # move optimizer to new device
                old_device = device
            
            do_other_things()


if __name__ == "__main__":
    with ProcessPool(use_torch=True) as pool:
        future = pool.submit(
            training_task,
            args=(model, optimizer, data),
            device_changeable=True # <-- turn on device hot migration
        )
```

### InferSessionPool — ONNX Runtime Concurrent Inference

`InferSessionPool` manages a pool of ONNX Runtime sessions for concurrent inference with auto device placement and optional IO binding.

```python
from smartpool import InferSessionPool, Device


def run_inference():
    infer_pool = InferSessionPool(max_workers=4)

    model_path = "yolov8n.onnx"
    images = [preprocess(img) for img in image_paths]

    futures = []
    for img in images:
        future = infer_pool.submit(
            model_path,
            args=(img,)
        )
        futures.append(future)

    results = [f.result() for f in futures]


# Use IO binding for zero-copy GPU inference
def run_inference_io_binding():
    import numpy as np
    import onnxruntime as ort

    infer_pool = InferSessionPool(max_workers=2)

    # Create OrtValue on GPU
    ort_value = ort.OrtValue.ort_value_from_numpy(
        np.random.randn(1, 3, 640, 640).astype(np.float32)
    )

    future = infer_pool.submit(
        "model.onnx",
        args=(ort_value,),
        use_io_binding=True,
        copy_outputs_to_cpu=False,  # keep results on GPU as OrtValue
    )
    result = future.result()  # returns OrtValue on device
```

### Task Decorator (`@pool.task`)

The `@pool.task` decorator lets you write concurrent code in the same style as synchronous code. Decorate any function, call it normally, and it returns a `Future` automatically.

```python
from smartpool import ThreadPool, InferSessionPool, Resource, DataSize


# --- Decorate preprocessing / postprocessing functions ---
# First, configure the pool(s) the decorator will use
ThreadPool.config(pool_name="pre_post", max_workers=4)

@ThreadPool.task(pool_name="pre_post", cpu_mode_res=Resource(cpu_cores=1, cpu_mem=15*DataSize.MB))
def preprocess(image_path: str):
    # ... read & transform image ...
    return img, blob, scale, pad

@ThreadPool.task(pool_name="pre_post", cpu_mode_res=Resource(cpu_cores=1, cpu_mem=81*DataSize.MB))
def postprocess(img, outputs, output_path, scale, pad):
    # ... decode results & save ...
    pass


# --- Pipeline ---
infer_session_pool = InferSessionPool()

def run_pipeline(image_paths: List[str]):
    futures = []
    for image_path in image_paths:
        preprocess_future = preprocess(image_path)              # returns Future, runs on ThreadPool
        img, blob, scale, pad = preprocess_future.unpack(4)     # unpack Future into sub-Futures

        infer_future = infer_session_pool.submit(model_path, args=(blob,))
        outputs = infer_future[0]                               # future[0] → sub-Future for first output

        postprocess_future = postprocess(img, outputs, output_path, scale, pad)
        futures.append(postprocess_future)

    for f in as_completed(futures):
        f.result()
```

The decorator can also be used **without arguments**:

```python
@ThreadPool.task
def my_task(x):
    return x * 2

future = my_task(42)
```

Use `Pool.config(pool_name, *args, **kwargs)` to pre-configure a named pool instance, and `Pool.instance(pool_name)` to retrieve it. The decorator automatically calls `Pool.instance(pool_name).submit(...)` at call time.

See full examples in `smartpool_examples/onnx_infer/` (YOLOv8 ONNX inference pipeline) and `smartpool_examples/optimize_cec2022/` (parallel optimization benchmark).

## API

### ProcessPool

Each worker run as a separate process with seperated GIL. Suitable for CPU-intensive tasks. 

```python
class ProcessPool:

    def __init__(
        self, max_workers:int=0,
        process_name_prefix:str="ProcessPool.worker:",
        mp_context:str="spawn",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
        use_torch:bool=False
    ): ...
        """
        Initializes a new ProcessPool instance.
        
        Args:
            max_workers: The maximum number of processes that can be used to
                execute the given calls. If None or not given then as many
                worker processes will be created as the machine has processors.
            mp_context: Select process start method from ['fork', 'spawn', 'forkserver']
            initializer: A callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
            initkwargs: A dictionary of keyword arguments to pass to the initializer.
            max_tasks_per_child: The maximum number of tasks a worker process
                can complete before it will exit and be replaced with a fresh
                worker process. The default of None means worker process will
                live as long as the executor. Requires a non-'fork' mp_context
                start method. When given, we default to using 'spawn' if no
                mp_context is supplied.
            use_torch: Whether to use PyTorch multiprocessing with tensor sharing and GPU device support.
        """

    def submit(
        self, func:Callable[..., Any],
        args:Optional[Tuple[Any]]=None,
        kwargs:Optional[Dict[str, Any]]=None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        use_torch: Optional[bool] = None,
        chunksize: Optional[int] = None,
        chunk_timeout: Optional[float] = 0.1,
    )->concurrent.futures.Future: ...
        """
        Submits a callable to be executed with the given arguments.
    
        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Args:
            func: The callable to execute.
            args: The arguments to pass to the callable. May contain `Future` objects — they will be automatically resolved before the task runs.
            kwargs: The keyword arguments to pass to the callable. May contain `Future` objects.
            cpu_mode_res: Resource requirements when running on CPU.
            gpu_mode_res: Resource requirements when running on GPU.
            use_torch: Whether to enable PyTorch tensor sharing and device migration.
            chunksize: If set, tasks with the same function are aggregated into
                      a batch and executed as a single call with concatenated args.
            chunk_timeout: Max seconds to wait before flushing a pending batch.
        
        Returns:
            A concurrent.futures.Future representing the given call.
        """

    @classmethod
    def config(cls, pool_name: str, *args, **kwargs) -> None: ...
        """
        Pre-configure a named pool instance for later retrieval via `cls.instance(pool_name)`.
        Used together with the `@pool.task` decorator.

        Example:
            ThreadPool.config(pool_name="pre_post", max_workers=4)
        """

    @classmethod
    def instance(cls, pool_name: str) -> Pool: ...
        """
        Get or create a pool instance by name. If the named pool was previously
        configured with `config()`, it is created with those arguments; otherwise
        default arguments are used.
        """

    @classmethod
    def task(cls, func: Callable = None, *, pool_name: str = "default", **submit_kwargs) -> Callable: ...
        """
        Decorator that turns a function into a task submitted to a named pool.
        Supports two forms:

        @ThreadPool.task                                 # no arguments (pool_name="default")
        @ThreadPool.task(pool_name="x", cpu_mode_res=...) # with arguments

        The decorated function returns a Future when called.
        """

    def map(
        self, func:Callable[..., Any],
        iterable:Iterable[Any],
        cpu_mode_res:Union[Resource, Iterable[Resource]] = Resource(cpu_cores_in_python=1),
        gpu_mode_res:Union[Resource, Iterable[Resource], None]=None,
        timeout:Optional[Union[float, int]]=None,
        chunksize:int=1
    )->Iterable[Any]: ...
        """
        Returns an iterator equivalent to map(func, iterable).
 
        Args:
            func: A callable that will take as many arguments as there are
                passed iterables.
            iterable: An iterable whose items will be passed to func as arguments.
            cpu_mode_res: Resource requirements (or iterable of resources) for CPU mode.
            gpu_mode_res: Resource requirements (or iterable of resources) for GPU mode.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a time.
        
        Returns:
            An iterator equivalent to: map(func, iterables).
        
        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """

    def starmap(
        self, func:Callable[..., Any],
        args_iterables:Iterable[Tuple[Any, ...]],
        cpu_mode_res:Union[Resource, Iterable[Resource]] = Resource(cpu_cores_in_python=1),
        gpu_mode_res:Union[Resource, Iterable[Resource], None]=None,
        timeout:Optional[Union[float, int]]=None,
        chunksize:int=1
    )->Iterable[Any]: ...
        """
        Like `map()` method but the elements of the `iterable` are expected to
        be iterables as well and will be unpacked as arguments. Hence
        `func` and (a, b) becomes func(a, b).
        """

    def flush(self) -> None: ...
        """
        Force-flush any pending chunked tasks immediately.
        Tasks submitted with `chunksize` that have not yet filled a full batch
        are submitted immediately.
        """

    @property
    def chunk_timeout(self) -> float: ...

    @chunk_timeout.setter
    def chunk_timeout(self, value: float) -> None: ...
        """
        Get/set the flush timeout for chunked task batching (default 0.1s).
        When a batch hasn't filled within this timeout, it is flushed automatically.
        """

    def shutdown(self, wait:bool=True, *, cancel_futures:bool=False)->None: ...
        """
        Clean-up the resources associated with the Executor.
  
        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.
        
        Args:
            wait: If True then shutdown will not return until all running
                futures have finished executing and the resources used by the
                executor have been reclaimed.
            cancel_futures: If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """

    def __enter__(self)->ProcessPool: ...

    def __exit__(self, exc_type, exc_val, exc_tb)->None: ...
```

### ThreadPool

Each worker run as a thread. Suitable for IO-intensive tasks.

Note: Unlike `ProcessPool`, `ThreadPool` does **not** support chunked task batching (`chunksize` is not accepted in `submit()`).

```python
class ThreadPool:

    def __init__(
        self, max_workers:int=0,
        thread_name_prefix:str="ThreadPool.worker:",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
        use_torch:bool=False
    ): ...
        """
        Initializes a new ThreadPool instance.

        Args: Same as ProcessPool
        """

    def submit(
        self, func:Callable[..., Any],
        args:Optional[Tuple[Any, ...]]=None,
        kwargs:Optional[Dict[str, Any]]=None,
        cpu_mode_res:Optional[Resource]=None,
        gpu_mode_res:Optional[Resource]=None,
        use_torch:Optional[bool]=None,
        device_changeable:bool=False
    )->Future: ...
        """
        Submits a callable to be executed with the given arguments.

        Args: Same as ProcessPool (without chunksize/chunk_timeout).
        """

    def map(self, ...): ...
    def starmap(self, ...): ...
    def shutdown(self, ...): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
        """
        All same as ProcessPool (without chunksize support).
        """
```

### InterpreterPool (Python 3.14+)

Each worker run as a thread within a isolated interpreter with seperated GIL. Suitable for CPU-intensive tasks.  
Less overhead than ProcessPool when create/destroy workers and task switching.   
But not support for numpy/torch.

```python
class InterpreterPool:

    def __init__(
        self, max_workers:int=0,
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
        use_torch:bool=False
    ): ...
        """
        Initializes a new InterpreterPool instance.

        Same as ProcessPool
        """

    def submit(self, ...): ...
    def map(self, ...): ...
    def starmap(self, ...): ...
    def shutdown(self, ...): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
        """
        All same as ProcessPool
        """
```

### InferSessionPool

Each worker runs as a thread with a dedicated ONNX Runtime `InferenceSession`.
Supports CPU, CUDA, DML, TensorRT, and other ONNX Runtime providers.
Transparently batches inference calls across the batch dimension for high throughput.

```python
class InferSessionPool(ThreadPool):

    def __init__(
        self, max_workers: int = 0,
        thread_name_prefix: str = "InferSessionPool.worker:",
        initializer: Optional[Callable[..., Any]] = None,
        initargs: Tuple[Any, ...] = (),
        initkwargs: Optional[Dict[str, Any]] = None,
        *,
        max_tasks_per_child: Optional[int] = None,
        print_info: bool = False,
    ): ...
        """
        Initializes a new InferSessionPool instance.

        Same as ThreadPool
        """

    def submit(
        self, model_path: str,
        args: Optional[Tuple[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        output_names: Optional[Union[List[str], None]] = None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
        run_options: Optional[ort.RunOptions] = None,
        use_io_binding: bool = False,
        copy_outputs_to_cpu: bool = True,
        chunksize: Optional[int] = None
    )->Future: ...
        """
        Submits an ONNX model for inference with the given input tensors.

        Args:
            model_path: Path to the .onnx model file.
            args: Input tensors to pass to the model. May contain `Future` objects for dependency resolution.
            kwargs: Additional keyword arguments (keyword → input name mapping). May contain `Future` objects.
            output_names: Optional subset of output names to return.
            cpu_mode_res: Resource requirements when running on CPU.
            gpu_mode_res: Resource requirements when running on GPU.
            providers: Optional ONNX Runtime provider list (e.g., `[("CUDAExecutionProvider", {...})]`).
            run_options: Optional ONNX Runtime RunOptions.
            use_io_binding: Enable zero-copy IO binding. Auto-enabled when any input is an `OrtValue` or when `copy_outputs_to_cpu=False`.
            copy_outputs_to_cpu: If True, results are copied to CPU as numpy arrays. If False, results remain as `OrtValue` objects on the device.
            chunksize: Max batch size for input concatenation along the batch dimension.
                      Multiple submit() calls with the same model_path are batched together.

        Returns:
            A concurrent.futures.Future representing the inference result.
        """

    def map(self, ...): ...
    def starmap(self, ...): ...
    def shutdown(self, ...): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
        """
        All same as ThreadPool
        """
```

### Device

A unified compute device abstraction that maps to PyTorch devices and ONNX Runtime providers.

```python
class Device:
    def __init__(self, device_type: str, device_id: int = 0, dml_id: Optional[int] = None): ...
        """
        Args:
            device_type: Device type string (e.g. "cuda", "cpu", "dml").
            device_id: Device index.
            dml_id: DirectML adapter ID (Windows only).
        """

    @property
    def supported_ort_providers(self) -> List[str]: ...
        """
        Get all ONNX Runtime providers supported by this device.
        For CUDA devices: includes CUDA, TensorRT, CPU.
        For DML devices: includes DML, CPU.
        For CPU: includes CPU, OpenVINO, CoreML (platform-dependent).
        """

    def select_provider(
        self, provider_type: str,
        device_id: Optional[int] = None,
        trt_cache_path: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict]: ...
        """
        Select a specific ONNX Runtime provider with appropriate options.
        Returns (provider_name, provider_options) tuple.

        Args:
            provider_type: One of "cuda", "tensorrt", "dml", "cpu", "openvino", "rocm", "migraphx".
            device_id: Override device ID (defaults to self.device_id).
            trt_cache_path: TensorRT engine cache directory.
        """
```

### ModelInfo

Inspect ONNX model metadata — inputs, outputs, shapes, data types, and signature.

```python
class ModelInfo:
    def __init__(self, model_path: str, hot_reload: bool = True): ...
        """
        Args:
            model_path: Path to the .onnx model file.
            hot_reload: If True, check file modification time on each access.
        """

    @property
    def inputs(self) -> List[NodeInfo]: ...
        """List of input nodes with name, shape, dtype properties."""

    @property
    def outputs(self) -> List[NodeInfo]: ...
        """List of output nodes with name, shape, dtype properties."""

    def print_info(self) -> None: ...
        """Print a formatted summary of the model's inputs and outputs."""

    def check_for_updates(self) -> bool: ...
        """Check if the model file has changed and reload if necessary. Returns True if reloaded."""

class NodeInfo:
    name: str
    shape: List[Optional[int]]
    dtype: str
```

### Future Enhancements

SmartPool extends `concurrent.futures.Future` with additional utilities for building non-blocking pipelines.

```python
from smartpool.futures import Future

class Future(concurrent.futures.Future):

    def __getitem__(self, key: Any) -> Future: ...
        """
        Create a child Future that resolves to `self.result()[key]`.
        Supports both integer indices and string/slice keys.
        Non-blocking — returns immediately.

        Example:
            first_output = infer_future[0]          # → Future resolving to first output
            named_output = infer_future["labels"]   # → Future resolving to named output
        """

    def unpack(self, n: int) -> List[Future]: ...
        """
        Create *n* child Futures that resolve to `self.result()[0]` ... `self.result()[n-1]`.
        Enables tuple unpacking of Future results.
        Non-blocking — returns immediately.

        Example:
            img, blob, scale, pad = preprocess_future.unpack(4)
            # Each variable is a Future; pass them directly to the next pipeline step
        """

    def add_start_callback(self, fn: Callable[[], None]) -> None: ...
        """
        Register a callback that fires *before* the task starts running (when
        the Future transitions from PENDING to RUNNING). Useful for progress tracking.
        """
```

Together with the **Future dependency resolution** in `submit()`, you can compose complex pipelines without blocking:

```python
preprocess_future = pool.submit(preprocess, args=(image_path,))
img, blob, scale, pad = preprocess_future.unpack(4)

# infer_future won't execute until blob is ready
infer_future = infer_session_pool.submit(model_path, args=(blob,))
outputs = infer_future[0]

# postprocess_future won't execute until all deps are ready
postprocess_future = pool.submit(postprocess, args=(img, outputs, scale, pad))
```

### Resource & DataSize

`Resource` describes the CPU/GPU resources a task requires. `DataSize` provides
human-readable memory size constants.

```python
class Resource:
    def __init__(
        self, *,
        cpu_cores: float = 1,
        cpu_cores_in_python: Optional[float] = None,
        cpu_cores_out_of_python: float = 0,
        cpu_mem: int = 0,
        gpu_cores: float = 0,
        gpu_mem: int = 0,
    ): ...
        """
        Args:
            cpu_cores: Shorthand for cpu_cores_in_python (used when
                       cpu_cores_in_python is not given).
            cpu_cores_in_python: Number of CPU cores consumed inside the
                                 Python process (e.g. by numpy/torch).
            cpu_cores_out_of_python: CPU cores consumed outside Python
                                     (e.g. by subprocesses).
            cpu_mem: CPU memory required in bytes.
            gpu_cores: Number of GPU cores (CUDA cores) required.
            gpu_mem: GPU memory required in bytes.
        """

    @property
    def cpu_cores(self) -> float:
        """Sum of cpu_cores_in_python and cpu_cores_out_of_python."""


class DataSize:
    B = 1
    KB = 1024 * B
    MB = 1024 * KB
    GB = 1024 * MB
    TB = 1024 * GB
    PB = 1024 * TB
```

Example:

```python
from smartpool import Resource, DataSize

# Request 2 CPU cores + 1 GB RAM + 1024 CUDA cores + 2 GB GPU memory
res = Resource(
    cpu_cores_in_python=2,
    cpu_mem=1 * DataSize.GB,
    gpu_cores=1024,
    gpu_mem=2 * DataSize.GB,
)
```

## License

MIT License