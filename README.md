# SmartPool

SmartPool is a Python library that provides intelligent resource-aware pooling mechanisms for parallel computing. It automatically manages CPU and GPU resources to optimize performance while preventing resource exhaustion.

## Features

- **Multiple Pool Types**: ProcessPool, ThreadPool, and InterpreterPool for different use cases
- **Intuitive API Design**: Almost the same usage as `concurrent.futures` pools
- **Automatic Resource Management**: Monitors and manages CPU cores, memory, and GPU resources
- **Hardware-Aware Scheduling**: Automatically detects system resources and schedules tasks accordingly
- **PyTorch Integration**: Support for PyTorch multiprocessing with tensor sharing to avoid serialization
- **Training Hot Migration**: Automatically moves CPU training tasks to GPU when `best_device()` changes

## Installation

```bash
pip install pysmartpool
```

## Examples

### Basic Usage

```python
from smartpool import ProcessPool

# Create a process pool that automatically manages system resources
with ProcessPool() as pool:
    # Submit tasks with proper argument passing
    futures = [pool.submit(expensive_computation, args=(arg,)) for arg in arguments]
    
    # Get results
    results = [future.result() for future in futures]
```

### Resource-Aware Task Scheduling

```python
from smartpool import ProcessPool, DataSize

# Tasks can specify their resource requirements
def memory_intensive_task(data):
    # Your computation here
    return processed_data

with ProcessPool(use_torch=True) as pool:
    # Pool automatically schedules tasks based on available memory
    future = pool.submit(
        memory_intensive_task, 
        args=(large_dataset,),
        need_cpu_cores=2,           # Request 2 CPU cores
        need_cpu_mem=1*DataSize.GB, # Request 4GB RAM
        need_gpu_cores=1024,        # Request 1024 CUDA cores (NOT percentage)
        need_gpu_mem=1*DataSize.GB  # Request 2GB GPU memory
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
    device = best_device() # <-- get best suitable device at init time
    old_device = device

    for epoch in range(epochs):
        for x, y in data_loader:
            device = best_device() # <-- get best suitable device at each batch
            x, y = x.to(device), y.to(device)

            if old_device != device:
                model.to(device) # move model to new device
                move_optimizer_to(optimizer, device) # move optimizer to new device
                old_device = device
            
                do_other_things()

with ProcessPool(use_torch=True) as pool:
    future = pool.submit(training_task, args=(model, optimizer, data))
```

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
        need_cpu_cores:int=1, need_cpu_mem:int=0,
        need_gpu_cores:int=0, need_gpu_mem:int=0
    )->concurrent.futures.Future: ...
        """
        Submits a callable to be executed with the given arguments.
    
        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Args:
            func: The callable to execute.
            args: The arguments to pass to the callable.
            kwargs: The keyword arguments to pass to the callable.
            need_cpu_cores: The number of CPU cores required for the task.
            need_cpu_mem: The amount of CPU memory required for the task.
            need_gpu_cores: The number of CUDA cores required for the task.
            need_gpu_mem: The amount of GPU memory required for the task.
        
        Returns:
            A concurrent.futures.Future representing the given call.
        """

    def map(
        self, func:Callable[..., Any],
        iterable:Iterable[Any],
        need_cpu_cores:Union[int, Iterable[int]]=1,
        need_cpu_mem:Union[int, Iterable[int]]=0,
        need_gpu_cores:Union[int, Iterable[int]]=0,
        need_gpu_mem:Union[int, Iterable[int]]=0,
        timeout:Optional[Union[float, int]]=None,
        chunksize:int=1
    )->Iterable[Any]: ...
        """
        Returns an iterator equivalent to map(func, iterable).
 
        Args:
            func: A callable that will take as many arguments as there are
                passed iterables.
            iterable: An iterable whose items will be passed to func as arguments.
            need_cpu_cores: The number of CPU cores required for the each task.
            need_cpu_mem: The amount of CPU memory required for each task.
            need_gpu_cores: The number of CUDA cores required for each task.
            need_gpu_mem: The amount of GPU memory required for each task.
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
        iterable:Iterable[Any],
        need_cpu_cores:Union[int, Iterable[int]]=1,
        need_cpu_mem:Union[int, Iterable[int]]=0,
        need_gpu_cores:Union[int, Iterable[int]]=0,
        need_gpu_mem:Union[int, Iterable[int]]=0,
        timeout:Optional[Union[float, int]]=None,
        chunksize:int=1
    )->Iterable[Any]: ...
        """
        Like `map()` method but the elements of the `iterable` are expected to
        be iterables as well and will be unpacked as arguments. Hence
        `func` and (a, b) becomes func(a, b).
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
        Initializes a new ProcessPool instance.

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

## License

MIT License