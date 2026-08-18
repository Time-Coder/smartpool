import psutil  # must import for first fetch cpu percent

__version__ = "0.1.7"

from .amd_gpuinfo import AMDGPUInfo
from .device import Device
from .futures import Future
from .gpuinfo import GPUInfo, GPUInfoSnapshot, GPUVendor
from .gpuinfos import GPUInfos
from .infer_session_pool import InferSessionPool, InferModel, ModelInfo
from .intel_gpuinfo import IntelGPUInfo
from .interpreter_pool import InterpreterPool
from .nvidia_gpuinfo import NvidiaGPUInfo
from .process_pool import ProcessPool
from .resource import DataSize, Resource
from .thread_pool import ThreadPool
from .utils import (
    TRT_CACHE_PATH,
    best_device,
    best_stream,
    limit_num_single_thread,
    move_optimizer_to,
)

__all__ = [
    "Future",
    "Device",
    "ProcessPool",
    "ThreadPool",
    "InterpreterPool",
    "InferSessionPool",
    "InferModel",
    "ModelInfo",
    "DataSize",
    "Resource",
    "move_optimizer_to",
    "limit_num_single_thread",
    "best_device",
    "best_stream",
    "GPUInfo",
    "GPUInfos",
    "GPUVendor",
    "GPUInfoSnapshot",
    "NvidiaGPUInfo",
    "IntelGPUInfo",
    "AMDGPUInfo",
    "TRT_CACHE_PATH"
]
