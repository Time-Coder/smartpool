__version__ = "0.1.7"

from .futures import Future
from .amd_gpuinfo import AMDGPUInfo
from .gpuinfo import GPUInfo, GPUInfoSnapshot, GPUVendor
from .gpuinfos import GPUInfos
from .infersessionpool import InferSessionPool, ModelInfo
from .intel_gpuinfo import IntelGPUInfo
from .interpreterpool import InterpreterPool
from .nvidia_gpuinfo import NvidiaGPUInfo
from .processpool import ProcessPool
from .resource import DataSize, Resource
from .threadpool import ThreadPool
from .utils import best_device, best_stream, limit_num_single_thread, move_optimizer_to
import os

TRT_CACHE_PATH: str = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/__trtcache__"


__all__ = [
    "Future",
    "ProcessPool",
    "ThreadPool",
    "InterpreterPool",
    "InferSessionPool",
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
    "TRT_CACHE_FOLDER"
]
