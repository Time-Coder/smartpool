from __future__ import annotations
import uuid
import sys
import os
from concurrent.futures import Future
from typing import List, Tuple, Any, Dict, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .worker import Worker
    from .resource import Resource


class Task:

    def __init__(self, func:Callable[..., Any], args:Tuple[Any, ...], kwargs:Dict[str, Any],
                 cpu_mode_res: Resource, gpu_mode_res: Resource,
                 use_torch: bool, use_onnx: bool,
                 calculate_module_deps:bool):
        self.id:str = str(uuid.uuid4())
        self.func:Callable[..., Any] = func
        self.args:Tuple[Any] = args
        self.kwargs:Dict[str, Any] = kwargs
        self.cpu_mode_res: Resource = cpu_mode_res
        self.gpu_mode_res: Resource = gpu_mode_res
        self.use_torch: bool = use_torch
        self.use_onnx: bool = use_onnx
        self.estimated_need_cpu_mem:float = 0.0
        self.modules_overlap_ratio:float = 0.0
        self.module_deps:Dict[str, int] = {}
        
        if calculate_module_deps:
            from .module_deps import module_deps
            self.module_deps:Dict[str, int] = module_deps(sys.modules[func.__module__])
        
        self._device:Optional[str] = None
        self.gpu_index:int = -1
        self.dml_id:int = -1
        self.worker:Worker = None
        self.mem_before_enter:int = 0
        self._onnx_provider:Optional[Tuple[str, Dict]] = None
        self.future = Future()

    @property
    def device(self)->str:
        return self._device
    
    @device.setter
    def device(self, device:str)->None:
        if self._device == device:
            return

        self._device = device
        self._onnx_provider = None

    @property
    def onnx_provider(self)->Optional[Tuple[str, Dict]]:
        if not self.use_onnx or self.device is None:
            return None
        
        if self._onnx_provider is not None:
            return self._onnx_provider
        
        if self.device == "cpu":
            self._onnx_provider = ("CPUExecutionProvider", {})
            return self._onnx_provider

        from .gpuinfo import GPUInfo

        items = self.device.split(":")
        device_prefix = items[0]
        device_id = int(items[1])

        GPUInfo.init_onnx_providers()
        gpuinfo_class = GPUInfo.gpuinfo_class(device_prefix)
        for provider_name in gpuinfo_class.supported_onnx_providers:
            options = {"device_id": device_id}
            if provider_name == "DmlExecutionProvider":
                options["device_id"] = self.dml_id
            elif provider_name == "TensorrtExecutionProvider":
                options["trt_engine_cache_enable"] = True
                options["trt_engine_cache_path"] = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/__trtcache__"
            self._onnx_provider = (provider_name, options)
            return self._onnx_provider

        self._onnx_provider = ("CPUExecutionProvider", {})
        return self._onnx_provider

    @property
    def effective_res(self) -> Resource:
        if self.device and self.device != "cpu":
            return self.gpu_mode_res
        
        return self.cpu_mode_res

    def info(self):
        return (self.id, self.device, self.func, self.args, self.kwargs)
    
    @property
    def device_id(self)->int:
        if isinstance(self.device, str) and ":" in self.device:
            parts = self.device.split(":")
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except (ValueError, IndexError):
                    pass
        return -1

    def exec(self)->Tuple[bool, Any]:
        try:
            result = self.func(*self.args, **self.kwargs)
            success = True
        except BaseException as e:
            result = e
            success = False

        return success, result