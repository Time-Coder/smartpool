from __future__ import annotations

import os
import sys
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from concurrent.futures import Future
    from .gpuinfo import GPUInfo
    from .resource import Resource
    from .worker import Worker


class Task:

    _torch_gpu_backend: Optional[str] = None

    def __init__(
        self, func: Union[Callable[..., Any], str], args: Tuple[Any, ...], kwargs: Dict[str, Any],
        cpu_mode_res: Resource, gpu_mode_res: Resource,
        use_torch: bool, device_changeable: bool, calculate_module_deps: bool
    ):
        self.id: str = str(uuid.uuid4())
        self.func: Union[Callable[..., Any], str] = func
        self.args: Tuple[Any] = args
        self.kwargs: Dict[str, Any] = kwargs
        self.cpu_mode_res: Resource = cpu_mode_res
        self.gpu_mode_res: Resource = gpu_mode_res
        self.use_torch: bool = use_torch
        self.device_changeable: bool = device_changeable
        self.estimated_need_cpu_mem: float = 0.0
        self.modules_overlap_ratio: float = 0.0
        self.module_deps: Dict[str, int] = {}
        self.user_providers: Optional[List[str]] = None
        self.output_names: List[str] = []

        if calculate_module_deps:
            from .module_deps import module_deps
            self.module_deps:Dict[str, int] = module_deps(sys.modules[func.__module__])

        self._device: Optional[str] = None
        self._device_prefix: Optional[str] = None
        self._device_id: Optional[int] = None
        self.gpu_index: int = -1
        self.dml_id: int = -1
        self.worker: Worker = None
        self.mem_before_enter: int = 0
        self._onnx_provider: Optional[Tuple[str, Dict]] = None
        self._future: Optional[Future] = None

    @property
    def future(self)->Future:
        if self._future is None:
            from concurrent.futures import Future

            self._future = Future()

        return self._future

    @property
    def device(self)->str:
        return self._device

    @device.setter
    def device(self, device:str)->None:
        if self._device == device:
            return

        self._device = device
        self._device_prefix = None
        self._device_id = None
        self._onnx_provider = None

    @property
    def onnx_provider(self)->Optional[Tuple[str, Dict]]:
        if self.device is None:
            return None

        if self._onnx_provider is not None:
            return self._onnx_provider

        if self.device == "cpu":
            self._onnx_provider = ("CPUExecutionProvider", {})
            return self._onnx_provider

        from .gpuinfo import GPUInfo

        gpuinfo_class = GPUInfo.gpuinfo_class(self.device_prefix)
        for provider_name in gpuinfo_class.supported_onnx_providers():
            options = {"device_id": self.device_id}
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
        if self._device_id is not None:
            return self._device_id

        self._device_prefix = ""
        self._device_id = 0
        device = self._device
        if isinstance(device, str):
            if ":" in device:
                parts = device.split(":")
                if len(parts) == 2:
                    try:
                        self._device_prefix = parts[0]
                        self._device_id = int(parts[1])
                    except (ValueError, IndexError):
                        pass
            else:
                self._device_prefix = device

        return self._device_id
    
    @property
    def device_prefix(self)->str:
        if self._device_prefix is not None:
            return self._device_prefix

        self._device_prefix = ""
        self._device_id = -1
        device = self._device
        if isinstance(device, str):
            if ":" in device:
                parts = device.split(":")
                if len(parts) == 2:
                    try:
                        self._device_prefix = parts[0]
                        self._device_id = int(parts[1])
                    except (ValueError, IndexError):
                        pass
            else:
                self._device_prefix = device

        return self._device_prefix

    def exec(self)->Tuple[bool, Any]:
        try:
            result = self.func(*self.args, **self.kwargs)
            success = True
        except BaseException as e:
            result = e
            success = False

        return success, result

    def filter_gpu_infos(self, gpu_infos: List[GPUInfo]) -> Iterable[GPUInfo]:
        if not self.use_torch or not gpu_infos:
            return gpu_infos

        if Task._torch_gpu_backend is None:
            import torch
            if torch.cuda.is_available():
                Task._torch_gpu_backend = "cuda"

            if getattr(torch, "hip", None) and torch.hip.is_available():
                Task._torch_gpu_backend = "hip"
            elif getattr(torch, "xpu", None) and torch.xpu.is_available():
                Task._torch_gpu_backend = "xpu"

        return filter(lambda gpu: gpu.device.startswith(Task._torch_gpu_backend), gpu_infos)
