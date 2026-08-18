from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

_VENDOR_ORT_PROVIDERS: Dict[str, List[str]] = {}
_TRT_CACHE_PATH: Optional[str] = None


def _init_ort_providers_cache() -> None:
    global _VENDOR_ORT_PROVIDERS
    if _VENDOR_ORT_PROVIDERS:
        return

    import onnxruntime as ort
    available = ort.get_available_providers()

    vendor_lists: Dict[str, List[str]] = {
        "cuda": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"],
        "xpu": ["OpenVINOExecutionProvider", "DnnlExecutionProvider", "DmlExecutionProvider"],
        "hip": ["ROCMExecutionProvider", "MIGraphXExecutionProvider", "DmlExecutionProvider"],
        "cpu": ["CPUExecutionProvider"]
    }

    for prefix, providers in vendor_lists.items():
        _VENDOR_ORT_PROVIDERS[prefix] = [p for p in providers if p in available]

    global _TRT_CACHE_PATH
    _TRT_CACHE_PATH = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/__trtcache__"


class Device:

    def __init__(self, torch_device: str = "cpu", gpu_index: int = -1, dml_id: int = -1):
        self.torch_device = torch_device
        self.gpu_index = gpu_index
        self.dml_id = dml_id
        self._ort_provider: Optional[Tuple[str, Dict[str, Any]]] = None
        if ":" in torch_device:
            parts = torch_device.split(":", 1)
            self.device_prefix = parts[0]
            self.device_id = int(parts[1])
        else:
            self.device_prefix = torch_device
            self.device_id = -1

    def __copy__(self) -> "Device":
        new_device = Device(self.torch_device, self.gpu_index, self.dml_id)
        new_device._ort_provider = self._ort_provider
        return new_device

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Device":
        new_device = Device(self.torch_device, self.gpu_index, self.dml_id)
        memo[id(self)] = new_device
        new_device._ort_provider = copy.deepcopy(self._ort_provider, memo)
        return new_device

    def __eq__(self, other) -> bool:
        if isinstance(other, Device):
            return self.torch_device == other.torch_device
        if isinstance(other, str):
            return self.torch_device == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.torch_device)

    @property
    def supported_ort_providers(self) -> List[str]:
        _init_ort_providers_cache()
        return _VENDOR_ORT_PROVIDERS[self.device_prefix]

    def select_provider(self, providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        for provider_name in self.supported_ort_providers:
            options: Dict[str, Any] = {"device_id": self.device_id}
            if provider_name == "DmlExecutionProvider":
                options["device_id"] = self.dml_id
            elif provider_name == "TensorrtExecutionProvider":
                options["trt_engine_cache_enable"] = True
                options["trt_engine_cache_path"] = _TRT_CACHE_PATH

            current_provider = (provider_name, options)
            if providers is None:
                self._ort_provider = current_provider
                return current_provider

            for provider in providers:
                if isinstance(provider, str):
                    if provider == provider_name:
                        self._ort_provider = current_provider
                        return current_provider
                else:
                    user_provider_name = provider[0]
                    if user_provider_name != provider_name:
                        continue

                    user_provider_options = provider[1]
                    if "device_id" in user_provider_options:
                        device_id: int = user_provider_options["device_id"]
                        if device_id != options["device_id"]:
                            continue
                    else:
                        user_provider_options["device_id"] = options["device_id"]

                    self._ort_provider = (user_provider_name, user_provider_options)
                    return self._ort_provider

        self._ort_provider = None
        return None

    @property
    def ort_provider(self)->Optional[Tuple[str, Dict[str, Any]]]:
        return self._ort_provider
