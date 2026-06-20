from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional
from collections import defaultdict

if TYPE_CHECKING:
    import onnxruntime as ort


class SessionInfo:

    def __init__(self, *args, **kwargs)->None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(*args, **kwargs)
        self._device_type: Optional[str] = None
        self._device_id: Optional[int] = None
        self._io_bindings: Dict[Tuple[int, str], ort.IOBinding] = defaultdict(self.session.io_binding)
        self._input_ortvalues: Dict[Tuple[int, str], Dict[str, ort.OrtValue]] = defaultdict(dict)

    @staticmethod
    def provider_device_type(provider_name: str) -> str:
        suffix = "ExecutionProvider"
        result = provider_name.lower()
        if provider_name.endswith(suffix):
            result = provider_name[:-len(suffix)].lower()

        if result == "tensorrt":
            result = "cuda"
        
        return result

    @property
    def device_type(self)->str:
        if self._device_type is None:
            self._device_type = self.provider_device_type(self.session.get_providers()[0])

        return self._device_type

    @property
    def device_id(self)->str:
        if self._device_id is None:
            provider_options = self.session.get_provider_options()
            if "device_id" in provider_options:
                self._device_id = provider_options["device_id"]
            else:
                self._device_id = 0

        return self._device_id

    def key(self, kwargs: Dict[str, Any])->Tuple[int, str]:
        import threading
        tid: int = threading.current_thread().ident
        shape_dict: Dict[str, Tuple[int, ...]] = {}
        for node in self.session.get_inputs():
            shape_dict[node.name] = kwargs[node.name].shape

        import json
        return tid, json.dumps(shape_dict, separators=(', ', ':'))

    def io_binding(self, kwargs: Dict[str, Any]) -> ort.IOBinding:
        key = self._update_input_ortvalues(kwargs)
        if key not in self._io_bindings:
            for name, ortvalue in self._input_ortvalues[key].items():
                self._io_bindings[key].bind_ortvalue_input(
                    name, ortvalue
                )

            for node in self.session.get_outputs():
                self._io_bindings[key].bind_output(
                    node.name, self.device_type, self.device_id, node.type
                )
    
        return self._io_bindings[key]

    def _update_input_ortvalues(self, kwargs: Dict[str, Any]) -> str:
        import onnxruntime as ort

        key = self.key(kwargs)
        if key not in self._input_ortvalues:
            for node in self.session.get_inputs():
                src_value = kwargs[node.name]
                if src_value.__class__.__name__ == "OrtValue":
                    src_value = src_value.numpy()
                elif src_value.__class__.__name__ == "Tensor":
                    src_value = src_value.detach().cpu().numpy()

                self._input_ortvalues[key][node.name] = ort.OrtValue.ortvalue_from_numpy(
                    src_value, device_type=self.device_type, device_id=self.device_id
                )
        else:
            for name, ortvalue in self._input_ortvalues[key].items():
                src_value = kwargs[name]
                if src_value.__class__.__name__ == "OrtValue":
                    src_value = src_value.numpy()
                elif src_value.__class__.__name__ == "Tensor":
                    src_value = src_value.detach().cpu().numpy()

                import numpy as np
                target_np_view = ortvalue.numpy()
                np.copyto(target_np_view, src_value)

        return key
