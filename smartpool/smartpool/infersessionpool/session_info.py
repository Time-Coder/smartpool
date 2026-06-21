from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional, List, Union
from collections import defaultdict

if TYPE_CHECKING:
    import onnxruntime as ort
    import numpy as np


class SessionInfo:

    def __init__(self, *args, **kwargs)->None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(*args, **kwargs)
        self._device_type: Optional[str] = None
        self._device_id: Optional[int] = None
        self._io_bindings: Dict[Tuple[int, str], ort.IOBinding] = defaultdict(self.session.io_binding)
        self._input_ortvalues: Dict[Tuple[int, str], Dict[str, ort.OrtValue]] = defaultdict(dict)
        self._output_ortvalues: Dict[Tuple[int, str], Dict[str, ort.OrtValue]] = defaultdict(dict)

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
    def device_id(self)->int:
        if self._device_id is None:
            provider_name = self.session.get_providers()[0]
            provider_options = self.session.get_provider_options().get(provider_name, {})
            self._device_id = int(provider_options.get("device_id", 0))

        return self._device_id

    def _inputs_key(self, input_feed: Dict[str, Any])->Tuple[int, str]:
        import threading
        tid: int = threading.current_thread().ident
        shape_dict: Dict[str, Tuple[int, ...]] = {}
        for node in self.session.get_inputs():
            shape_dict[node.name] = input_feed[node.name].shape

        import json
        return tid, json.dumps(shape_dict, separators=(', ', ':'))

    def run_with_iobinding(self, output_names: Optional[List[str]], input_feed: Dict[str, Any], run_options: ort.RunOptions, copy_outputs_to_cpu: bool) -> Union[List[ort.OrtValue], List[np.ndarray]]:
        key: Tuple[int, str] = self._update_input_ortvalues(input_feed)
        if key not in self._io_bindings:
            for name, ortvalue in self._input_ortvalues[key].items():
                self._io_bindings[key].bind_ortvalue_input(
                    name, ortvalue
                )

            for node in self.session.get_outputs():
                self._io_bindings[key].bind_output(
                    node.name, self.device_type, self.device_id
                )

        io_binding = self._io_bindings[key]
        self.session.run_with_iobinding(io_binding, run_options)
        ort_outputs = io_binding.get_outputs()
        self._update_output_ortvalues(key, ort_outputs)

        if copy_outputs_to_cpu:
            outputs = io_binding.copy_outputs_to_cpu()
        else:
            outputs = ort_outputs

        if output_names is None:
            return outputs
        else:
            output_dict: Dict[str, Any] = {}
            for i, node in enumerate(self.session.get_outputs()):
                output_dict[node.name] = outputs[i]

            result = []
            for output_name in output_names:
                result.append(output_dict[output_name])

            return result
    
    def _update_input_ortvalues(self, input_feed: Dict[str, Any]) -> Tuple[int, str]:
        import onnxruntime as ort

        key: Tuple[int, str] = self._inputs_key(input_feed)
        if key not in self._input_ortvalues:
            for node in self.session.get_inputs():
                input_value = input_feed[node.name]
                if (
                    input_value.__class__.__name__ == "OrtValue" and
                    input_value.device_name() == self.device_type and
                    input_value.device_id() == self.device_id
                ):
                    self._input_ortvalues[key][node.name] = input_value
                else:
                    if input_value.__class__.__name__ == "Tensor":
                        input_value = input_value.detach().cpu().numpy()

                    elif input_value.__class__.__name__ == "OrtValue":
                        input_value = input_value.numpy()

                    self._input_ortvalues[key][node.name] = ort.OrtValue.ortvalue_from_numpy(
                        input_value, device_type=self.device_type, device_id=self.device_id
                    )
        else:
            for name, ortvalue in self._input_ortvalues[key].items():
                input_value = input_feed[name]
                if ortvalue is input_value:
                    continue

                if input_value.__class__.__name__ == "OrtValue":
                    input_value = input_value.numpy()
                elif input_value.__class__.__name__ == "Tensor":
                    input_value = input_value.detach().cpu().numpy()

                import numpy as np
                target_np_view = ortvalue.numpy()
                np.copyto(target_np_view, input_value)

        return key

    def _update_output_ortvalues(self, key: Tuple[int, str], outputs: List[ort.OrtValue])->None:
        if key not in self._output_ortvalues:
            for i, node in enumerate(self.session.get_outputs()):
                self._output_ortvalues[key][node.name] = outputs[i]
                self._io_bindings[key].bind_ortvalue_output(
                    node.name, outputs[i]
                )