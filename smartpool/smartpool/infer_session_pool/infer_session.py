from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional, List, Union
from collections import defaultdict
import threading
import time
import os

if TYPE_CHECKING:
    import onnxruntime as ort
    import numpy as np
    from ..device import Device


class InferSession:

    def __init__(self, model_path, *args, device: Device, **kwargs)->None:
        self._session_args = args
        self._session_kwargs = kwargs
        self._session: Optional[ort.InferenceSession] = None
        self.model_path: str = model_path
        self.provider_name: str = ""
        self.provider_options: Dict[str, Any] = {}
        self.device: Device = device
        self.device_type: str = ""
        self.device_id: int = 0
        self._gpu_index: int = -1
        self._io_bindings: Dict[Tuple[int, str], ort.IOBinding] = {}
        self._input_ortvalues: Dict[Tuple[int, str], Dict[str, ort.OrtValue]] = defaultdict(dict)
        self.estimated_cpu_mem: int = 0
        self.estimated_gpu_mem: int = 0
        self._lock: threading.Lock = threading.Lock()
        self.last_use_time: float = 0
        self.running_count: int = 0
        self._running_count_lock: threading.Lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._session is not None:
                return

            import onnxruntime as ort
            from .infer_session_pool import InferSessionPool

            self._session = ort.InferenceSession(self.model_path, *self._session_args, **self._session_kwargs)
            self.provider_name: str = self._session.get_providers()[0]
            self.provider_options: Dict[str, Any] = self._session.get_provider_options().get(self.provider_name, {})
            self.device_id = self.provider_options.get("device_id", 0)
            self.device_type = self._provider_device_type(self.provider_name)
            self._io_bindings = defaultdict(self._session.io_binding)
            cpu_mul, gpu_mul = InferSessionPool.PROVIDER_MEMORY_MULTIPLIERS.get(
                self.provider_name, (3, 0)
            )
            file_size = os.path.getsize(self.model_path)
            self.estimated_cpu_mem = file_size * cpu_mul
            self.estimated_gpu_mem = file_size * gpu_mul
            self.last_use_time = time.time()
            self._take_session_resource()
        
    def stop(self) -> None:
        with self._lock:
            if self._session is None:
                return

            self._release_session_resource()
            self._session = None
            self.device_type = ""
            self.device_id = 0
            self.provider_name = ""
            self.provider_options = {}
            self._gpu_index = -1
            self._io_bindings = {}
            self._input_ortvalues = defaultdict(dict)
            self.estimated_cpu_mem = 0
            self.estimated_gpu_mem = 0

    def _increase_running_count(self):
        with self._running_count_lock:
            self.last_use_time = time.time()
            self.running_count += 1

    def _decrease_running_count(self):
        with self._running_count_lock:
            self.running_count -= 1

    @property
    def session(self)->ort.InferenceSession:
        self.start()
        return self._session

    @staticmethod
    def _provider_device_type(provider_name: str) -> str:
        suffix = "ExecutionProvider"
        result = provider_name.lower()
        if provider_name.endswith(suffix):
            result = provider_name[:-len(suffix)].lower()

        if result == "tensorrt":
            result = "cuda"
        
        return result

    def _take_session_resource(self)->None:
        from ..pool import Pool

        with Pool._sys_info_lock:
            Pool._sys_info.cpu_mem_free -= self.estimated_cpu_mem
            if self.provider_name != "CPUExecutionProvider":
                device_id_key: str = ("device_id" if self.provider_name != "DmlExecutionProvider" else "dml_id")
                for index, gpu in enumerate(Pool._sys_info.gpu_infos):
                    gpu_device_id: int = getattr(gpu.device, device_id_key)
                    if gpu_device_id == self.device_id:
                        gpu.mem_free -= self.estimated_gpu_mem
                        self._gpu_index = index
                        break

    def _release_session_resource(self)->None:
        from ..pool import Pool

        with Pool._sys_info_lock:
            Pool._sys_info.cpu_mem_free += self.estimated_cpu_mem
            if self._gpu_index >= 0:
                Pool._sys_info.gpu_infos[self._gpu_index].mem_free += self.estimated_gpu_mem

    def _inputs_key(self, input_feed: Dict[str, Any])->Tuple[int, str]:
        import threading
        tid: int = threading.current_thread().ident
        shape_dict: Dict[str, Tuple[int, ...]] = {}
        for node in self.session.get_inputs():
            shape_dict[node.name] = input_feed[node.name].shape

        import json
        return tid, json.dumps(shape_dict, separators=(', ', ':'))

    def run_with_iobinding(self, output_names: Optional[List[str]], input_feed: Dict[str, Any], run_options: ort.RunOptions, copy_outputs_to_cpu: bool) -> Union[List[ort.OrtValue], List[np.ndarray]]:
        try:
            self._increase_running_count()
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
        finally:
            self._decrease_running_count()

    def run_async(self, output_names: Optional[List[str]], input_feed: Dict[str, Any], callback, user_data, run_options: ort.RunOptions):
        self._increase_running_count()

        def _wrapped_callback(*args, **kwargs):
            try:
                callback(*args, **kwargs)
            finally:
                self._decrease_running_count()

        try:
            self.session.run_async(output_names, input_feed, _wrapped_callback, user_data, run_options)
        except Exception as e:
            self._decrease_running_count()
            raise

    def _update_input_ortvalues(self, input_feed: Dict[str, Any]) -> Tuple[int, str]:
        import onnxruntime as ort

        key: Tuple[int, str] = self._inputs_key(input_feed)
        if key not in self._input_ortvalues:
            for node in self.session.get_inputs():
                input_value = input_feed[node.name]
                
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
