from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Dict, Optional, List, Union

from ..task import Task

if TYPE_CHECKING:
    from ..gpuinfo import GPUInfoSnapshot
    from .infersessionpool import InferSessionPool
    from .session_info import SessionInfo
    from ..resource import Resource
    import onnxruntime as ort


class InfersessionTask(Task):

    def __init__(
        self, infer_session_pool: InferSessionPool,
        args: Tuple[Any, ...], kwargs: Dict[str, Any],
        cpu_mode_res: Resource, gpu_mode_res: Resource,
        output_names: Optional[List[str]],
        user_providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]],
        use_io_binding: bool, output_ort_values: bool
    ):
        self.user_providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]] = user_providers
        self.output_names: Optional[List[str]] = output_names
        self.session_info: Optional[SessionInfo] = None
        self.use_io_binding: bool = use_io_binding
        self.output_ort_values: bool = output_ort_values
        self._provider: Optional[Tuple[str, Dict]] = None
        Task.__init__(
            self, pool=infer_session_pool,
            func=None, args=args, kwargs=kwargs,
            cpu_mode_res=cpu_mode_res,
            gpu_mode_res=gpu_mode_res,
            use_torch=False,
            device_changeable=False,
            calculate_module_deps=False
        )

    def can_use(self, gpu:GPUInfoSnapshot)->bool:
        return bool(gpu.ort_provider(self.user_providers))

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
        self._provider = None

    @property
    def provider(self)->Optional[Tuple[str, Dict]]:
        if self.device is None:
            return None

        if self._provider is not None:
            return self._provider

        if self.device == "cpu":
            self._provider = ("CPUExecutionProvider", {})
            return self._provider

        from ..pool import Pool
        with Pool._sys_info_lock:
            gpuinfo_snapshot = Pool._sys_info.gpu_infos[self.gpu_index]

        self._provider = gpuinfo_snapshot.ort_provider(self.user_providers)
        return self._provider
    
    @staticmethod
    def _provider_device_type(provider_name: str) -> str:
        suffix = "ExecutionProvider"
        if provider_name.endswith(suffix):
            return provider_name[:-len(suffix)].lower()
        return provider_name.lower()

    def _bind_input(self, io_binding: ort.IOBinding, name: str, value: Any) -> None:
        import numpy as np

        if isinstance(value, np.ndarray):
            io_binding.bind_cpu_input(name, value)
            return

        import onnxruntime as ort

        if isinstance(value, ort.OrtValue):
            io_binding.bind_ortvalue_input(name, value)
            return

        if hasattr(value, "data_ptr") and getattr(value, "is_cuda", False):
            provider = self.provider
            device_type = self._provider_device_type(provider[0])
            device_id = provider[1].get("device_id", 0)

            from .model_info import ModelInfo
            io_binding.bind_input(
                name,
                device_type=device_type,
                device_id=device_id,
                element_type=ModelInfo.get_dtype(value),
                shape=ModelInfo.get_shape(value),
                buffer_ptr=value.data_ptr(),
            )
            return

        raise TypeError(f"IO binding does not support input type: {type(value)}")

    def _exec(self) -> Any:
        session = self.session_info.session
        io_binding = self.session_info.io_binding

        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        for name, value in self.kwargs.items():
            self._bind_input(io_binding, name, value)

        output_names = self.output_names
        if output_names is None:
            output_names = [o.name for o in session.get_outputs()]

        provider = self.provider
        if provider and provider[0] != "CPUExecutionProvider":
            device_type = self._provider_device_type(provider[0])
            device_id = provider[1].get("device_id", 0)
            for name in output_names:
                io_binding.bind_output(name, device_type=device_type, device_id=device_id)
        else:
            for name in output_names:
                io_binding.bind_output(name)

        session.run_with_iobinding(io_binding, output_names)

        outputs = io_binding.get_outputs()
        if self.output_ort_values:
            return outputs
        else:
            return [o.numpy() for o in outputs]

    def exec(self) -> Tuple[bool, Any]:
        try:
            result = self._exec()
            success = True
        except Exception as e:
            result = e
            success = False

        return success, result