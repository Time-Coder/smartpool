from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Dict, Optional, List, Union

from ..task import Task

if TYPE_CHECKING:
    from ..gpuinfo import GPUInfoSnapshot
    from .infer_session_pool import InferSessionPool
    from .session_info import SessionInfo
    from .model_info import ModelInfo
    from ..resource import Resource

    import onnxruntime as ort
    import numpy as np


class InferSessionTask(Task):

    def __init__(
        self, infer_session_pool: InferSessionPool,
        model_info: ModelInfo, kwargs: Dict[str, Any],
        cpu_mode_res: Resource, gpu_mode_res: Resource,
        check_args: bool,
        output_names: Optional[List[str]],
        user_providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]],
        run_options: Optional[ort.RunOptions],
        use_io_binding: bool, copy_outputs_to_cpu: bool,
        chunksize: int = 1
    ):
        self.model_info: ModelInfo = model_info
        self.user_providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]] = user_providers
        self.run_options: Optional[ort.RunOptions] = run_options
        self.output_names: Optional[List[str]] = output_names
        self.session_info: Optional[SessionInfo] = None
        self.use_io_binding: bool = use_io_binding
        self.copy_outputs_to_cpu: bool = copy_outputs_to_cpu
        self._provider: Optional[Tuple[str, Dict]] = None
        Task.__init__(
            self, pool=infer_session_pool,
            func=None, args=None, kwargs=kwargs,
            cpu_mode_res=cpu_mode_res,
            gpu_mode_res=gpu_mode_res,
            check_args=check_args,
            chunksize=chunksize,
            use_torch=False,
            device_changeable=False,
            calculate_module_deps=False
        )

    def can_use(self, gpu:GPUInfoSnapshot)->bool:
        return bool(gpu.ort_provider(self.user_providers))

    def _callback_hook(self, result: Any):
        if result.__class__.__name__ == "OrtValue":
            self.use_io_binding = True

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

    def exec(self) -> Tuple[bool, Any]:
        try:
            result = self.session_info.run_with_iobinding(self.output_names, self.kwargs, self.run_options, self.copy_outputs_to_cpu)
            success = True
        except Exception as e:
            result = e
            success = False

        return success, result
    
    def run_async(self) -> None:
        try:
            self.session_info.run_async(self.output_names, input_feed=self.kwargs, callback=self.callback, user_data=None, run_options=self.run_options)
        except Exception as e:
            self.is_working = False
            infer_session_pool: InferSessionPool = self.pool
            infer_session_pool._remove_provider_running_device(self.provider)
            infer_session_pool._on_task_done(self.id, False, e)
    
    def callback(self, results: np.ndarray, user_data: None, error_str: str) -> None:
        task_id = self.id
        if error_str:
            success = False
            result = RuntimeError(error_str)
        else:
            success = True
            result = results

        infer_session_pool: InferSessionPool = self.pool
        infer_session_pool._remove_provider_running_device(self.provider)
        infer_session_pool._on_task_done(task_id, success, result)