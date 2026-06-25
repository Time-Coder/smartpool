from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, Optional
import numpy as np
from .infer_session_task import InfersessionTask
from ..resource import Resource

if TYPE_CHECKING:
    from .infer_session_pool import InferSessionPool
    from ..futures import Future
    from .model_info import ModelInfo
    import onnxruntime as ort


class InferChunkTask(InfersessionTask):

    _default_providers_str: Optional[str] = None

    def __init__(
        self, pool: InferSessionPool,
        model_info: ModelInfo,
        use_io_binding: bool,
        user_providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]],
        run_options: ort.RunOptions,
        chunksize: int,
        key: str
    ):
        self._sub_tasks: List[InfersessionTask] = []
        self._kwargs_list: List[Dict[str, Any]] = []
        self._submitted: bool = False
        self._key: Tuple[str, int] = key
        self.last_add_time: float = 0.0

        InfersessionTask.__init__(
            self,
            infer_session_pool=pool,
            model_info=model_info,
            kwargs={},
            cpu_mode_res=Resource(),
            gpu_mode_res=Resource(),
            check_args=False,
            user_providers=user_providers,
            run_options=run_options,
            use_io_binding=use_io_binding,
            copy_outputs_to_cpu=True,
            output_names=None,
            chunksize=chunksize
        )
        self.future.add_done_callback(self._fan_out_batch_results)

    @staticmethod
    def get_key(model_path: str, use_io_binding: bool, user_providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]], run_options: ort.RunOptions, chunksize: int) -> Tuple[str, bool, str, str, int]:
        return (
            model_path, use_io_binding,
            InferChunkTask._providers_to_str(user_providers),
            InferChunkTask._run_options_to_str(run_options),
            chunksize
        )

    @staticmethod
    def _run_options_to_str(run_options: ort.RunOptions) -> str:
        import json
        result_dict = {}

        known_attrs = [
            "log_severity_level",
            "log_verbosity_level",
            "run_tag"
        ]
        for attr in known_attrs:
            if hasattr(run_options, attr):
                try:
                    result_dict[attr] = getattr(run_options, attr)
                except Exception:
                    pass

        return json.dumps(result_dict, sort_keys=True, separators=(", ", ":"))

    @staticmethod
    def _providers_to_str(providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]]):
        import json

        if providers is not None:
            for i, provider in enumerate(providers):
                if isinstance(provider, str):
                    providers[i] = (provider, {})
            return json.dumps(providers, sort_keys=True, separators=(", ", ":"))
        else:
            if InferChunkTask._default_providers_str is None:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                for i, provider in enumerate(providers):
                    if isinstance(provider, str):
                        providers[i] = (provider, {})

                InferChunkTask._default_providers_str = json.dumps(providers, sort_keys=True, separators=(", ", ":"))

            return InferChunkTask._default_providers_str

    def add_task(self, sub_task: InfersessionTask) -> None:
        if self._submitted:
            return

        self._sub_tasks.append(sub_task)
        self._kwargs_list.append(sub_task.kwargs)
        self._update_res(self.cpu_mode_res, sub_task.cpu_mode_res)
        self._update_res(self.gpu_mode_res, sub_task.gpu_mode_res)

        try:
            self.pool._validate_resource_feasibility(self)
        except Exception as e:
            for sub_task in self._sub_tasks:
                sub_task.future.set_exception(e)

            if self._key in self.pool._chunk_tasks:
                del self.pool._chunk_tasks[self._key]
                
            return

        import time
        self.last_add_time = time.time()
        if len(self._sub_tasks) >= self.chunksize:
            self.submit()

    def submit(self) -> None:
        if self._submitted:
            return
        
        self._submitted = True

        self.kwargs = {}
        for name in self.model_info.inputs:
            parts = [self._to_numpy(kwargs[name]) for kwargs in self._kwargs_list]
            self.kwargs[name] = np.concatenate(parts, axis=0)

        if self._key in self.pool._chunk_tasks:
            del self.pool._chunk_tasks[self._key]

        self.pool._submit(self)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        cls = value.__class__.__name__
        if cls == "Tensor":
            return value.detach().cpu().numpy()
        
        if cls == "OrtValue":
            return value.numpy()
        
        return value

    def _fan_out_batch_results(self, future: Future) -> None:
        if future.cancelled():
            for task in self._sub_tasks:
                task.future.cancel()

            return

        try:
            batch_result = future.result()
        except Exception as e:
            for task in self._sub_tasks:
                task.future.set_exception(e)

            return
        
        batch_output_dict = {}
        full_output_names = []
        for node, batch_output in zip(self.model_info.outputs, batch_result):
            batch_output_dict[node.name] = batch_output
            full_output_names.append(node.name)

        for i, task in enumerate(self._sub_tasks):
            try:
                outputs = []
                output_names = task.output_names
                if output_names is None:
                    output_names = full_output_names

                for output_name in output_names:
                    outputs.append(batch_output_dict[output_name][i:i+1])

                task.future.set_result(outputs)
            except Exception as e:
                task.future.set_exception(e)

    @staticmethod
    def _update_res(src_res: Resource, target_res: Resource) -> None:
        if target_res.cpu_cores_in_python > src_res.cpu_cores_in_python:
            src_res.cpu_cores_in_python = target_res.cpu_cores_in_python

        if target_res.cpu_cores_out_of_python > src_res.cpu_cores_out_of_python:
            src_res.cpu_cores_out_of_python = target_res.cpu_cores_out_of_python

        if target_res.cpu_mem > src_res.cpu_mem:
            src_res.cpu_mem = target_res.cpu_mem

        if target_res.gpu_cores > src_res.gpu_cores:
            src_res.gpu_cores = target_res.gpu_cores
        
        if target_res.gpu_mem > src_res.gpu_mem:
            src_res.gpu_mem = target_res.gpu_mem
