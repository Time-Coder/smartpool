from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from ..pool import Pool
from .infersessionworker import InferSessionWorker

if TYPE_CHECKING:
    from ..futures import Future

    import onnxruntime as ort

    from ..resource import Resource
    from .infersessiontask import InfersessionTask
    from .model_info import ModelInfo
    from .session_info import SessionInfo


class InferSessionPool(Pool):

    _not_thread_safe_providers_lock: threading.Lock = threading.Lock()
    _not_thread_safe_providers: Dict[str, Set[int]] = {
        "DmlExecutionProvider": set(),
    }

    _model_infos: Dict[str, ModelInfo] = {}

    def __init__(
        self, model_path: str, max_workers: int = 0,
        thread_name_prefix: str = "InferSessionPool.worker:",
        session_options: Optional[ort.SessionOptions] = None
    ):
        from .model_info import ModelInfo

        model_path = os.path.abspath(model_path).replace("\\", "/")
        if model_path not in InferSessionPool._model_infos:
            InferSessionPool._model_infos[model_path] = ModelInfo(model_path)

        Pool.__init__(
            self, max_workers=max_workers,
            initializer=None, initargs=(), initkwargs=None,
            result_queue_cls=None,
            worker_cls=InferSessionWorker
        )
        self._thread_name_prefix: str = thread_name_prefix
        self._session_options: Optional[ort.SessionOptions] = session_options
        self._model_info: ModelInfo = InferSessionPool._model_infos[model_path]
        self._sessions: Dict[str, SessionInfo] = {}
        self.print_info: bool = False

    @classmethod
    def instance(cls, model_path: str, pool_name: str) -> InferSessionPool:
        model_path = os.path.abspath(model_path).replace("\\", "/")

        if (model_path, pool_name) in Pool._instance_dict[cls]:
            return Pool._instance_dict[cls][model_path, pool_name]

        if pool_name not in Pool._instance_config[cls]:
            args = ()
            kwargs = {}
        else:
            args, kwargs = Pool._instance_config[cls][pool_name]

        Pool._instance_dict[cls][model_path, pool_name] = cls(model_path, *args, **kwargs)
        return Pool._instance_dict[cls][model_path, pool_name]

    @staticmethod
    def run_async(
        model_path: str,
        args: Optional[Tuple[Any, ...]]=None,
        kwargs: Optional[Dict[str, Any]]=None,
        pool_name: str="default", **submit_kwargs
    )->Future:
        infer_session_pool = InferSessionPool.instance(model_path, pool_name)
        return infer_session_pool.submit(args=args, kwargs=kwargs, **submit_kwargs)

    @property
    def model_info(self)->ModelInfo:
        return self._model_info

    @property
    def session_options(self) -> ort.SessionOptions:
        if self._session_options is None:
            import onnxruntime as ort
            self._session_options = ort.SessionOptions()
            self._session_options.enable_cpu_mem_arena = True
            self._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return self._session_options

    @staticmethod
    def _provider_key(provider: Tuple[str, Dict[str, Any]])->str:
        import json

        return "(" + provider[0] + ", " + json.dumps(provider[1], sort_keys=True, separators=(', ', ':')) + ")"

    def _session_info(self, provider: Tuple[str, Dict[str, Any]])->SessionInfo:
        model_path: str = self._model_info.model_path
        key = self._provider_key(provider)
        if key not in self._sessions:
            from .session_info import SessionInfo
            self._sessions[key] = SessionInfo(model_path, self.session_options, providers=[provider], enable_fallback=False)

        return self._sessions[key]

    def _add_provider_running_device(self, provider: Tuple[str, Dict[str, Any]])->None:
        provider_name: str = provider[0]
        device_id: int = 0
        if "device_id" in provider[1]:
            device_id: int = provider[1]["device_id"]

        with self._not_thread_safe_providers_lock:
            if provider_name in self._not_thread_safe_providers:
                self._not_thread_safe_providers[provider_name].add(device_id)

    def _remove_provider_running_device(self, provider: Tuple[str, Dict[str, Any]])->None:
        provider_name: str = provider[0]
        device_id: int = 0
        if "device_id" in provider[1]:
            device_id: int = provider[1]["device_id"]

        with self._not_thread_safe_providers_lock:
            if provider_name in self._not_thread_safe_providers:
                self._not_thread_safe_providers[provider_name].remove(device_id)

    def _can_use_provider(self, provider: Tuple[str, Dict[str, Any]])->bool:
        provider_name: str = provider[0]
        device_id: int = 0
        if "device_id" in provider[1]:
            device_id: int = provider[1]["device_id"]

        with self._not_thread_safe_providers_lock:
            return (
                provider_name not in self._not_thread_safe_providers or
                device_id not in self._not_thread_safe_providers[provider_name]
            )

    def _check_providers(self, providers:Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]])->None:
        if providers is None:
            return

        for provider in providers:
            provider_name: str = (provider if isinstance(provider, str) else provider[0])
            if provider_name == "CPUExecutionProvider":
                return

        with self._sys_info_lock:
            for gpu_info in self._sys_info.gpu_infos:
                if gpu_info.ort_provider(providers):
                    return

        raise ValueError(f"all providers are not supported: {providers}")

    def submit(
        self,
        args: Optional[Tuple[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        output_names: Optional[Union[List[str], None]] = None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
        use_io_binding: bool = False,
        output_ort_values: bool = False
    ) -> Future:
        if self._shutdown:
            raise RuntimeError("cannot submit after shutdown")
        
        from ..resource import Resource
        from .infersessiontask import InfersessionTask

        validated_kwargs, has_ort_value = self._model_info.check_args(args, kwargs)
        self._model_info.check_outputs(output_names)
        self._check_providers(providers)

        if cpu_mode_res is None:
            cpu_mode_res = Resource(cpu_cores_in_python=1)

        if gpu_mode_res is None:
            gpu_mode_res = cpu_mode_res

        if not use_io_binding:
            if output_ort_values or has_ort_value:
                use_io_binding = True

        task = InfersessionTask(
            infer_session_pool=self, args=(), kwargs=validated_kwargs,
            cpu_mode_res=cpu_mode_res, gpu_mode_res=gpu_mode_res,
            output_names=output_names, user_providers=providers,
            use_io_binding=use_io_binding, output_ort_values=output_ort_values
        )
        self._validate_resource_feasibility(task)

        with self._lock:
            self._tasks[task.id] = task
            return self._submit(task, append_to_delay=True)

    def _try_assign_task(self, task: InfersessionTask) -> bool:
        if not task.ready_to_run or self._workers_working_count >= self._max_workers:
            return False

        gpu_res = task.gpu_mode_res
        modes = []
        if gpu_res.gpu_cores > 0 or gpu_res.gpu_mem > 0:
            modes.append("gpu")
        modes.append("cpu")

        for mode in modes:
            if mode == "cpu" and task.user_providers is not None:
                has_cpu_provider: bool = False
                for provider in task.user_providers:
                    if isinstance(provider, str):
                        provider_name = provider
                    else:
                        provider_name = provider[0]

                    if provider_name == "CPUExecutionProvider":
                        has_cpu_provider: bool = True
                        break

                if not has_cpu_provider:
                    continue

            devices, _ = self._choose_task_device(task, mode, kill_workers=False)
            for device in devices:
                if isinstance(device, str):
                    task.device = device
                    task.gpu_index = -1
                    task.dml_id = -1
                else:
                    task.device = device.device
                    task.gpu_index = device.index
                    task.dml_id = device.dml_id

                if not self._choose_task_session(task) or not self._choose_task_worker(task, task.effective_res):
                    task.device = None
                    task.gpu_index = -1
                    task.dml_id = -1
                    continue

            if task.device is not None and task.session_info is not None and task.worker is not None:
                return True

        return False

    def _choose_task_session(self, task: InfersessionTask) -> Optional[ort.InferenceSession]:
        provider: Tuple[str, Dict[str, Any]] = task.provider
        if not self._can_use_provider(provider):
            task.session_info = None
            return None

        task.session_info = self._session_info(provider)
        return task.session_info

    def _put_task(self, task: InfersessionTask) -> None:
        self._take_resource(task)
        worker: InferSessionWorker = task.worker
        worker.is_working = True
        worker.add_task(task)

    def _take_resource(self, task: InfersessionTask) -> None:
        with self._sys_info_lock:
            res = task.effective_res
            task.estimated_need_cpu_mem = res.cpu_mem
            self._sys_info.cpu_cores_free -= res.cpu_cores
            self._sys_info.cpu_mem_free -= res.cpu_mem
            if task.gpu_index != -1:
                self._sys_info.gpu_infos[task.gpu_index].n_cores_free -= res.gpu_cores
                self._sys_info.gpu_infos[task.gpu_index].mem_free -= res.gpu_mem

    def _release_resource(self, task: InfersessionTask) -> None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free += res.cpu_cores
            self._sys_info.cpu_mem_free += task.estimated_need_cpu_mem
            if task.gpu_index != -1:
                self._sys_info.gpu_infos[task.gpu_index].n_cores_free += res.gpu_cores
                self._sys_info.gpu_infos[task.gpu_index].mem_free += res.gpu_mem

    def _estimate_cpu_cores_needed(self, res: Resource) -> float:
        return res.cpu_cores
