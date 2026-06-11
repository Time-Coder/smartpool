from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from ..pool import Pool

if TYPE_CHECKING:
    from concurrent.futures import Future

    import onnxruntime as ort

    from ..resource import Resource
    from ..task import Task
    from .model_info import ModelInfo


class InferSessionPool(Pool):

    _thread_safe_providers: Set[str] = {
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    }

    _model_infos: Dict[str, ModelInfo] = {}
    _session_options: Optional[ort.SessionOptions] = None
    _shared_sessions: Dict[str, ort.InferenceSession] = {}
    _class_session_creation_lock: threading.Lock = threading.Lock()
    _model_mem_multiplier: float = 2.0

    def __init__(self, model_path: str, max_workers: int = 0):
        from .model_info import ModelInfo

        model_path = os.path.abspath(model_path).replace("\\", "/")
        if model_path not in InferSessionPool._model_infos:
            InferSessionPool._model_infos[model_path] = ModelInfo(model_path)

        Pool.__init__(
            self, max_workers=max_workers,
            initializer=None, initargs=(), initkwargs=None,
            result_queue_cls=None,
            worker_cls=None,
            use_onnx=True,
        )
        self._model_info: ModelInfo = InferSessionPool._model_infos[model_path]
        self._cpu_session: Optional[ort.InferenceSession] = None
        self._in_flight_count: int = 0
        self._in_flight_counts: Dict[str, int] = {}
        self._all_done: threading.Event = threading.Event()
        self._session_reserved: Dict[str, int] = {}

    @staticmethod
    def _get_session_options() -> ort.SessionOptions:
        if InferSessionPool._session_options is None:
            import onnxruntime as ort
            InferSessionPool._session_options = ort.SessionOptions()
            InferSessionPool._session_options.enable_cpu_mem_arena = True
            InferSessionPool._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return InferSessionPool._session_options

    @staticmethod
    def _is_provider_thread_safe(provider: Tuple[str, Dict]) -> bool:
        return provider[0] in InferSessionPool._thread_safe_providers

    def _session_key(self, device_key: str, user_providers=None) -> str:
        key = f"{self._model_info.model_path}:{device_key}"
        if user_providers:
            providers_str = "|".join(
                p[0] if isinstance(p, tuple) else p for p in user_providers
            )
            key += f":providers={providers_str}"

        return key

    def _device_capacity(self, provider: Tuple[str, Dict]) -> int:
        return self._max_workers if self._is_provider_thread_safe(provider) else 1

    @staticmethod
    def _get_primary_provider(provider: Union[str, Tuple[str, Dict]]) -> str:
        return provider[0] if isinstance(provider, tuple) else provider

    def _effective_provider(self, task: Task) -> Tuple[str, Dict]:
        user = task.user_providers
        if user:
            p = user[0]
            return (p, {}) if isinstance(p, str) else p
        
        return task.onnx_provider

    def _effective_primary_provider(self, task: Task) -> str:
        return self._get_primary_provider(self._effective_provider(task))

    @property
    def _model_mem(self) -> int:
        return int(self._model_info.file_size * self._model_mem_multiplier)

    def submit(
        self,
        args: Optional[Tuple[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        output_names: Optional[Union[List[str], None]] = None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
    ) -> Future:
        from ..resource import Resource
        from ..task import Task

        if self._cpu_session is None:
            import onnxruntime as ort
            self._cpu_session = ort.InferenceSession(
                self._model_info.model_path,
                InferSessionPool._get_session_options(),
                providers=["CPUExecutionProvider"],
            )

        validated_kwargs = self._model_info.check_args(self._cpu_session, args, kwargs)

        if cpu_mode_res is None:
            cpu_mode_res = Resource(cpu_cores_in_python=1)

        if gpu_mode_res is None:
            gpu_mode_res = cpu_mode_res

        task = Task(
            func=None, args=(), kwargs=validated_kwargs,
            cpu_mode_res=cpu_mode_res, gpu_mode_res=gpu_mode_res,
            use_torch=False, device_changeable=False,
            calculate_module_deps=False,
        )
        task.output_names = output_names
        task.user_providers = providers
        self._validate_resource_feasibility(task)

        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot submit after shutdown")
            
            self._tasks[task.id] = task
            if not self._try_assign_infer(task):
                self._delayed_tasks.append(task)

        return task.future

    def _try_assign_infer(self, task: Task) -> bool:
        if not self._try_assign_device(task):
            return False
        
        if not self._try_acquire_slot(task):
            task.device = None
            task.gpu_index = -1
            task.dml_id = -1
            return False
        
        self._dispatch_infer(task)
        return True

    def _try_assign_device(self, task: Task) -> bool:
        gpu_res = task.gpu_mode_res
        if gpu_res.gpu_cores > 0 or gpu_res.gpu_mem > 0:
            device, _ = self._choose_task_device(task, "gpu", kill_workers=False)
            if device is not None:
                return True
            
        device, _ = self._choose_task_device(task, "cpu", kill_workers=True)
        return device is not None

    def _try_acquire_slot(self, task: Task) -> bool:
        provider = self._effective_provider(task)
        device_key = task.device
        if self._in_flight_count >= self._max_workers:
            return False
        
        if self._in_flight_counts.get(device_key, 0) >= self._device_capacity(provider):
            return False
        
        self._in_flight_count += 1
        self._in_flight_counts[device_key] = self._in_flight_counts.get(device_key, 0) + 1
        return True

    def _decrement_in_flight_count(self, task: Task) -> None:
        self._in_flight_count -= 1
        device = task.device
        val = self._in_flight_counts.get(device, 0) - 1
        if val <= 0:
            self._in_flight_counts.pop(device, None)
        else:
            self._in_flight_counts[device] = val
        if self._shutdown and self._in_flight_count == 0:
            self._all_done.set()

    def _dispatch_infer(self, task: Task) -> None:
        self._take_resource(task)
        task.future.set_running_or_notify_cancel()

        output_names = task.output_names
        task_id = task.id

        def callback(result, user_data):
            with self._lock:
                task = self._tasks.pop(task_id, None)
                if task is None:
                    return

                self._release_resource(task)
                self._decrement_in_flight_count(task)
                self._postprocess_after_task_done()
                task.future.set_result(result)

        try:
            session = self._get_or_create_session(task)
            session.run_async(output_names, task.kwargs, callback)
        except BaseException as e:
            self._tasks.pop(task_id, None)
            self._release_resource(task)
            self._decrement_in_flight_count(task)
            self._postprocess_after_task_done()
            task.future.set_exception(e)

    def _get_or_create_session(self, task: Task) -> ort.InferenceSession:
        import onnxruntime as ort

        user_providers = task.user_providers
        device_key = task.device

        if device_key == "cpu":
            if self._cpu_session is None:
                self._cpu_session = ort.InferenceSession(
                    self._model_info.model_path,
                    InferSessionPool._get_session_options(),
                    providers=["CPUExecutionProvider"],
                )
                self._reserve_session_mem("cpu", -1)
                
            return self._cpu_session

        key = self._session_key(device_key, user_providers)
        if key in InferSessionPool._shared_sessions and key in self._session_reserved:
            return InferSessionPool._shared_sessions[key]

        with InferSessionPool._class_session_creation_lock:
            if key not in InferSessionPool._shared_sessions:
                if user_providers:
                    providers = [*user_providers, ("CPUExecutionProvider", {})]
                else:
                    provider = task.onnx_provider
                    providers = [provider, ("CPUExecutionProvider", {})]
                session = ort.InferenceSession(
                    self._model_info.model_path,
                    InferSessionPool._get_session_options(),
                    providers=providers,
                )
                InferSessionPool._shared_sessions[key] = session

            if key not in self._session_reserved:
                self._reserve_session_mem(key, task.gpu_index)

            return InferSessionPool._shared_sessions[key]

    def _reserve_session_mem(self, session_key: str, gpu_index: int) -> None:
        with self._sys_info_lock:
            if gpu_index == -1:
                self._sys_info.cpu_mem_free -= self._model_mem
            else:
                self._sys_info.gpu_infos[gpu_index].mem_free -= self._model_mem

            self._session_reserved[session_key] = gpu_index

    def _release_all_session_resources(self) -> None:
        with self._sys_info_lock:
            for gpu_index in self._session_reserved.values():
                if gpu_index == -1:
                    self._sys_info.cpu_mem_free += self._model_mem
                else:
                    self._sys_info.gpu_infos[gpu_index].mem_free += self._model_mem
            self._session_reserved.clear()

    def _postprocess_after_task_done(self) -> None:
        if self._shutdown or not self._delayed_tasks:
            return
        
        remaining = []
        for delayed_task in self._delayed_tasks:
            if delayed_task.future.cancelled():
                self._tasks.pop(delayed_task.id, None)
                continue

            if not self._try_assign_device(delayed_task):
                remaining.append(delayed_task)
                continue

            if not self._try_acquire_slot(delayed_task):
                delayed_task.device = None
                delayed_task.gpu_index = -1
                delayed_task.dml_id = -1
                remaining.append(delayed_task)
                continue

            self._dispatch_infer(delayed_task)

        self._delayed_tasks = remaining

    def _take_resource(self, task: Task) -> None:
        with self._sys_info_lock:
            res = task.effective_res
            task.estimated_need_cpu_mem = res.cpu_mem
            self._sys_info.cpu_cores_free -= res.cpu_cores
            self._sys_info.cpu_mem_free -= res.cpu_mem
            if task.gpu_index != -1:
                self._sys_info.gpu_infos[task.gpu_index].n_cores_free -= res.gpu_cores
                self._sys_info.gpu_infos[task.gpu_index].mem_free -= res.gpu_mem

    def _release_resource(self, task: Task) -> None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free += res.cpu_cores
            self._sys_info.cpu_mem_free += task.estimated_need_cpu_mem
            if task.gpu_index != -1:
                self._sys_info.gpu_infos[task.gpu_index].n_cores_free += res.gpu_cores
                self._sys_info.gpu_infos[task.gpu_index].mem_free += res.gpu_mem

    def _choose_task_worker(self, task, res):
        return None

    def _estimate_cpu_cores_needed(self, res: Resource) -> float:
        return res.cpu_cores

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if self._shutdown:
                return
            
            self._shutdown = True
            if cancel_futures:
                for task in self._tasks.values():
                    task.future.cancel()

            if self._in_flight_count == 0:
                self._all_done.set()

        if wait:
            self._all_done.wait()

        self._release_all_session_resources()
