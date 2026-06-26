from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from ..pool import Pool
from .infer_session_worker import InferSessionWorker

if TYPE_CHECKING:
    from ..futures import Future

    import onnxruntime as ort

    from ..resource import Resource
    from ..worker import Worker
    from .infer_session_task import InferSessionTask
    from .model_info import ModelInfo
    from .session_info import SessionInfo


class InferSessionPool(Pool):

    _not_thread_safe_providers_lock: threading.Lock = threading.Lock()
    _not_thread_safe_providers: Dict[str, Set[int]] = {
        "DmlExecutionProvider": set(),
    }

    _model_infos: Dict[str, ModelInfo] = {}

    PROVIDER_MEMORY_MULTIPLIERS: Dict[str, Tuple[int, int]] = {
        "CPUExecutionProvider": (4, 0),
        "CUDAExecutionProvider": (3, 7),
        "TensorrtExecutionProvider": (356, 45),
        "DmlExecutionProvider": (1, 2),
        "OpenVINOExecutionProvider": (2, 0),
        "CoreMLExecutionProvider": (2, 1),
        "ROCMExecutionProvider": (1, 2),
        "MIGraphXExecutionProvider": (1, 2),
        "ArmNNExecutionProvider": (3, 0),
        "XnnpackExecutionProvider": (3, 0),
        "NnapiExecutionProvider": (2, 0),
        "AclExecutionProvider": (2, 1),
    }

    def __init__(
        self, max_workers: int = 0,
        thread_name_prefix: str = "InferSessionPool.worker:",
        session_options: Optional[ort.SessionOptions] = None,
        chunk_timeout: float = 0.1
    ):
        if max_workers <= 0:
            import os
            max_workers = os.cpu_count()
            
        Pool.__init__(
            self, max_workers=max_workers,
            initializer=None, initargs=(), initkwargs=None,
            result_queue_cls=None,
            worker_cls=InferSessionWorker,
            chunk_timeout=chunk_timeout
        )
        self._thread_name_prefix: str = thread_name_prefix
        self._session_options: Optional[ort.SessionOptions] = session_options
        self._sessions: Dict[Tuple[str, str, str], SessionInfo] = {}
        self.print_info: bool = False

    @staticmethod
    def model_info(model_path:str)->ModelInfo:
        model_path = os.path.abspath(model_path).replace("\\", "/")
        if model_path not in InferSessionPool._model_infos:
            from .model_info import ModelInfo
            InferSessionPool._model_infos[model_path] = ModelInfo(model_path)

        return InferSessionPool._model_infos[model_path]

    @property
    def session_options(self) -> ort.SessionOptions:
        if self._session_options is None:
            import onnxruntime as ort
            self._session_options = ort.SessionOptions()
            self._session_options.enable_cpu_mem_arena = True
            self._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return self._session_options

    @staticmethod
    def _provider_key(model_path:str, provider: Tuple[str, Dict[str, Any]])->Tuple[str, str, str]:
        import json

        model_path = os.path.abspath(model_path).replace("\\", "/")
        return (
            model_path,
            provider[0],
            json.dumps(provider[1], sort_keys=True, separators=(', ', ':'))
        )

    def _session_info(self, model_path:str, provider: Tuple[str, Dict[str, Any]])->SessionInfo:
        key = self._provider_key(model_path, provider)
        if key not in self._sessions:
            from .session_info import SessionInfo
            session_info = SessionInfo(model_path, self.session_options, providers=[provider], enable_fallback=False)
            self._sessions[key] = session_info

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
        self, model_path: str,
        args: Optional[Tuple[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        output_names: Optional[Union[List[str], None]] = None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        check_args: bool = True,
        providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
        run_options: Optional[ort.RunOptions] = None,
        use_io_binding: bool = False,
        copy_outputs_to_cpu: bool = True,
        chunksize: int = 1
    ) -> Future:
        if self._shutdown:
            raise RuntimeError("cannot submit after shutdown")
        
        from ..resource import Resource
        from .infer_session_task import InferSessionTask
        
        model_info: ModelInfo = self.model_info(model_path)
        merged_kwargs, has_ortvalue = model_info.merge_args(args, kwargs, check_args)
        
        if check_args:
            model_info.check_outputs(output_names)
            self._check_providers(providers)

        if cpu_mode_res is None:
            cpu_mode_res = Resource(cpu_cores_in_python=1)

        if gpu_mode_res is None:
            gpu_mode_res = cpu_mode_res

        if not use_io_binding:
            if not copy_outputs_to_cpu or has_ortvalue:
                use_io_binding = True

        if not model_info.support_batch_input:
            chunksize = 1

        task = InferSessionTask(
            infer_session_pool=self, model_info=model_info, kwargs=merged_kwargs,
            cpu_mode_res=cpu_mode_res, gpu_mode_res=gpu_mode_res, check_args=check_args,
            output_names=output_names, user_providers=providers, run_options=run_options,
            use_io_binding=use_io_binding, copy_outputs_to_cpu=copy_outputs_to_cpu,
            chunksize=chunksize
        )
        if check_args:
            self._validate_resource_feasibility(task)

        self._submit_or_chunk(task)
        return task.future

    def map(
        self,
        model_path: str,
        *iterables: Iterable[Any],
        output_names: Optional[Union[List[str], None]] = None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        check_args: bool = True,
        providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
        run_options: Optional[ort.RunOptions] = None,
        use_io_binding: bool = False,
        copy_outputs_to_cpu: bool = True,
        timeout: Optional[Union[float, int]] = None,
        chunksize: int = 1
    ) -> Iterable[Any]:
        import time

        if not iterables:
            raise ValueError("at least one iterable is required")

        if chunksize < 1:
            raise ValueError("chunksize must be >= 1")

        end_time = None
        if timeout is not None:
            end_time = timeout + time.monotonic()

        model_info: ModelInfo = self.model_info(model_path)
        input_names = list(model_info.inputs.keys())

        if len(iterables) != len(input_names):
            raise ValueError(
                f"model has {len(input_names)} inputs ({input_names}), "
                f"but {len(iterables)} iterables were provided"
            )

        futures: List[Future] = []
        for item in zip(*iterables):
            kwargs = {name: item[i] for i, name in enumerate(input_names)}
            future = self.submit(
                model_path,
                kwargs=kwargs,
                output_names=output_names,
                cpu_mode_res=cpu_mode_res,
                gpu_mode_res=gpu_mode_res,
                check_args=check_args,
                providers=providers,
                run_options=run_options,
                use_io_binding=use_io_binding,
                copy_outputs_to_cpu=copy_outputs_to_cpu,
                chunksize=chunksize
            )
            futures.append(future)

        return Pool._result_iterator(futures, end_time)

    def _put_in_chunk(self, task: InferSessionTask) -> Future:
        from .infer_chunk_task import InferChunkTask
        import threading
        from queue import Queue

        if self._flush_chunk_thread is None:
            self._flush_chunk_queue = Queue()
            self._flush_chunk_thread = threading.Thread(target=self._flushing_chunk, daemon=True, name="flushing_chunk")
            self._flush_chunk_thread.start()

        key = InferChunkTask.get_key(task.model_info.model_path, task.use_io_binding, task.user_providers, task.run_options, task.chunksize)
        if key not in self._chunk_tasks:
            self._chunk_tasks[key] = InferChunkTask(
                pool=self,
                model_info=task.model_info,
                use_io_binding=task.use_io_binding,
                user_providers=task.user_providers,
                run_options=task.run_options,
                chunksize=task.chunksize,
                key=key
            )

        chunk_task: InferChunkTask = self._chunk_tasks[key]
        chunk_task.add_task(task)

        if self._flush_chunk_queue is not None and not self._shutdown:
            self._flush_chunk_queue.put(True)

        return task.future

    def _try_assign_task(self, task: InferSessionTask) -> bool:
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

            devices, reclaim_items = self._choose_task_device(task, mode)
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
                else:
                    break

            if task.device is not None and task.session_info is not None and task.worker is not None:
                self._reclaim_session_items(reclaim_items)
                return True

        return False

    def _reclaim_cpu_memory(self, task: InferSessionTask, res: Resource, mode: str, reclaim_items: List) -> bool:
        idle = [(k, s) for k, s in self._sessions.items() if s._running_count == 0]
        if not idle:
            return False
        idle.sort(key=lambda x: x[1]._last_use_time)
        freed = 0
        for key, sess in idle:
            reclaim_items.append((key, sess))
            freed += sess.estimated_cpu_mem
            if res.cpu_mem <= self._sys_info.cpu_mem_free + freed:
                break
        if res.cpu_mem > self._sys_info.cpu_mem_free + freed:
            return False
        return True

    def _reclaim_gpu_memory(self, task: InferSessionTask, res: Resource, mode: str) -> bool:
        idle = [(k, s) for k, s in self._sessions.items() if s._running_count == 0]
        if not idle:
            return False
        idle.sort(key=lambda x: x[1]._last_use_time)

        to_evict = []
        with self._sys_info_lock:
            freed_by_gpu = {}
            for _, s in idle:
                if s._gpu_index >= 0:
                    freed_by_gpu[s._gpu_index] = freed_by_gpu.get(s._gpu_index, 0) + s.estimated_gpu_mem
            gpus = task.filter_gpu_infos(self._sys_info.gpu_infos)
            for gpu in gpus:
                extra = freed_by_gpu.get(gpu.index, 0)
                if gpu.mem_free + extra >= res.gpu_mem and gpu.n_cores_free >= res.gpu_cores:
                    to_evict = idle[:]
                    break

        if to_evict:
            for key, sess in to_evict:
                if key in self._sessions:
                    del self._sessions[key]
                    sess.stop()
            return True
        return False

    def _reclaim_session_items(self, items: List) -> None:
        for item in items:
            if isinstance(item, tuple):
                key, sess = item
                if key in self._sessions:
                    del self._sessions[key]
                    sess.stop()

    def _choose_task_session(self, task: InferSessionTask) -> Optional[SessionInfo]:
        provider: Tuple[str, Dict[str, Any]] = task.provider
        if not self._can_use_provider(provider):
            task.session_info = None
            return None

        task.session_info = self._session_info(task.model_info.model_path, provider)
        return task.session_info

    def _choose_task_worker(self, task: InferSessionTask, res: Resource) -> Optional[Worker]:
        for worker in self._workers:
            if worker.is_working:
                continue

            if (worker.executor is not None) == task.use_io_binding:
                task.worker = worker
                return worker

        for worker in self._workers:
            if worker.is_working:
                continue

            task.worker = worker
            return worker

        if len(self._workers) < self._max_workers:
            task.worker = self._add_worker()
        else:
            task.worker = None

        return task.worker

    def _put_task(self, task: InferSessionTask) -> None:
        self._take_resource(task)
        worker: InferSessionWorker = task.worker
        worker.is_working = True
        worker.add_task(task)

    def _take_resource(self, task: InferSessionTask) -> None:
        with self._sys_info_lock:
            res = task.effective_res
            task.estimated_need_cpu_mem = res.cpu_mem
            self._sys_info.cpu_cores_free -= res.cpu_cores
            self._sys_info.cpu_mem_free -= res.cpu_mem
            if task.gpu_index != -1:
                self._sys_info.gpu_infos[task.gpu_index].n_cores_free -= res.gpu_cores
                self._sys_info.gpu_infos[task.gpu_index].mem_free -= res.gpu_mem

    def _release_resource(self, task: InferSessionTask) -> None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free += res.cpu_cores
            self._sys_info.cpu_mem_free += task.estimated_need_cpu_mem
            if task.gpu_index != -1:
                self._sys_info.gpu_infos[task.gpu_index].n_cores_free += res.gpu_cores
                self._sys_info.gpu_infos[task.gpu_index].mem_free += res.gpu_mem

    def _estimate_cpu_cores_needed(self, res: Resource) -> float:
        return res.cpu_cores

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        for session in self._sessions.values():
            session.stop()
        self._sessions.clear()
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
