from __future__ import annotations

import os
import threading
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from ..pool import Pool
from .infer_session_worker import InferSessionWorker

if TYPE_CHECKING:
    import onnxruntime as ort
    from queue import SimpleQueue

    from ..futures import Future
    from ..gpuinfo import GPUInfoSnapshot
    from ..resource import Resource
    from ..worker import Worker
    from .infer_session import InferSession
    from .infer_session_task import InferSessionTask
    from .model_info import ModelInfo
    from .infer_model import InferModel


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
        from queue import SimpleQueue

        if max_workers <= 0:
            import os
            max_workers = os.cpu_count()

        self._thread_name_prefix: str = thread_name_prefix
        self._task_queue: SimpleQueue = SimpleQueue()
        self._session_options: Optional[ort.SessionOptions] = session_options
        self._sessions: Dict[Tuple[str, str, str], InferSession] = {}
        self.print_info: bool = False

        Pool.__init__(
            self, max_workers=max_workers,
            initializer=None, initargs=(), initkwargs=None,
            result_queue_cls=None,
            worker_cls=InferSessionWorker,
            chunk_timeout=chunk_timeout
        )

    @staticmethod
    def model_info(model_path:str)->ModelInfo:
        model_path = os.path.abspath(model_path).replace("\\", "/")
        if model_path not in InferSessionPool._model_infos:
            from .model_info import ModelInfo
            InferSessionPool._model_infos[model_path] = ModelInfo(model_path)

        return InferSessionPool._model_infos[model_path]

    @classmethod
    def config(
        cls, pool_name: str="default", *,
        max_workers: int = 0,
        thread_name_prefix: str = "InferSessionPool.worker:",
        session_options: Optional[ort.SessionOptions] = None,
        chunk_timeout: float = 0.1
    ) -> None:
        Pool.config(
            cls, pool_name,
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
            session_options=session_options,
            chunk_timeout=chunk_timeout
        )

    @classmethod
    def task(cls, *args, **kwargs):
        raise NotImplementedError("InferSessionPool.task is not implemented")

    @staticmethod
    def model(
        model_path: str,
        pool_name: str = "default",
        output_names: Optional[List[str]] = None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        check_args: bool = True,
        providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
        run_options: Optional[ort.RunOptions] = None,
        use_io_binding: bool = False,
        copy_outputs_to_cpu: bool = True,
        chunksize: int = 1
    )->InferModel:
        from .infer_model import InferModel
        return InferModel(
            model_path, pool_name,
            output_names,
            cpu_mode_res,
            gpu_mode_res,
            check_args,
            providers,
            run_options,
            use_io_binding,
            copy_outputs_to_cpu,
            chunksize
        )

    @property
    def session_options(self) -> ort.SessionOptions:
        if self._session_options is None:
            import onnxruntime as ort
            self._session_options = ort.SessionOptions()
            self._session_options.enable_cpu_mem_arena = True
            self._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return self._session_options

    def _get_task_queue(self)->SimpleQueue:
        return self._task_queue

    @staticmethod
    def _provider_key(model_path:str, provider: Tuple[str, Dict[str, Any]])->Tuple[str, str, str]:
        import json

        model_path = os.path.abspath(model_path).replace("\\", "/")
        return (
            model_path,
            provider[0],
            json.dumps(provider[1], sort_keys=True, separators=(', ', ':'))
        )

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
                if gpu_info.device.select_provider(providers):
                    return

        raise ValueError(f"all providers are not supported: {providers}")

    def submit(
        self, model_path: str,
        args: Optional[Tuple[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        output_names: Optional[List[str]] = None,
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
        merged_kwargs, has_ortvalue = model_info._merge_args(args, kwargs, check_args)

        if check_args:
            model_info._check_outputs(output_names)
            self._check_providers(providers)

        if cpu_mode_res is None:
            cpu_mode_res = Resource(cpu_cores_in_python=1)

        if gpu_mode_res is None:
            gpu_mode_res = cpu_mode_res

        if not copy_outputs_to_cpu or has_ortvalue:
            use_io_binding = True

        if chunksize > 1 and not model_info.support_batch_input:
            chunksize = 1

            import warnings
            warnings.warn(f"model {model_info.model_name} not support batch input, reset chunksize to 1", stacklevel=1)

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
        output_names: Optional[List[str]] = None,
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

        self.flush()

        return Pool._result_iterator(futures, end_time)

    def _put_in_chunk(self, task: InferSessionTask) -> Future:
        import threading
        from queue import Queue

        from .infer_chunk_task import InferChunkTask

        if self._flush_chunk_thread is None:
            self._flush_chunk_queue = Queue()
            self._flush_chunk_thread = threading.Thread(target=self._flushing_chunk, daemon=True, name="flushing_chunk")
            self._flush_chunk_thread.start()

        key = InferChunkTask.get_key(task.model_info.model_path, task.use_io_binding, task.user_providers, task.run_options, task.chunksize)
        if key not in self._chunk_tasks:
            chunk_task: InferChunkTask = InferChunkTask(
                pool=self,
                model_info=task.model_info,
                use_io_binding=task.use_io_binding,
                user_providers=task.user_providers,
                run_options=task.run_options,
                chunksize=task.chunksize,
                key=key
            )
            self._chunk_tasks[key] = chunk_task
            self._flush_chunk_queue.put(chunk_task)
        else:
            chunk_task: InferChunkTask = self._chunk_tasks[key]

        chunk_task.add_task(task)

        return task.future

    def _try_assign_task(self, task: InferSessionTask) -> bool:
        if self._workers_working_count >= self._max_workers:
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

            success = False
            devices, cpu_to_evict, gpu_to_evicts = self._choose_task_device(task, mode)
            gpu_to_evict = None
            for device, to_evict in zip_longest(devices, gpu_to_evicts):
                task.device = device

                if self._choose_task_session(task) is None:
                    task.device = None
                    continue

                if self._choose_task_worker(task, task.effective_res) is None:
                    task.device = None
                    continue

                gpu_to_evict = to_evict
                success = True
                break

            if success:
                for session in cpu_to_evict:
                    session.stop()

                if gpu_to_evict:
                    for session in gpu_to_evict:
                        session.stop()

                return True

        return False

    def _reclaim_cpu_memory(self, task: InferSessionTask, res: Resource) -> List[InferSession]:
        idle_sessions = [session for session in self._sessions.values() if (session._session is not None and session.running_count == 0)]
        if not idle_sessions or (len(idle_sessions) == 1 and idle_sessions[0].model_path == task.model_info.model_path):
            return []

        idle_sessions.sort(key=lambda x: x.last_use_time)
        to_evict = []
        freed_cpu_mem = 0
        for session in idle_sessions:
            if res.cpu_mem <= self._sys_info.cpu_mem_free + freed_cpu_mem:
                break

            to_evict.append(session)
            freed_cpu_mem += session.estimated_cpu_mem

        if res.cpu_mem > self._sys_info.cpu_mem_free + freed_cpu_mem:
            return []

        return to_evict

    def _reclaim_gpu_memory(self, task: InferSessionTask, res: Resource, gpu: GPUInfoSnapshot, reclaim_items: List[InferSession]) -> List[InferSession]:
        idle_sessions = [session for session in self._sessions.values() if (session._session is not None and session.running_count == 0 and session not in reclaim_items and session.device == gpu.device)]
        if not idle_sessions or (len(idle_sessions) == 1 and idle_sessions[0].model_path == task.model_info.model_path):
            return []

        idle_sessions.sort(key=lambda x: x.last_use_time)
        to_evict = []
        freed_gpu_mem = 0
        for session in idle_sessions:
            if res.gpu_mem <= gpu.mem_free + freed_gpu_mem:
                break

            to_evict.append(session)
            freed_gpu_mem += session.estimated_gpu_mem

        if res.gpu_mem > gpu.mem_free + freed_gpu_mem:
            return []

        return to_evict

    def _choose_task_session(self, task: InferSessionTask) -> Optional[InferSession]:
        provider: Tuple[str, Dict[str, Any]] = task.provider
        if not self._can_use_provider(provider):
            task.session = None
            return None

        model_path: str = task.model_info.model_path
        key = self._provider_key(model_path, provider)
        if key not in self._sessions:
            from .infer_session import InferSession
            session = InferSession(model_path, self.session_options, providers=[provider], enable_fallback=False, device=task.device)
            self._sessions[key] = session

        task.session = self._sessions[key]
        return task.session

    def _choose_task_worker(self, task: InferSessionTask, res: Resource) -> Optional[Worker]:
        if self._workers_working_count < len(self._workers):
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
        self._register_gpu_candidate(task)
        worker: InferSessionWorker = task.worker
        worker.is_working = True
        worker.add_task(task)

    def _take_resource(self, task: InferSessionTask) -> None:
        with self._sys_info_lock:
            res = task.effective_res
            task.estimated_need_cpu_mem = res.cpu_mem
            self._sys_info.cpu_cores_free -= res.cpu_cores
            self._sys_info.cpu_mem_free -= res.cpu_mem
            gpu_index = task.device.gpu_index
            if gpu_index != -1:
                self._sys_info.gpu_infos[gpu_index].n_cores_free -= res.gpu_cores
                self._sys_info.gpu_infos[gpu_index].mem_free -= res.gpu_mem

    def _release_resource(self, task: InferSessionTask) -> None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free += res.cpu_cores
            self._sys_info.cpu_mem_free += task.estimated_need_cpu_mem
            gpu_index = task.device.gpu_index
            if gpu_index != -1:
                self._sys_info.gpu_infos[gpu_index].n_cores_free += res.gpu_cores
                self._sys_info.gpu_infos[gpu_index].mem_free += res.gpu_mem

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        for session in self._sessions.values():
            session.stop()
        self._sessions.clear()
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
