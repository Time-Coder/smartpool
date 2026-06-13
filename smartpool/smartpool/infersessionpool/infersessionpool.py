from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from ..pool import Pool
from .infersessionworker import InferSessionWorker

if TYPE_CHECKING:
    from concurrent.futures import Future

    import onnxruntime as ort

    from ..resource import Resource
    from ..task import Task
    from .model_info import ModelInfo


class InferSessionPool(Pool):

    _not_thread_safe_providers: Dict[str, Set[int]] = {
        "DmlExecutionProvider": set(),
    }

    _model_infos: Dict[str, ModelInfo] = {}

    def __init__(self, model_path: str, max_workers: int = 0):
        from .model_info import ModelInfo
        import threading

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
        self._workers_dict: Dict[str, InferSessionWorker] = {}
        self._running_count_lock: threading.Lock = threading.Lock()
        self._running_count: int = 0
        self.print_info: bool = False

    @staticmethod
    def _provider_key(provider: Tuple[str, Dict[str, Any]])->str:
        import json

        return "(" + provider[0] + ", " + json.dumps(provider[1], sort_keys=True, separators=(', ', ':')) + ")"

    def _add_worker(self, provider: Tuple[str, Dict[str, Any]], gpu_index: int)->Optional[InferSessionWorker]:
        worker = InferSessionWorker(self, provider, gpu_index)
        self._workers.append(worker)
        key = self._provider_key(provider)
        self._workers_dict[key] = worker
        return worker

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

        validated_kwargs = self._model_info.check_args(args, kwargs)
        self._model_info.check_outputs(output_names)

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
            if not self._try_assign_task(task):
                self._delayed_tasks.append(task)

        return task.future

    def _try_assign_task(self, task: Task) -> bool:
        if self._running_count >= self._max_workers:
            return False

        gpu_res = task.gpu_mode_res
        modes = []
        if gpu_res.gpu_cores > 0 or gpu_res.gpu_mem > 0:
            modes.append("gpu")
        modes.append("cpu")

        for mode in modes:
            device, _ = self._choose_task_device(task, mode, kill_workers=False)
            if device is None:
                continue

            if not self._choose_task_worker(task):
                task.device = None
                task.gpu_index = -1
                task.dml_id = -1
                continue

            self._put_task(task)
            return True

        return False

    def _choose_task_worker(self, task: Task) -> Optional[InferSessionWorker]:        
        provider: Tuple[str, Dict[str, Any]] = task.onnx_provider
        provider_key: str = self._provider_key(provider)
        is_thread_safe: bool = (provider[0] in InferSessionPool._thread_safe_providers)
        if provider_key in self._workers_dict:
            worker = self._workers_dict[provider_key]
            if is_thread_safe or worker.running_count == 0:
                task.worker = worker
                return worker
            else:
                task.worker = None
                return None
        else:
            worker = self._add_worker(provider, task.gpu_index)
            task.worker = worker
            return worker

    def _put_task(self, task: Task) -> None:
        self._take_resource(task)
        worker:InferSessionWorker = task.worker
        worker.is_working = True
        worker.add_task(task)

    def _postprocess_after_task_done(self) -> None:
        if self._shutdown or not self._delayed_tasks:
            return
        
        remaining = []
        for delayed_task in self._delayed_tasks:
            if delayed_task.future.cancelled():
                self._tasks.pop(delayed_task.id, None)
                continue

            if not self._try_assign_task(delayed_task):
                remaining.append(delayed_task)

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

    def _estimate_cpu_cores_needed(self, res: Resource) -> float:
        return res.cpu_cores
