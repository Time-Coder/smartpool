from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Any, Dict
from .task import Task
from .resource import Resource

if TYPE_CHECKING:
    from .pool import Pool
    from .futures import Future


class ChunkTask(Task):

    def __init__(self, pool: Pool, chunksize: int):
        self._sub_tasks: List[Task] = []
        self._args_list: List[Tuple[Any, ...]] = []
        self._kwargs_list: List[Dict[str, Any]] = []
        self.last_add_time: float = 0.0
        Task.__init__(
            self, pool=pool, func=self._process_batch, args=None, kwargs=None,
            cpu_mode_res=Resource(), gpu_mode_res=Resource(), check_args=False, chunksize=chunksize,
            use_torch=False, device_changeable=False, calculate_module_deps=False
        )
        self.future.add_done_callback(self._fan_out_batch_results)
        
    def add_task(self, task: Task)->None:
        import time

        self.last_add_time = time.time()
        if not self.args:
            self.args = (task.func, self._args_list, self._kwargs_list)

        self._sub_tasks.append(task)
        self._args_list.append(task.args)
        self._kwargs_list.append(task.kwargs)
        if task.use_torch:
            self.use_torch = True

        if task.check_args:
            self.check_args = True

        if task.device_changeable:
            self.device_changeable = True

        if task.module_deps:
            self.module_deps.update(task.module_deps)

        self._update_res(self.cpu_mode_res, task.cpu_mode_res)
        self._update_res(self.gpu_mode_res, task.gpu_mode_res)

        try:
            self.pool._validate_resource_feasibility(self)
        except Exception as e:
            for sub_task in self._sub_tasks:
                sub_task.future.set_exception(e)

            if self.id in self.pool._chunk_tasks:
                del self.pool._chunk_tasks[self.id]

            return

        if len(self._sub_tasks) >= self.chunksize:
            if self.id in self.pool._chunk_tasks:
                del self.pool._chunk_tasks[self.id]

            self.submit()

    def submit(self):
        self.pool._tasks[self.id] = self
        self.pool._submit(self)

    @staticmethod
    def _process_batch(func, args_list, kwargs_list):
        results = []
        for args, kwargs in zip(args_list, kwargs_list):
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = e
                success = False

            results.append((success, result))

        return results
    
    def _fan_out_batch_results(self, future: Future) -> None:
        if future.cancelled():
            for task in self._sub_tasks:
                task.future.cancel()

            return

        try:
            results = future.result()
        except Exception as exc:
            for task in self._sub_tasks:
                task.future.set_exception(exc)
                
            return

        for (success, result), task in zip(results, self._sub_tasks):
            if success:
                task.future.set_result(result)
            else:
                task.future.set_exception(result)

    @staticmethod
    def _update_res(src_res: Resource, target_res: Resource)->None:
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