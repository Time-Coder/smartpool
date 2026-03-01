from __future__ import annotations
from ..pool import Pool
from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional, Callable

if TYPE_CHECKING:
    from ..task import Task
    from .threadworker import ThreadWorker


class ThreadPool(Pool):

    def __init__(
        self, max_workers:int=0, thread_name_prefix:str="ThreadPool.worker:",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
        use_torch:bool=False
    ):
        from queue import SimpleQueue

        Pool.__init__(
            self, max_workers=max_workers,

            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,

            result_queue_cls=SimpleQueue,

            max_tasks_per_child=max_tasks_per_child,
            use_torch=use_torch,
            need_module_deps=False
        )

        self._thread_name_prefix:str = thread_name_prefix
        self.__max_used_cpu_cores = None
        self.__max_used_gpu_cores = {}

    def _max_used_cpu_cores(self)->int:
        if self.__max_used_cpu_cores is not None:
            return self.__max_used_cpu_cores

        max_used_cores = 0
        for task in self._tasks.values():
            if task.worker is None or not task.worker.is_working:
                continue

            if task.need_cpu_cores > max_used_cores:
                max_used_cores = task.need_cpu_cores

        self.__max_used_cpu_cores = max_used_cores
        return max_used_cores
    
    def _max_used_gpu_cores(self, gpu_id:int)->int:
        if gpu_id in self.__max_used_gpu_cores:
            return self.__max_used_gpu_cores[gpu_id]

        max_used_cores = 0
        for task in self._tasks.values():
            if task.worker is None or not task.worker.is_working or task.gpu_id != gpu_id:
                continue

            if task.need_gpu_cores > max_used_cores:
                max_used_cores = task.need_gpu_cores

        self.__max_used_gpu_cores[gpu_id] = max_used_cores
        return max_used_cores

    def _has_gil(self)->bool:
        from ..utils import has_gil
        return has_gil()

    def _take_resource(self, task:Task)->None:
        with self._sys_info_lock:
            if not self._has_gil():
                self._sys_info.cpu_cores_free -= task.need_cpu_cores
            else:
                max_used_cpu_cores = self._max_used_cpu_cores()
                if task.need_cpu_cores > max_used_cpu_cores:
                    self.__max_used_cpu_cores = task.need_cpu_cores
                    self._sys_info.cpu_cores_free -= (task.need_cpu_cores - max_used_cpu_cores)

            self._sys_info.cpu_mem_free -= task.need_cpu_mem
            task_gpu_id:int = task.gpu_id
            if task_gpu_id != -1:
                if not self._has_gil():
                    self._sys_info.gpu_infos[task_gpu_id].n_cores_free -= task.need_gpu_cores
                else:
                    max_used_gpu_cores = self._max_used_gpu_cores(task_gpu_id)
                    if task.need_gpu_cores > max_used_gpu_cores:
                        self.__max_used_gpu_cores[task_gpu_id] = task.need_gpu_cores
                        self._sys_info.gpu_infos[task_gpu_id].n_cores_free -= (task.need_gpu_cores - max_used_gpu_cores)

                self._sys_info.gpu_infos[task_gpu_id].mem_free -= task.need_gpu_mem

    def _release_resource(self, task:Task)->None:
        with self._sys_info_lock:
            if not self._has_gil():
                self._sys_info.cpu_cores_free += task.need_cpu_cores
            else:
                max_used_cpu_cores = self._max_used_cpu_cores()
                if task.need_cpu_cores >= max_used_cpu_cores:
                    self.__max_used_cpu_cores = None
                    max_used_cpu_cores = self._max_used_cpu_cores()
                    self._sys_info.cpu_cores_free += (task.need_cpu_cores - max_used_cpu_cores)
                
            self._sys_info.cpu_mem_free += task.need_cpu_mem
            task_gpu_id:int = task.gpu_id
            if task_gpu_id != -1:
                if not self._has_gil():
                    self._sys_info.gpu_infos[task_gpu_id].n_cores_free += task.need_gpu_cores
                else:
                    max_used_gpu_cores = self._max_used_gpu_cores(task_gpu_id)
                    if task.need_gpu_cores >= max_used_gpu_cores:
                        self.__max_used_gpu_cores[task_gpu_id] = None
                        max_used_gpu_cores = self._max_used_gpu_cores(task_gpu_id)
                        self._sys_info.gpu_infos[task_gpu_id].n_cores_free += (task.need_gpu_cores - max_used_gpu_cores)

                self._sys_info.gpu_infos[task_gpu_id].mem_free += task.need_gpu_mem

    def _estimate_need_gpu_cores(self, task:Task, gpu_id:int)->int:
        if self._has_gil():
            return max(0, task.need_gpu_cores - self._max_used_gpu_cores(gpu_id))
        else:
            return task.need_gpu_cores

    def _estimate_need_cpu_cores(self, task:Task)->int:
        if self._has_gil():
            return max(0, task.need_cpu_cores - self._max_used_cpu_cores())
        else:
            return task.need_cpu_cores
        
    def _estimate_need_cpu_mem(self, task:Task)->int:
        return task.need_cpu_mem

    def _put_task(self, task:Task)->None:
        self._take_resource(task)
        worker:ThreadWorker = task.worker
        worker.is_working = True
        task.future.set_running_or_notify_cancel()
        worker.task_queue.put(task)

    def _add_worker(self)->ThreadWorker:
        from .threadworker import ThreadWorker

        worker = ThreadWorker(
            len(self._workers), self._thread_name_prefix,
            self._result_queue,
            initializer=self._initializer,
            initargs=self._initargs,
            initkwargs=self._initkwargs
        )
        self._workers.append(worker)
        return worker
