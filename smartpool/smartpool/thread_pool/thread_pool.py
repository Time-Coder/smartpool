from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from ..pool import Pool

if TYPE_CHECKING:
    from ..futures import Future
    from ..resource import Resource
    from ..task import Task
    from .thread_worker import ThreadWorker
    from queue import SimpleQueue


class ThreadPool(Pool):

    def __init__(
        self, max_workers:int=0,
        thread_name_prefix:str="ThreadPool.worker:",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        use_torch:bool=False
    ):
        from .thread_worker import ThreadWorker
        from queue import SimpleQueue

        if max_workers <= 0:
            import os
            max_workers = min(32, (os.cpu_count() or 1) + 4)

        self._thread_name_prefix:str = thread_name_prefix
        self.__max_used_cpu_cores_in_python = None
        self._task_queue: SimpleQueue = SimpleQueue()

        Pool.__init__(
            self, max_workers=max_workers,

            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,

            result_queue_cls=None,

            use_torch=use_torch,
            worker_cls=ThreadWorker,
            need_module_deps=False
        )

    def submit(
        self, func:Callable[..., Any],
        args:Optional[Tuple[Any, ...]]=None,
        kwargs:Optional[Dict[str, Any]]=None,
        cpu_mode_res:Optional[Resource]=None,
        gpu_mode_res:Optional[Resource]=None,
        check_args:bool=True,
        use_torch:Optional[bool]=None,
        device_changeable:bool=False
    )->Future:
        return super().submit(
            func, args=args, kwargs=kwargs,
            cpu_mode_res=cpu_mode_res, gpu_mode_res=gpu_mode_res,
            check_args=check_args, chunksize=1,
            use_torch=use_torch, device_changeable=device_changeable
        )

    def map(
        self, func:Callable[..., Any],
        *iterable:Iterable[Any],
        cpu_mode_res:Optional[Union[Resource, Iterable[Resource]]] = None,
        gpu_mode_res:Optional[Union[Resource, Iterable[Resource]]] = None,
        timeout:Optional[Union[float, int]]=None
    )->Iterable[Any]:
        import itertools
        import time
        from collections.abc import Iterable as IterableType

        if cpu_mode_res is None:
            from ..resource import Resource as _Res
            cpu_mode_res = _Res()

        if not isinstance(cpu_mode_res, IterableType):
            cpu_mode_res = itertools.repeat(cpu_mode_res)

        if gpu_mode_res is None:
            gpu_mode_res = cpu_mode_res
        elif not isinstance(gpu_mode_res, IterableType):
            gpu_mode_res = itertools.repeat(gpu_mode_res)

        end_time = None
        if timeout is not None:
            end_time = timeout + time.monotonic()

        futures:List[Future] = []
        for args, cpu_res, gpu_res in zip(zip(*iterable), cpu_mode_res, gpu_mode_res):
            future = super().submit(
                func, args=args,
                cpu_mode_res=cpu_res,
                gpu_mode_res=gpu_res,
                chunksize=1
            )
            futures.append(future)

        return Pool._result_iterator(futures, end_time)

    def _max_used_cpu_cores_in_python(self)->int:
        if self.__max_used_cpu_cores_in_python is not None:
            return self.__max_used_cpu_cores_in_python

        max_in = 0
        for task in self._tasks.values():
            if task.worker is None or not task.worker.is_working:
                continue

            task_in = task.effective_res.cpu_cores_in_python
            if task_in > max_in:
                max_in = task_in

        self.__max_used_cpu_cores_in_python = max_in
        return max_in

    def _has_gil(self)->bool:
        from ..utils import has_gil
        return has_gil()
    
    def _get_task_queue(self)->SimpleQueue:
        return self._task_queue

    def _take_resource(self, task:Task)->None:
        with self._sys_info_lock:
            res = task.effective_res
            task.estimated_need_cpu_mem = res.cpu_mem

            if not self._has_gil():
                self._sys_info.cpu_cores_free -= res.cpu_cores
            else:
                max_in = self._max_used_cpu_cores_in_python()
                if res.cpu_cores_in_python > max_in:
                    self.__max_used_cpu_cores_in_python = res.cpu_cores_in_python
                    self._sys_info.cpu_cores_free -= (res.cpu_cores_in_python - max_in)
                self._sys_info.cpu_cores_free -= res.cpu_cores_out_of_python

            self._sys_info.cpu_mem_free -= task.estimated_need_cpu_mem
            if task.device is not None and task.device.gpu_index != -1:
                gpu_index = task.device.gpu_index
                self._sys_info.gpu_infos[gpu_index].n_cores_free -= res.gpu_cores
                self._sys_info.gpu_infos[gpu_index].mem_free -= res.gpu_mem

    def _release_resource(self, task:Task)->None:
        with self._sys_info_lock:
            res = task.effective_res

            if not self._has_gil():
                self._sys_info.cpu_cores_free += res.cpu_cores
            else:
                max_in = self._max_used_cpu_cores_in_python()
                if res.cpu_cores_in_python >= max_in:
                    self.__max_used_cpu_cores_in_python = None
                    max_in = self._max_used_cpu_cores_in_python()
                    self._sys_info.cpu_cores_free += (res.cpu_cores_in_python - max_in)
                self._sys_info.cpu_cores_free += res.cpu_cores_out_of_python

            self._sys_info.cpu_mem_free += task.estimated_need_cpu_mem
            if task.device is not None and task.device.gpu_index != -1:
                gpu_index = task.device.gpu_index
                self._sys_info.gpu_infos[gpu_index].n_cores_free += res.gpu_cores
                self._sys_info.gpu_infos[gpu_index].mem_free += res.gpu_mem

    def _estimate_cpu_cores_needed(self, res: Resource) -> float:
        if self._has_gil():
            max_in = self._max_used_cpu_cores_in_python()
            return max(0, res.cpu_cores_in_python - max_in) + res.cpu_cores_out_of_python
        else:
            return res.cpu_cores

    def _put_task(self, task:Task)->None:
        self._take_resource(task)
        worker:ThreadWorker = task.worker
        worker.is_working = True
        worker.add_task(task)

    def _reclaim_cpu_memory(self, task: Task, res: Resource) -> List:
        return []