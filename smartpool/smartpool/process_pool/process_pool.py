from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from ..pool import Pool

if TYPE_CHECKING:
    from ..task import Task
    from .process_worker import ProcessWorker
    from ..resource import Resource


class ProcessPool(Pool):

    def __init__(
        self, max_workers:int=0,
        process_name_prefix:str="ProcessPool.worker:",
        mp_context:str="spawn",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
        chunk_timeout:float=0.1,
        use_torch:bool=False
    ):
        if use_torch:
            import torch.multiprocessing as mp
            from torch.multiprocessing.queue import Queue, SimpleQueue
        else:
            import multiprocessing as mp
            from multiprocessing.queues import Queue, SimpleQueue

        from .process_worker import ProcessWorker

        if max_workers <= 0:
            import os
            max_workers = os.cpu_count()

        self._ctx = mp.get_context(mp_context)
        self._process_name_prefix:str = process_name_prefix

        Pool.__init__(
            self, max_workers=max_workers,

            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,

            result_queue_cls=SimpleQueue,
            result_queue_kwargs={"ctx": self._ctx},

            max_tasks_per_child=max_tasks_per_child,
            use_torch=use_torch,
            worker_cls=ProcessWorker,
            chunk_timeout=chunk_timeout,
            need_module_deps=True
        )

    @classmethod
    def config(
        cls, pool_name: str="default", *,
        max_workers:int=0,
        process_name_prefix:str="ProcessPool.worker:",
        mp_context:str="spawn",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        max_tasks_per_child:Optional[int]=None,
        chunk_timeout:float=0.1,
        use_torch:bool=False
    ) -> None:
        Pool.config(
            cls, pool_name,
            max_workers=max_workers,
            process_name_prefix=process_name_prefix,
            mp_context=mp_context,
            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,
            max_tasks_per_child=max_tasks_per_child,
            chunk_timeout=chunk_timeout,
            use_torch=use_torch
        )

    @classmethod
    def task(
        cls, func:Callable=None, *,
        pool_name:str="default",
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        check_args: bool = True,
        chunksize: int = 1,
        use_torch: Optional[bool] = None,
        device_changeable: bool = False
    ):
        return Pool.task(
            cls, func,
            pool_name=pool_name,
            cpu_mode_res=cpu_mode_res,
            gpu_mode_res=gpu_mode_res,
            check_args=check_args,
            chunksize=chunksize,
            use_torch=use_torch,
            device_changeable=device_changeable
        )

    def _take_resource(self, task:Task)->None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free -= res.cpu_cores

            worker: ProcessWorker = task.worker
            task.mem_before_enter = worker.cached_rss
            task.estimated_need_cpu_mem = max(0, res.cpu_mem - task.modules_overlap_ratio * worker.cached_rss)
            self._sys_info.cpu_mem_free -= task.estimated_need_cpu_mem

            if task.device.gpu_index != -1:
                gpu_index = task.device.gpu_index
                self._sys_info.gpu_infos[gpu_index].n_cores_free -= res.gpu_cores
                self._sys_info.gpu_infos[gpu_index].mem_free -= res.gpu_mem

    def _release_resource(self, task:Task)->None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free += res.cpu_cores

            worker:ProcessWorker = task.worker
            hold_cpu_mem = worker.cached_rss - task.mem_before_enter
            result_cpu_mem = min(res.result_cpu_mem, max(0, task.estimated_need_cpu_mem - hold_cpu_mem))
            released_cpu_mem = task.estimated_need_cpu_mem - hold_cpu_mem - result_cpu_mem
            self._sys_info.cpu_mem_free += released_cpu_mem

            result_gpu_mem = 0
            if task.device.gpu_index != -1:
                gpu_index = task.device.gpu_index
                gpu_info = self._sys_info.gpu_infos[gpu_index]
                gpu_info.n_cores_free += res.gpu_cores
                result_gpu_mem = min(res.result_gpu_mem, res.gpu_mem)
                gpu_info.mem_free += res.gpu_mem - result_gpu_mem

        self._hold_result_mem(task, result_cpu_mem, result_gpu_mem)

    def _put_task(self, task:Task)->None:
        self._take_resource(task)
        self._register_gpu_candidate(task)
        worker:ProcessWorker = task.worker
        worker.is_working = True

        if task.future.cancelled():
            self._on_task_cancelled(task)
            return

        try:
            worker.imported_modules.update(task.module_deps)
            worker.add_task(task)
        except Exception as e:
            self._on_task_cancelled(task)
            task.future.set_exception(e)
