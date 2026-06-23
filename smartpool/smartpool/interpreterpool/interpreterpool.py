from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from ..pool import Pool

if TYPE_CHECKING:

    from ..resource import Resource
    from ..task import Task
    from .interpreterworker import InterpreterWorker


class InterpreterPool(Pool):

    def __init__(
        self, max_workers:int=0,
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
        use_torch:bool=False
    ):
        import concurrent.interpreters as interpreters
        
        from .interpreterworker import InterpreterWorker

        if max_workers <= 0:
            import os
            max_workers = min(32, (os.cpu_count() or 1) + 4)

        Pool.__init__(
            self, max_workers=max_workers,

            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,

            result_queue_cls=interpreters.create_queue,

            max_tasks_per_child=max_tasks_per_child,
            use_torch=use_torch,
            worker_cls=InterpreterWorker,
            need_module_deps=True
        )

    def _take_resource(self, task:Task)->None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free -= res.cpu_cores

            task.estimated_need_cpu_mem = (1 - task.modules_overlap_ratio) * res.cpu_mem
            self._sys_info.cpu_mem_free -= task.estimated_need_cpu_mem

            task_gpu_index:int = task.gpu_index
            if task_gpu_index != -1:
                self._sys_info.gpu_infos[task_gpu_index].n_cores_free -= res.gpu_cores
                self._sys_info.gpu_infos[task_gpu_index].mem_free -= res.gpu_mem

    def _release_resource(self, task:Task)->None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free += res.cpu_cores
            self._sys_info.cpu_mem_free += task.estimated_need_cpu_mem
            task_gpu_index:int = task.gpu_index
            if task_gpu_index != -1:
                self._sys_info.gpu_infos[task_gpu_index].n_cores_free += res.gpu_cores
                self._sys_info.gpu_infos[task_gpu_index].mem_free += res.gpu_mem

    def _estimate_cpu_cores_needed(self, res: Resource) -> float:
        return res.cpu_cores

    def _put_task(self, task:Task)->None:
        self._take_resource(task)
        worker:InterpreterWorker = task.worker
        worker.is_working = True
        worker.imported_modules.update(task.module_deps)
        worker.add_task(task)
