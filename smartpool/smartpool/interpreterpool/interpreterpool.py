from __future__ import annotations
from ..pool import Pool
from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional, Callable

if TYPE_CHECKING:
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

        Pool.__init__(
            self, max_workers=max_workers,
            
            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,

            result_queue_cls=interpreters.create_queue,

            max_tasks_per_child=max_tasks_per_child,
            use_torch=use_torch,
            need_module_deps=True
        )

    def _take_resource(self, task):
        with self._sys_info_lock:
            self._sys_info.cpu_cores_free -= task.need_cpu_cores
            self._sys_info.cpu_mem_free -= task.estimated_need_cpu_mem
            task_gpu_id:int = task.gpu_id
            if task_gpu_id != -1:
                self._sys_info.gpu_infos[task_gpu_id].n_cores_free -= task.need_gpu_cores
                self._sys_info.gpu_infos[task_gpu_id].mem_free -= task.need_gpu_mem

    def _release_resource(self, task):
        with self._sys_info_lock:
            self._sys_info.cpu_cores_free += task.need_cpu_cores
            self._sys_info.cpu_mem_free += task.estimated_need_cpu_mem
            task_gpu_id:int = task.gpu_id
            if task_gpu_id != -1:
                self._sys_info.gpu_infos[task_gpu_id].n_cores_free += task.need_gpu_cores
                self._sys_info.gpu_infos[task_gpu_id].mem_free += task.need_gpu_mem

    def _estimate_need_cpu_cores(self, task):
        return task.need_cpu_cores
    
    def _estimate_need_gpu_cores(self, task, gpu_id):
        return task.need_gpu_cores
    
    def _estimate_need_cpu_mem(self, task):
        return (1 - task.modules_overlap_ratio) * task.need_cpu_mem

    def _put_task(self, task:Task)->None:
        self._take_resource(task)
        worker:InterpreterWorker = task.worker
        worker.is_working = True
        worker.imported_modules.update(task.module_deps)
        task.future.set_running_or_notify_cancel()
        worker.task_queue.put(task.info())

    def _add_worker(self)->InterpreterWorker:
        from .interpreterworker import InterpreterWorker

        worker = InterpreterWorker(
            len(self._workers), self._result_queue,
            initializer=self._initializer,
            initargs=self._initargs,
            initkwargs=self._initkwargs,
            torch_cuda_available=self._torch_cuda_available
        )
        self._workers.append(worker)
        return worker
