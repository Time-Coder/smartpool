from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional, Callable

from ..pool import Pool

if TYPE_CHECKING:
    from ..task import Task
    from .processworker import ProcessWorker


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
        use_torch:bool=False
    ):
        import threading
        import queue

        if use_torch:
            import torch.multiprocessing as mp
            from torch.multiprocessing.queue import SimpleQueue
        else:
            import multiprocessing as mp
            from multiprocessing.queues import SimpleQueue

        self._ctx = mp.get_context(mp_context)
        Pool.__init__(
            self, max_workers=max_workers,

            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,

            result_queue_cls=SimpleQueue,
            result_queue_kwargs={"ctx": self._ctx},
            
            max_tasks_per_child=max_tasks_per_child,
            use_torch=use_torch,
            need_module_deps=True
        )

        self._process_name_prefix:str = process_name_prefix
        
        self._feeding_queue:queue.SimpleQueue[Tuple[Task, SimpleQueue]] = queue.SimpleQueue()
        self._feeding_thread = threading.Thread(target=self._feeding, daemon=True, name="feeding")
        self._feeding_thread.start()

    def _take_resource(self, task:Task):
        with self._sys_info_lock:
            self._sys_info.cpu_cores_free -= task.need_cpu_cores
            self._sys_info.cpu_mem_free -= task.estimated_need_cpu_mem
            task_gpu_id:int = task.gpu_id
            if task_gpu_id != -1:
                self._sys_info.gpu_infos[task_gpu_id].n_cores_free -= task.need_gpu_cores
                self._sys_info.gpu_infos[task_gpu_id].mem_free -= task.need_gpu_mem

    def _release_resource(self, task:Task):
        with self._sys_info_lock:
            self._sys_info.cpu_cores_free += task.need_cpu_cores
            worker:ProcessWorker = task.worker
            hold_cpu_mem = max(worker.cached_rss - task.mem_before_enter, 0)
            self._sys_info.cpu_mem_free += max(task.need_cpu_mem - hold_cpu_mem, 0)
            task_gpu_id:int = task.gpu_id
            if task_gpu_id != -1:
                self._sys_info.gpu_infos[task_gpu_id].n_cores_free += task.need_gpu_cores
                self._sys_info.gpu_infos[task_gpu_id].mem_free += task.need_gpu_mem

    def _estimate_need_cpu_cores(self, task):
        return task.need_cpu_cores
    
    def _estimate_need_cpu_mem(self, task):
        worker:ProcessWorker = task.worker
        task.mem_before_enter = worker.cached_rss
        return max(0, task.need_cpu_mem - task.modules_overlap_ratio * worker.cached_rss)
    
    def _estimate_need_gpu_cores(self, task, gpu_id):
        return task.need_gpu_cores

    def _put_task(self, task:Task)->None:
        self._take_resource(task)
        worker:ProcessWorker = task.worker
        worker.is_working = True
        worker.imported_modules.update(task.module_deps)
        self._feeding_queue.put((task, worker.task_queue))
        
    def _feeding(self)->None:
        while not self._shutdown:
            task, task_queue = self._feeding_queue.get()
            if task.future.cancelled():
                continue

            try:
                task_queue.put(task.info())
                task.future.set_running_or_notify_cancel()
            except BaseException as e:
                task.future.set_exception(e)

    def _add_worker(self)->ProcessWorker:
        from .processworker import ProcessWorker

        worker = ProcessWorker(
            len(self._workers), self._process_name_prefix,
            self._result_queue, self._ctx,
            initializer=self._initializer,
            initargs=self._initargs,
            initkwargs=self._initkwargs,
            use_torch=self._use_torch,
            torch_cuda_available=self._torch_cuda_available
        )
        self._workers.append(worker)
        return worker
