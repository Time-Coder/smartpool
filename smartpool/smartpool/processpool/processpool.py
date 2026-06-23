from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..pool import Pool

if TYPE_CHECKING:
    from ..resource import Resource
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
        chuncksize:int=1,
        use_torch:bool=False
    ):
        import queue
        import threading

        if use_torch:
            import torch.multiprocessing as mp
            from torch.multiprocessing.queue import SimpleQueue
        else:
            import multiprocessing as mp
            from multiprocessing.queues import SimpleQueue

        from .processworker import ProcessWorker

        if max_workers <= 0:
            import os
            max_workers = os.cpu_count()

        self._ctx = mp.get_context(mp_context)
        Pool.__init__(
            self, max_workers=max_workers,

            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,

            result_queue_cls=SimpleQueue,
            result_queue_kwargs={"ctx": self._ctx},

            max_tasks_per_child=max_tasks_per_child,
            chunksize=chuncksize,
            use_torch=use_torch,
            worker_cls=ProcessWorker,
            need_module_deps=True
        )

        self._process_name_prefix:str = process_name_prefix

        self._feeding_queue:queue.SimpleQueue[Task] = queue.SimpleQueue()
        self._feeding_thread = threading.Thread(target=self._feeding, daemon=True, name="feeding")
        self._feeding_thread.start()

    def _take_resource(self, task:Task)->None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free -= res.cpu_cores

            worker: ProcessWorker = task.worker
            task.mem_before_enter = worker.cached_rss
            task.estimated_need_cpu_mem = max(0, res.cpu_mem - task.modules_overlap_ratio * worker.cached_rss)
            self._sys_info.cpu_mem_free -= task.estimated_need_cpu_mem

            task_gpu_index:int = task.gpu_index
            if task_gpu_index != -1:
                self._sys_info.gpu_infos[task_gpu_index].n_cores_free -= res.gpu_cores
                self._sys_info.gpu_infos[task_gpu_index].mem_free -= res.gpu_mem

    def _release_resource(self, task:Task)->None:
        with self._sys_info_lock:
            res = task.effective_res
            self._sys_info.cpu_cores_free += res.cpu_cores

            worker:ProcessWorker = task.worker
            hold_cpu_mem = worker.cached_rss - task.mem_before_enter
            released_cpu_mem = task.estimated_need_cpu_mem - hold_cpu_mem
            self._sys_info.cpu_mem_free += released_cpu_mem

            task_gpu_index:int = task.gpu_index
            if task_gpu_index != -1:
                self._sys_info.gpu_infos[task_gpu_index].n_cores_free += res.gpu_cores
                self._sys_info.gpu_infos[task_gpu_index].mem_free += res.gpu_mem

    def _estimate_cpu_cores_needed(self, res: Resource) -> float:
        return res.cpu_cores

    def _sorted_idle_workers(self, exclude: ProcessWorker) -> Tuple[List[ProcessWorker], int]:
        workers:List[ProcessWorker] = []
        total_hold_mem:int = 0
        for worker in self._workers:
            if not worker.is_working and worker is not exclude and worker.cached_rss > 0:
                workers.append(worker)
                total_hold_mem += worker.cached_rss

        workers.sort(key=lambda worker: worker.cached_rss, reverse=True)
        return workers, total_hold_mem

    def _put_task(self, task:Task)->None:
        self._take_resource(task)
        worker:ProcessWorker = task.worker
        worker.is_working = True
        worker.imported_modules.update(task.module_deps)
        self._feeding_queue.put(task)

    def _feeding(self)->None:
        while not self._shutdown:
            task = self._feeding_queue.get()
            if task.future.cancelled():
                continue

            try:
                task.worker.add_task(task)
            except Exception as e:
                task.future.set_exception(e)
