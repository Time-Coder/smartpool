from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional, Callable

from ..pool import Pool

if TYPE_CHECKING:
    from ..task import Task
    from .processworker import ProcessWorker


class ProcessPool(Pool):

    def __init__(
        self, max_workers:int=0, process_name_prefix:str="ProcessPool.worker:",
        mp_context:str="spawn",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
        use_torch:bool=False
    ):
        Pool.__init__(
            self, max_workers=max_workers,
            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,
            use_torch=use_torch
        )

        import threading
        import queue

        if use_torch:
            import torch.multiprocessing as mp
            from torch.multiprocessing.queue import SimpleQueue
        else:
            import multiprocessing as mp
            from multiprocessing.queues import SimpleQueue

        self._need_module_deps:bool = True
        self._max_tasks_per_child:Optional[int] = max_tasks_per_child
        self._process_name_prefix:str = process_name_prefix
        self._ctx = mp.get_context(mp_context)
        self._result_queue:SimpleQueue[Optional[Tuple[str, bool, Any]]] = SimpleQueue(ctx=self._ctx)

        self._add_worker()

        self._feeding_queue:queue.SimpleQueue[Tuple[Task, SimpleQueue]] = queue.SimpleQueue()
        self._feeding_thread = threading.Thread(target=self._feeding, daemon=True, name="feeding")
        self._feeding_thread.start()

        self._result_thread = threading.Thread(target=self._collecting_result, daemon=True, name="collecting_result")
        self._result_thread.start()

    def _put_task(self, task:Task)->None:
        with self._sys_info_lock:
            self._sys_info.cpu_cores_free -= task.need_cpu_cores
            self._sys_info.cpu_mem_free -= task.estimated_need_cpu_mem
            task_gpu_id:int = task.gpu_id
            if task_gpu_id != -1:
                self._sys_info.gpu_infos[task_gpu_id].n_cores_free -= task.need_gpu_cores
                self._sys_info.gpu_infos[task_gpu_id].mem_free -= task.need_gpu_mem
        
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

    def _collecting_result(self)->None:
        while not self._shutdown:
            task_id, success, result = self._result_queue.get()

            with self._lock:
                task = self._tasks.pop(task_id)
                if success:
                    task.future.set_result(result)
                else:
                    task.future.set_exception(result)
                
                worker:ProcessWorker = task.worker
                worker.is_working = False
                worker.n_finished_tasks += 1
                if self._max_tasks_per_child is not None and worker.n_finished_tasks >= self._max_tasks_per_child:
                    worker.restart()

                with self._sys_info_lock:
                    self._sys_info.cpu_cores_free += task.need_cpu_cores
                    hold_cpu_mem = max(worker.cached_rss - task.mem_before_enter, 0)
                    self._sys_info.cpu_mem_free += max(task.need_cpu_mem - hold_cpu_mem, 0)
                    task_gpu_id:int = task.gpu_id
                    if task_gpu_id != -1:
                        self._sys_info.gpu_infos[task_gpu_id].n_cores_free += task.need_gpu_cores
                        self._sys_info.gpu_infos[task_gpu_id].mem_free += task.need_gpu_mem
                
                self._postprocess_after_task_done()

    def _choose_task_device(self, task:Task)->str:
        from ..worker import Worker

        with self._sys_info_lock:
            if Worker.total_working_count() == 0:
                self._sys_info.update()

            need_cpu_cores:int = task.need_cpu_cores
            if need_cpu_cores > self._sys_info.cpu_cores_free:
                task.device = None
                task.worker = None
                return None
            
            worker:ProcessWorker = task.worker
            task.mem_before_enter = worker.cached_rss
            task.estimated_need_cpu_mem = 0
            if task.need_cpu_mem > 0:
                task.estimated_need_cpu_mem = max(0, task.need_cpu_mem - task.modules_overlap_ratio * worker.cached_rss)

            if task.estimated_need_cpu_mem > self._sys_info.cpu_mem_free:
                task.device = None
                task.worker = None
                return None

            if not self._torch_cuda_available or (task.need_gpu_cores == 0 and task.need_gpu_mem == 0):
                task.device = "cpu"
                return "cpu"

            gpus = self._sys_info.gpu_infos
            if not gpus:
                task.device = "cpu"
                return "cpu"

            best_gpu = None
            for gpu in gpus:
                if gpu.mem_free >= task.need_gpu_mem and gpu.n_cores_free >= task.need_gpu_cores:
                    if best_gpu is None or gpu.n_cores_free > best_gpu.n_cores_free:
                        best_gpu = gpu

        if best_gpu is None:
            task.device = "cpu"
            return "cpu"

        task.device = f"cuda:{best_gpu.id}"
        return task.device
