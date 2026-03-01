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
        Pool.__init__(
            self, max_workers=max_workers,
            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,
            use_torch=use_torch
        )

        import threading
        from concurrent.interpreters import Queue
        import concurrent.interpreters as interpreters


        self._max_tasks_per_child:Optional[int] = max_tasks_per_child
        self._result_queue:Queue[Optional[Tuple[str, bool, Any, int]]] = interpreters.create_queue()

        self._add_worker()

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

    def _collecting_result(self)->None:
        while not self._shutdown:
            task_id, success, result = self._result_queue.get()

            with self._lock:
                task = self._tasks.pop(task_id)
                if success:
                    task.future.set_result(result)
                else:
                    task.future.set_exception(result)
                
                worker:InterpreterWorker = task.worker
                worker.is_working = False
                worker.n_finished_tasks += 1
                if self._max_tasks_per_child is not None and worker.n_finished_tasks >= self._max_tasks_per_child:
                    worker.restart()

                with self._sys_info_lock:
                    self._sys_info.cpu_cores_free += task.need_cpu_cores
                    self._sys_info.cpu_mem_free += task.estimated_need_cpu_mem
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
            
            task.estimated_need_cpu_mem = 0
            if task.need_cpu_mem > 0:
                task.estimated_need_cpu_mem = (1 - task.modules_overlap_ratio) * task.need_cpu_mem

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
