from __future__ import annotations
from ..pool import Pool
from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional, Callable

if TYPE_CHECKING:
    from ..task import Task, Result
    from .threadworker import ThreadWorker


class ThreadPool(Pool):

    def __init__(
        self, max_workers:int=0, thread_name_prefix:str="ThreadPool.worker:",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
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
        from queue import SimpleQueue
        

        self._thread_name_prefix:str = thread_name_prefix
        self._result_queue:SimpleQueue[Result] = SimpleQueue()
        self.__max_used_cpu_cores = None
        self.__max_used_gpu_cores = {}

        self._add_worker()

        self._result_thread = threading.Thread(target=self._collecting_result, daemon=True, name="collecting_result")
        self._result_thread.start()

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

    def _put_task(self, task:Task)->None:
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

    def _collecting_result(self)->None:
        while not self._shutdown:
            result:Result = self._result_queue.get()

            with self._lock:
                task = self._tasks.pop(result.task_id)
                if result.exception is None:
                    task.future.set_result(result.result)
                else:
                    task.future.set_exception(result.exception)
                
                task.worker.is_working = False
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
                
                self._postprocess_after_task_done()

    @property
    def working_count(self)->int:
        return sum(worker.is_working for worker in self._workers)
    
    def _choose_task_worker(self, task:Task)->Optional[ThreadWorker]:
        for worker in self._workers:
            if not worker.is_working:
                task.worker = worker
                return worker

        if len(self._workers) < self._max_workers:
            task.worker = self._add_worker()
        else:
            task.worker = None
        
        return task.worker

    def _try_move_to_gpu(self, task:Task)->None:
        if (
            task.device is None or
            task.device.startswith("cuda") or
            task.need_gpu_cores == 0 or
            task.worker is None or
            not task.worker.is_working
        ):
            return
        
        with self._sys_info_lock:
            gpus = self._sys_info.gpu_infos
            if not gpus:
                return
            
            need_gpu_cores:int = task.need_gpu_cores
            need_gpu_mem:int = task.need_gpu_mem

            best_gpu = None
            for gpu in gpus:
                if gpu.mem_free >= need_gpu_mem and gpu.n_cores_free >= need_gpu_cores:
                    if best_gpu is None or gpu.n_cores_free > best_gpu.n_cores_free:
                        best_gpu = gpu

            if best_gpu is None:
                return

            task.worker.change_device(best_gpu.device)
            task.device = best_gpu.device
            best_gpu.n_cores_free -= task.need_gpu_cores
            best_gpu.mem_free -= task.need_gpu_mem

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

            if task.need_cpu_mem > self._sys_info.cpu_mem_free:
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
