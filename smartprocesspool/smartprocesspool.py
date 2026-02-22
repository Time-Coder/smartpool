from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, List, Tuple, Any, Optional, Callable
    from concurrent.futures import Future

    from .task import Task, Result
    from .worker import Worker


class SmartProcessPool:

    def __init__(
        self, max_workers:Optional[int]=None, mp_context:str="spawn",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
        torch:bool=False
    ):
        import threading
        if torch:
            import torch.multiprocessing as mp
        else:
            import multiprocessing as mp

        from .sysinfo import SysInfo
        

        if max_workers is None:
            max_workers = mp.cpu_count()

        self._max_tasks_per_child:Optional[int] = max_tasks_per_child
        self._is_torch:bool = torch
        self._sys_info = SysInfo()
        self._max_workers:int = max_workers
        self._mp_context:str = mp_context
        self._initializer:Optional[Callable[..., Any]] = initializer
        self._initargs:Tuple[Any, ...] = initargs
        self._initkwargs:Optional[Dict[str, Any]] = initkwargs
        self._result_queue:mp.Queue[Result] = mp.Queue()
        self._workers:List[Worker] = []
        self._tasks:Dict[str, Task] = {}
        self._delayed_tasks:List[Task] = []
        self._futures:Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._shutdown = False

        self._add_worker()

        self._result_thread = threading.Thread(target=self._collecting_result, daemon=True)
        self._result_thread.start()

    def submit(
        self, func:Callable[..., Any],
        args:Optional[Tuple[Any]]=None, kwargs:Optional[Dict[str, Any]]=None,
        need_cpu_cores:int=1, need_cpu_mem:int=0,
        need_gpu_cores:int=0, need_gpu_mem:int=0
    ):
        from concurrent.futures import Future
        from .task import Task

        if args is None:
            args = []

        if kwargs is None:
            kwargs = {}

        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot submit after shutdown")
            
            task = Task(
                func=func,
                args=args,
                kwargs=kwargs,
                need_cpu_cores=need_cpu_cores,
                need_cpu_mem=need_cpu_mem,
                need_gpu_cores=need_gpu_cores,
                need_gpu_mem=need_gpu_mem
            )
            future = Future()
            self._futures[task.id] = future
            self._tasks[task.id] = task
            
            worker = self._choose_task_worker(task)
            if worker is None:
                self._delayed_tasks.append(task)
                return future

            device = self._choose_task_device(task, worker)
            if device is None:
                self._delayed_tasks.append(task)
                return future
            
            self._put_task(task, worker)
            return future

    def _put_task(self, task:Task, worker:Worker)->None:
        self._sys_info.cpu_cores_free -= task.need_cpu_cores

        real_need_cpu_mem = max(0, task.need_cpu_mem - worker.rss)
        self._sys_info.cpu_mem_free -= real_need_cpu_mem
        task_gpu_id:int = task.gpu_id
        if task_gpu_id != -1:
            self._sys_info.gpu_infos[task_gpu_id].n_cores_free -= task.need_gpu_cores
            self._sys_info.gpu_infos[task_gpu_id].memory_free -= task.need_gpu_mem
        
        self._futures[task.id].set_running_or_notify_cancel()
        worker.add_task(task)

    def _add_worker(self)->Worker:
        from .worker import Worker

        worker = Worker(
            len(self._workers), self._result_queue, self._mp_context,
            initializer=self._initializer,
            initargs=self._initargs,
            initkwargs=self._initkwargs,
            torch=self._is_torch
        )
        self._workers.append(worker)
        return worker

    def _collecting_result(self)->None:
        while not self._shutdown:
            result:Result = self._result_queue.get()

            with self._lock:
                future = self._futures.pop(result.task_id)
                if result.success:
                    future.set_result(result.result)
                else:
                    future.set_exception(result.result)

                task = self._tasks.pop(result.task_id)
                worker = self._workers[task.worker_index]
                worker.is_working = False
                worker.n_finished_tasks += 1
                if self._max_tasks_per_child is not None and worker.n_finished_tasks >= self._max_tasks_per_child:
                    worker.restart()

                self._sys_info.cpu_cores_free += task.need_cpu_cores
                hold_cpu_mem = max(worker.rss - task.mem_before_enter, 0)
                self._sys_info.cpu_mem_free += max(task.need_cpu_mem - hold_cpu_mem, 0)
                task_gpu_id:int = task.gpu_id
                if task_gpu_id != -1:
                    self._sys_info.gpu_infos[task_gpu_id].n_cores_free += task.need_gpu_cores
                    self._sys_info.gpu_infos[task_gpu_id].memory_free += task.need_gpu_mem
                
                should_pop_indices = []
                cancelled_task_ids = []
                for i, delayed_task in enumerate(self._delayed_tasks):
                    future = self._futures[delayed_task.id]
                    if future.cancelled():
                        cancelled_task_ids.append(delayed_task.id)
                        should_pop_indices.append(i)
                        continue

                    used_worker = self._choose_task_worker(delayed_task)
                    if used_worker is None:
                        continue

                    device = self._choose_task_device(delayed_task, used_worker)
                    if device is None:
                        continue

                    self._put_task(delayed_task, used_worker)
                    should_pop_indices.append(i)

                for i in reversed(should_pop_indices):
                    del self._delayed_tasks[i]

                for task_id in cancelled_task_ids:
                    del self._futures[task_id]
                    del self._tasks[task_id] 

                for task in self._tasks.values():
                    self._try_move_to_gpu(task)

    @property
    def working_count(self)->int:
        return sum(worker.is_working for worker in self._workers)
    
    def _choose_task_worker(self, task:Task)->Optional[Worker]:
        best_worker:Optional[Worker] = None
        max_overlap_size:int = 0
        for worker in self._workers:
            if worker.is_working:
                continue

            current_overlap_size = worker.overlap_modules_size(task)
            if not worker.is_working and (best_worker is None or current_overlap_size > max_overlap_size):
                max_overlap_size = current_overlap_size
                best_worker = worker
            
        if best_worker is not None:
            return best_worker
            
        if len(self._workers) < self._max_workers:
            return self._add_worker()
        else:
            return None

    def _try_move_to_gpu(self, task:Task)->None:
        if (
            task.device is None or
            task.device.startswith("cuda") or
            task.need_gpu_cores == 0 or
            task.worker_index < 0 or
            not self._workers[task.worker_index].is_working
        ):
            return
        
        gpus = self._sys_info.gpu_infos
        if not gpus:
            return
        
        need_gpu_cores:int = task.need_gpu_cores
        need_gpu_mem:int = task.need_gpu_mem

        best_gpu = None
        for gpu in gpus:
            if gpu.memory_free >= need_gpu_mem and gpu.n_cores_free >= need_gpu_cores:
                if best_gpu is None or gpu.n_cores_free > best_gpu.n_cores_free:
                    best_gpu = gpu

        if best_gpu is None:
            return

        worker = self._workers[task.worker_index]
        worker.change_device(best_gpu.device)
        task.device = best_gpu.device
        best_gpu.n_cores_free -= task.need_gpu_cores
        best_gpu.memory_free -= task.need_gpu_mem

    def _choose_task_device(self, task:Task, worker:Worker)->str:
        if self.working_count == 0 and len(self._delayed_tasks) == 0:
            self._sys_info.update()

        need_cpu_cores:int = task.need_cpu_cores
        if need_cpu_cores > self._sys_info.cpu_cores_free:
            task.device = None
            return None
        
        real_need_cpu_mem:int = task.need_cpu_mem - worker.rss
        if real_need_cpu_mem > self._sys_info.cpu_mem_free:
            task.device = None
            return None

        need_gpu_cores:int = task.need_gpu_cores
        need_gpu_mem:int = task.need_gpu_mem
        
        gpus = self._sys_info.gpu_infos
        if not gpus:
            task.device = "cpu"
            task.worker_index = worker.index
            return "cpu"

        best_gpu = None
        for gpu in gpus:
            if gpu.memory_free >= need_gpu_mem and gpu.n_cores_free >= need_gpu_cores:
                if best_gpu is None or gpu.n_cores_free > best_gpu.n_cores_free:
                    best_gpu = gpu

        if best_gpu is None:
            task.device = "cpu"
            task.worker_index = worker.index
            return "cpu"

        task.device = f"cuda:{best_gpu.id}"
        task.worker_index = worker.index
        return f"cuda:{best_gpu.id}"

    def shutdown(self, wait:bool=True, *, cancel_futures:bool=False)->None:
        with self._lock:
            if self._shutdown:
                return
            
            self._shutdown = True

            for worker in self._workers:
                worker.stop()

            if cancel_futures:
                for future in self._futures.values():
                    future.cancel()

            if wait:
                for worker in self._workers:
                    worker.join()

            self._result_queue.close()

    def __enter__(self)->SmartProcessPool:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)->None:
        self.shutdown(wait=True)
