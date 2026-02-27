from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Any, Optional, Callable, Union, Iterable

if TYPE_CHECKING:
    from concurrent.futures import Future

    from ..task import Task
    from .worker import Worker


class ProcessPool:

    def __init__(
        self, max_workers:int=0, mp_context:str="spawn",
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
            import torch
            import torch.multiprocessing as mp
            from torch.multiprocessing.queue import SimpleQueue
            self._torch_cuda_available = torch.cuda.is_available()
        else:
            import multiprocessing as mp
            from multiprocessing.queues import SimpleQueue
            self._torch_cuda_available = False

        from ..sysinfo import SysInfo
        

        if not max_workers:
            max_workers = mp.cpu_count()

        self._max_tasks_per_child:Optional[int] = max_tasks_per_child
        self._use_torch:bool = use_torch
        self._sys_info = SysInfo()
        self._max_workers:int = max_workers
        self._initializer:Optional[Callable[..., Any]] = initializer
        self._initargs:Tuple[Any, ...] = initargs
        self._initkwargs:Optional[Dict[str, Any]] = initkwargs

        self._ctx = mp.get_context(mp_context)
        self._result_queue:SimpleQueue[Optional[Tuple[str, bool, Any]]] = SimpleQueue(ctx=self._ctx)
        
        self._workers:List[Worker] = []
        self._tasks:Dict[str, Task] = {}
        self._delayed_tasks:List[Task] = []
        self._lock = threading.Lock()
        self._shutdown = False

        self._add_worker()

        self._result_thread = threading.Thread(target=self._collecting_result, daemon=True, name="collecting_result")
        self._result_thread.start()

        self._feeding_queue:queue.Queue[Tuple[Task, SimpleQueue]] = queue.Queue()
        self._feeding_thread = threading.Thread(target=self._feeding, daemon=True, name="feeding")
        self._feeding_thread.start()

    def submit(
        self, func:Callable[..., Any],
        args:Optional[Tuple[Any]]=None, kwargs:Optional[Dict[str, Any]]=None,
        need_cpu_cores:int=1, need_cpu_mem:int=0,
        need_gpu_cores:int=0, need_gpu_mem:int=0
    ):
        from ..task import Task

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
            self._tasks[task.id] = task
            
            worker = self._choose_task_worker(task)
            if worker is None:
                self._delayed_tasks.append(task)
                return task.future

            device = self._choose_task_device(task, worker)
            if device is None:
                self._delayed_tasks.append(task)
                return task.future
            
            self._put_task(task, worker)
            return task.future

    @staticmethod
    def _result_iterator(futures:List[Future], end_time:Optional[float]):
        from concurrent.futures._base import _result_or_cancel
        import time

        try:
            futures.reverse()
            while futures:
                if end_time is None:
                    yield _result_or_cancel(futures.pop())
                else:
                    yield _result_or_cancel(futures.pop(), end_time - time.monotonic())
        finally:
            for future in futures:
                future.cancel()

    def map(
        self, func:Callable[..., Any],
        args_iterables:Iterable[Tuple[Any, ...]],
        need_cpu_cores:Union[int, Iterable[int]]=1,
        need_cpu_mem:Union[int, Iterable[int]]=0,
        need_gpu_cores:Union[int, Iterable[int]]=0,
        need_gpu_mem:Union[int, Iterable[int]]=0,
        timeout:Optional[Union[float, int]]=None,
        chunksize:int=1
    ):
        from functools import partial
        import itertools
        import time
        from collections.abc import Iterable
        from concurrent.futures.process import _process_chunk, _chain_from_iterable_of_lists
        from .module_deps import batched

        if not isinstance(need_cpu_cores, Iterable):
            need_cpu_cores = itertools.repeat(need_cpu_cores)

        if not isinstance(need_cpu_mem, Iterable):
            need_cpu_mem = itertools.repeat(need_cpu_mem)

        if not isinstance(need_gpu_cores, Iterable):
            need_gpu_cores = itertools.repeat(need_gpu_cores)

        if not isinstance(need_gpu_mem, Iterable):
            need_gpu_mem = itertools.repeat(need_gpu_mem)

        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        end_time = None
        if timeout is not None:
            end_time = timeout + time.monotonic()

        target_func = partial(_process_chunk, func)
        futures:List[Future] = []
        iterator = zip(args_iterables, need_cpu_cores, need_cpu_mem, need_gpu_cores, need_gpu_mem)
        for batch in batched(iterator, chunksize):
            args_batch, cpu_cores_batch, cpu_mem_batch, gpu_cores_batch, gpu_mem_batch = zip(*batch)
            future = self.submit(
                target_func, args=(args_batch,),
                need_cpu_cores=max(cpu_cores_batch),
                need_cpu_mem=max(cpu_mem_batch),
                need_gpu_cores=max(gpu_cores_batch),
                need_gpu_mem=max(gpu_mem_batch)
            )
            futures.append(future)

        return _chain_from_iterable_of_lists(ProcessPool._result_iterator(futures, end_time))

    def _put_task(self, task:Task, worker:Worker)->None:
        self._sys_info.cpu_cores_free -= task.need_cpu_cores
        self._sys_info.cpu_mem_free -= task.estimated_need_cpu_mem
        task_gpu_id:int = task.gpu_id
        if task_gpu_id != -1:
            self._sys_info.gpu_infos[task_gpu_id].n_cores_free -= task.need_gpu_cores
            self._sys_info.gpu_infos[task_gpu_id].memory_free -= task.need_gpu_mem
        
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

    def _add_worker(self)->Worker:
        from .worker import Worker

        worker = Worker(
            len(self._workers), self._result_queue, self._ctx,
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
                
                worker = self._workers[task.worker_index]
                worker.is_working = False
                worker.n_finished_tasks += 1
                if self._max_tasks_per_child is not None and worker.n_finished_tasks >= self._max_tasks_per_child:
                    worker.restart()

                self._sys_info.cpu_cores_free += task.need_cpu_cores
                hold_cpu_mem = max(worker.cached_rss - task.mem_before_enter, 0)
                self._sys_info.cpu_mem_free += max(task.need_cpu_mem - hold_cpu_mem, 0)
                task_gpu_id:int = task.gpu_id
                if task_gpu_id != -1:
                    self._sys_info.gpu_infos[task_gpu_id].n_cores_free += task.need_gpu_cores
                    self._sys_info.gpu_infos[task_gpu_id].memory_free += task.need_gpu_mem
                
                should_pop_indices = []
                cancelled_task_ids = []
                for i, delayed_task in enumerate(self._delayed_tasks):
                    if delayed_task.future.cancelled():
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
                    del self._tasks[task_id] 

                if self._torch_cuda_available:
                    for task in self._tasks.values():
                        self._try_move_to_gpu(task)

    @property
    def working_count(self)->int:
        return sum(worker.is_working for worker in self._workers)
    
    def _choose_task_worker(self, task:Task)->Optional[Tuple[Worker, int]]:
        best_worker:Optional[Worker] = None
        task.modules_overlap_ratio = 0.0
        for worker in self._workers:
            if worker.is_working:
                continue

            if task.need_cpu_mem == 0 or task.func.__module__ in worker.imported_modules:
                task.modules_overlap_ratio = 1.0
                return worker

            current_overlap_ratio = worker.overlap_modules_ratio(task)
            if best_worker is None or current_overlap_ratio > task.modules_overlap_ratio:
                task.modules_overlap_ratio = current_overlap_ratio
                best_worker = worker
            
        if best_worker is not None:
            return best_worker
            
        task.modules_overlap_ratio = 0.0
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
        
        task.mem_before_enter = worker.cached_rss
        task.estimated_need_cpu_mem = 0
        if task.need_cpu_mem > 0:
            task.estimated_need_cpu_mem = max(0, task.need_cpu_mem - task.modules_overlap_ratio * worker.cached_rss)

        if task.estimated_need_cpu_mem > self._sys_info.cpu_mem_free:
            task.device = None
            return None

        if not self._torch_cuda_available or (task.need_gpu_cores == 0 and task.need_gpu_mem == 0):
            task.device = "cpu"
            task.worker_index = worker.index
            return "cpu"

        gpus = self._sys_info.gpu_infos
        if not gpus:
            task.device = "cpu"
            task.worker_index = worker.index
            return "cpu"

        best_gpu = None
        for gpu in gpus:
            if gpu.memory_free >= task.need_gpu_mem and gpu.n_cores_free >= task.need_gpu_cores:
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
                for task in self._tasks.values():
                    task.future.cancel()

            if wait:
                for worker in self._workers:
                    worker.join()

            self._result_queue.close()

    def __enter__(self)->ProcessPool:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)->None:
        self.shutdown(wait=True)
