from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Any, Optional, Callable, Union, Iterable

if TYPE_CHECKING:
    from concurrent.futures import Future

    from ..task import Task, Result
    from .worker import Worker


class ThreadPool:

    def __init__(
        self, max_workers:int=0, thread_name_prefix:str="ThreadPool.worker:",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        use_torch:bool=False
    ):
        import threading
        from queue import SimpleQueue
        import os
        if use_torch:
            import torch
            self._torch_cuda_available = torch.cuda.is_available()
        else:
            self._torch_cuda_available = False

        from ..sysinfo import SysInfo
        

        if not max_workers:
            max_workers = os.cpu_count()

        self._use_torch:bool = use_torch
        self._sys_info = SysInfo()
        self._max_workers:int = max_workers
        self._thread_name_prefix:str = thread_name_prefix
        self._initializer:Optional[Callable[..., Any]] = initializer
        self._initargs:Tuple[Any, ...] = initargs
        self._initkwargs:Optional[Dict[str, Any]] = initkwargs
        self._result_queue:SimpleQueue[Result] = SimpleQueue()
        self._workers:List[Worker] = []
        self._tasks:Dict[str, Task] = {}
        self._delayed_tasks:List[Task] = []
        self._lock = threading.Lock()
        self._shutdown = False
        self.__max_used_cpu_cores = None
        self.__max_used_gpu_cores = {}

        self._add_worker()

        self._result_thread = threading.Thread(target=self._collecting_result, daemon=True, name="collecting_result")
        self._result_thread.start()

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

            device = self._choose_task_device(task)
            if device is None:
                self._delayed_tasks.append(task)
                return task.future
            
            self._put_task(task)
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

    def starmap(
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
        from ..utils import batched

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

        return _chain_from_iterable_of_lists(ThreadPool._result_iterator(futures, end_time))

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
        
        worker:Worker = task.worker
        worker.is_working = True
        worker.task_queue.put(task)

    def _add_worker(self)->Worker:
        from .worker import Worker

        worker = Worker(
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
                
                should_pop_indices = []
                cancelled_task_ids = []
                for i, delayed_task in enumerate(self._delayed_tasks):
                    if delayed_task.future.cancelled():
                        cancelled_task_ids.append(delayed_task.id)
                        should_pop_indices.append(i)
                        continue

                    used_worker = self._choose_task_worker()
                    if used_worker is None:
                        continue

                    device = self._choose_task_device(delayed_task)
                    if device is None:
                        continue

                    self._put_task(delayed_task)
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
    
    def _choose_task_worker(self, task:Task)->Optional[Worker]:
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
        if self.working_count == 0 and len(self._delayed_tasks) == 0:
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

    def __enter__(self)->ThreadPool:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)->None:
        self.shutdown(wait=True)
