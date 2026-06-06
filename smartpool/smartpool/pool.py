from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple, Any, Optional, Callable, Union, Iterable
import weakref

from .utils import Resource

if TYPE_CHECKING:
    import threading
    from concurrent.futures import Future

    from .sysinfo import SysInfo
    from .task import Task
    from .worker import Worker
    from .utils import QueueLike


class Pool(ABC):

    _sys_info_lock:Optional[threading.Lock] = None
    _sys_info:Optional[SysInfo] = None
    _updating_sysinfo_thread:Optional[threading.Thread] = None
    _instances:weakref.WeakSet[Pool] = weakref.WeakSet()

    def __init__(
        self, max_workers:int,

        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]],

        result_queue_cls:type,
        result_queue_args:Tuple[Any, ...]=(),
        result_queue_kwargs:Dict[str, Any]=None,
        
        *,

        max_tasks_per_child:Optional[int],
        use_torch:bool,
        need_module_deps:bool,
        need_result_thread:bool
    ):
        self._init_sys_info()

        import threading
        import os

        if use_torch:
            import torch
            self._torch_gpu_available = torch.cuda.is_available()
        else:
            self._torch_gpu_available = False

        if not max_workers:
            max_workers = os.cpu_count()

        self._max_tasks_per_child:int = max_tasks_per_child
        self._need_module_deps:bool = need_module_deps
        self._use_torch:bool = use_torch
        self._max_workers:int = max_workers
        self._workers_working_count:int = 0
        self._initializer:Optional[Callable[..., Any]] = initializer
        self._initargs:Tuple[Any, ...] = initargs
        self._initkwargs:Optional[Dict[str, Any]] = initkwargs

        if result_queue_kwargs is None:
            result_queue_kwargs = {}

        if result_queue_cls is not None:
            self._result_queue:QueueLike[Tuple[str, bool, Any]] = result_queue_cls(*result_queue_args, **result_queue_kwargs)
        
        self._workers:List[Worker] = []
        self._tasks:Dict[str, Task] = {}
        self._delayed_tasks:List[Task] = []
        self._lock:threading.Lock = threading.Lock()
        self._shutdown:bool = False
        self._need_result_thread:bool = need_result_thread
        self._result_thread:Optional[threading.Thread] = None

        Pool._instances.add(self)

    def submit(
        self, func:Callable[..., Any],
        args:Optional[Tuple[Any]]=None, kwargs:Optional[Dict[str, Any]]=None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None
    )->Future:
        import threading
        from .task import Task

        if args is None:
            args = []

        if kwargs is None:
            kwargs = {}

        if cpu_mode_res is None:
            cpu_mode_res = Resource(cpu_cores_in_python=1)

        if gpu_mode_res is None:
            gpu_mode_res = cpu_mode_res

        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot submit after shutdown")
            
            task = Task(
                func=func,
                args=args,
                kwargs=kwargs,
                cpu_mode_res=cpu_mode_res,
                gpu_mode_res=gpu_mode_res,
                calculate_module_deps=self._need_module_deps
            )
            self._tasks[task.id] = task

            if not self._try_assign_task(task):
                self._delayed_tasks.append(task)
                return task.future
            
            self._put_task(task)
            if self._need_result_thread and self._result_thread is None:
                self._result_thread = threading.Thread(target=self._collecting_result, daemon=True, name="collecting_result")
                self._result_thread.start()

            return task.future
        
    def _try_assign_task(self, task: Task) -> bool:
        from .worker import Worker

        gpu_res = task.gpu_mode_res
        if self._torch_gpu_available and (gpu_res.gpu_cores > 0 or gpu_res.gpu_mem > 0):
            self._choose_task_worker(task, gpu_res)
            if task.worker is not None:
                device = self._choose_task_device(task, gpu_res, dry_run=True)
                if device is not None:
                    device = self._choose_task_device(task, gpu_res, dry_run=False)
                    if device is not None:
                        return True

                task.worker = None
                task.modules_overlap_ratio = 0.0

        cpu_res = task.cpu_mode_res
        self._choose_task_worker(task, cpu_res)
        if task.worker is not None:
            device = self._choose_task_device(task, cpu_res, dry_run=False)
            if device is not None:
                return True

        task.worker = None
        task.modules_overlap_ratio = 0.0
        return False

    def _init_sys_info(self)->None:
        if Pool._sys_info is not None:
            return

        from .sysinfo import SysInfo
        import threading

        Pool._sys_info = SysInfo()
        Pool._sys_info_lock = threading.Lock()
        Pool._updating_sysinfo_thread = threading.Thread(target=Pool._updating_sysinfo, daemon=True)
        Pool._updating_sysinfo_thread.start()

    @staticmethod
    def _updating_sysinfo()->None:
        while True:
            Pool._sys_info.update_cpu_percent()

            for pool in Pool._instances:
                if pool._workers_working_count > 0 or not pool._delayed_tasks:
                    continue

                with pool._lock:
                    pool._postprocess_after_task_done()

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

    def _batch_resource(self, resources: List[Resource]) -> Resource:
        return Resource(
            cpu_cores_in_python=max(r.cpu_cores_in_python for r in resources),
            cpu_cores_out_of_python=max(r.cpu_cores_out_of_python for r in resources),
            cpu_mem=max(r.cpu_mem for r in resources),
            gpu_cores=max(r.gpu_cores for r in resources),
            gpu_mem=max(r.gpu_mem for r in resources),
        )

    def starmap(
        self, func:Callable[..., Any],
        args_iterables:Iterable[Tuple[Any, ...]],
        cpu_mode_res:Union[Resource, Iterable[Resource]] = Resource(cpu_cores_in_python=1),
        gpu_mode_res:Union[Resource, Iterable[Resource], None]=None,
        timeout:Optional[Union[float, int]]=None,
        chunksize:int=1
    )->Iterable[Any]:
        from functools import partial
        import itertools
        import time
        from collections.abc import Iterable
        from concurrent.futures.process import _process_chunk, _chain_from_iterable_of_lists
        from .utils import batched

        if not isinstance(cpu_mode_res, Iterable):
            cpu_mode_res = itertools.repeat(cpu_mode_res)

        if gpu_mode_res is None:
            gpu_mode_res = cpu_mode_res
        elif not isinstance(gpu_mode_res, Iterable):
            gpu_mode_res = itertools.repeat(gpu_mode_res)

        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        end_time = None
        if timeout is not None:
            end_time = timeout + time.monotonic()

        target_func = partial(_process_chunk, func)
        futures:List[Future] = []
        iterator = zip(args_iterables, cpu_mode_res, gpu_mode_res)
        for batch in batched(iterator, chunksize):
            args_batch, cpu_res_batch, gpu_res_batch = zip(*batch)
            future = self.submit(
                target_func, args=(args_batch,),
                cpu_mode_res=self._batch_resource(cpu_res_batch),
                gpu_mode_res=self._batch_resource(gpu_res_batch),
            )
            futures.append(future)

        return _chain_from_iterable_of_lists(Pool._result_iterator(futures, end_time))

    def map(
        self, func:Callable[..., Any],
        iterable:Iterable[Any],
        cpu_mode_res:Union[Resource, Iterable[Resource]] = Resource(cpu_cores_in_python=1),
        gpu_mode_res:Union[Resource, Iterable[Resource], None]=None,
        timeout:Optional[Union[float, int]]=None,
        chunksize:int=1
    )->Iterable[Any]:
        args_iterable = ((item,) for item in iterable)
        
        return self.starmap(
            func=func,
            args_iterables=args_iterable,
            cpu_mode_res=cpu_mode_res,
            gpu_mode_res=gpu_mode_res,
            timeout=timeout,
            chunksize=chunksize
        )

    def shutdown(self, wait:bool=True, *, cancel_futures:bool=False)->None:
        with self._lock:
            if self._shutdown:
                return
            
            self._shutdown = True

            for worker in self._workers:
                worker.stop(wait=False, clear=False)

            if cancel_futures:
                for task in self._tasks.values():
                    task.future.cancel()

            if wait:
                for worker in self._workers:
                    worker.join()

    def __enter__(self)->Pool:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)->None:
        self.shutdown(wait=True)

    def _on_task_done(self, task_id:str, success:bool, result:Any)->None:
        with self._lock:
            task = self._tasks.pop(task_id)
            if success:
                task.future.set_result(result)
            else:
                task.future.set_exception(result)
            
            worker:Worker = task.worker
            worker.is_working = False
            self._workers_working_count -= 1
            worker.n_finished_tasks += 1
            if self._max_tasks_per_child is not None and worker.n_finished_tasks >= self._max_tasks_per_child:
                worker.stop(wait=False, clear=True)

            self._release_resource(task)
            self._postprocess_after_task_done()

    def _collecting_result(self)->None:
        while not self._shutdown:
            task_id, success, result = self._result_queue.get()
            self._on_task_done(task_id, success, result)
    
    def _postprocess_after_task_done(self)->None:
        if self._delayed_tasks:
            should_pop_indices = []
            cancelled_task_ids = []
            for i, delayed_task in enumerate(self._delayed_tasks):
                if delayed_task.future.cancelled():
                    cancelled_task_ids.append(delayed_task.id)
                    should_pop_indices.append(i)
                    continue

                if not self._try_assign_task(delayed_task):
                    continue

                self._put_task(delayed_task)
                should_pop_indices.append(i)

            for i in reversed(should_pop_indices):
                del self._delayed_tasks[i]

            for task_id in cancelled_task_ids:
                del self._tasks[task_id] 

        if self._torch_gpu_available:
            for task in self._tasks.values():
                self._try_move_to_gpu(task)

    def _sorted_idle_workers(self, exclude:Worker)->Tuple[List[Worker], int]:
        return [], 0

    def _choose_task_device(self, task:Task, res: Resource, dry_run: bool = False) -> Optional[str]:
        from .worker import Worker

        with self._sys_info_lock:
            if Worker.total_working_count() == 0:
                self._sys_info.update()

            need_cpu_cores = self._estimate_need_cpu_cores(task, res)
            if need_cpu_cores > self._sys_info.cpu_cores_free:
                if not dry_run:
                    task.device = None
                return None

            need_cpu_mem = res.cpu_mem
            if need_cpu_mem > max(0, self._sys_info.cpu_mem_free):
                if hasattr(task.worker, "cached_rss"):
                    idle_workers, total_hold_mem = self._sorted_idle_workers(exclude=task.worker)
                    if need_cpu_mem > self._sys_info.cpu_mem_free + total_hold_mem:
                        if not dry_run:
                            task.device = None
                        return None
                    
                    if not dry_run:
                        for idle_worker in idle_workers:
                            self._sys_info.cpu_mem_free += idle_worker.cached_rss
                            idle_worker.stop(wait=False, clear=True)
                            if need_cpu_mem <= self._sys_info.cpu_mem_free:
                                break
                else:
                    if not dry_run:
                        task.device = None
                    return None

            if res.gpu_cores == 0 and res.gpu_mem == 0:
                if not dry_run:
                    task.device = "cpu"
                return "cpu"

            if not self._torch_gpu_available:
                if not dry_run:
                    task.device = None
                return None

            gpus = self._sys_info.gpu_infos
            if not gpus:
                if not dry_run:
                    task.device = None
                return None

            best_gpu = None
            for gpu in gpus:
                need_gpu_cores:int = self._estimate_need_gpu_cores(task, gpu.id, res)
                if gpu.mem_free >= res.gpu_mem and gpu.n_cores_free >= need_gpu_cores:
                    if best_gpu is None or gpu.n_cores_free > best_gpu.n_cores_free:
                        best_gpu = gpu

        if best_gpu is None:
            if not dry_run:
                task.device = "cpu"
            return "cpu"

        if not dry_run:
            task.device = best_gpu.device
        return best_gpu.device

    def _choose_task_worker(self, task:Task, res: Resource) -> Optional[Worker]:
        best_worker:Optional[Worker] = None
        task.modules_overlap_ratio = 0.0
        max_overlap_size = 0.0
        for worker in self._workers:
            if worker.is_working:
                continue

            if not self._need_module_deps:
                task.worker = worker
                return worker

            if res.cpu_mem == 0:
                task.modules_overlap_ratio = 1.0
                task.worker = worker
                return worker

            current_overlap_ratio = worker.overlap_modules_ratio(task)
            if hasattr(task.worker, "cached_rss"):
                current_overlap_size = current_overlap_ratio * worker.cached_rss
                if best_worker is None or current_overlap_size > max_overlap_size:
                    task.modules_overlap_ratio = current_overlap_ratio
                    max_overlap_size = current_overlap_size
                    best_worker = worker
            else:
                if best_worker is None or current_overlap_ratio > task.modules_overlap_ratio:
                    task.modules_overlap_ratio = current_overlap_ratio
                    best_worker = worker
            
        if best_worker is not None:
            task.worker = best_worker
            return best_worker
            
        task.modules_overlap_ratio = 0.0
        if len(self._workers) < self._max_workers:
            task.worker = self._add_worker()
        else:
            task.worker = None

        return task.worker

    @abstractmethod
    def _take_resource(self, task:Task)->None:
        pass

    @abstractmethod
    def _release_resource(self, task:Task)->None:
        pass

    @abstractmethod
    def _estimate_need_gpu_cores(self, task:Task, gpu_id:int, res: Resource) -> float:
        pass

    @abstractmethod
    def _estimate_need_cpu_cores(self, task:Task, res: Resource) -> float:
        pass

    @abstractmethod
    def _put_task(self, task:Task)->None:
        pass

    def _try_move_to_gpu(self, task:Task)->None:
        gpu_res = task.gpu_mode_res
        if (
            task.device is None or
            task.device != "cpu" or
            gpu_res.gpu_cores == 0 or
            task.worker is None or
            not task.worker.is_working
        ):
            return
        
        with self._sys_info_lock:
            gpus = self._sys_info.gpu_infos
            if not gpus:
                return

            best_gpu = None
            need_best_gpu_cores:int = 0
            for gpu in gpus:
                need_gpu_cores:int = self._estimate_need_gpu_cores(task, gpu.id, gpu_res)
                if gpu.mem_free >= gpu_res.gpu_mem and gpu.n_cores_free >= need_gpu_cores:
                    if best_gpu is None or gpu.n_cores_free > best_gpu.n_cores_free:
                        best_gpu = gpu
                        need_best_gpu_cores = need_gpu_cores

            if best_gpu is None:
                return

            worker:Worker = task.worker
            worker.change_device(best_gpu.device)
            task.device = best_gpu.device
            best_gpu.n_cores_free -= need_best_gpu_cores
            best_gpu.mem_free -= gpu_res.gpu_mem

    @abstractmethod
    def _add_worker(self)->Worker:
        pass