from __future__ import annotations

import weakref
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from .resource import Resource

if TYPE_CHECKING:
    import threading
    
    from .futures import Future
    from .gpuinfo import GPUInfoSnapshot
    from .sysinfo import SysInfo
    from .task import Task
    from .utils import QueueLike
    from .worker import Worker


class Pool(ABC):

    _sys_info_lock: Optional[threading.Lock] = None
    _sys_info: Optional[SysInfo] = None
    _updating_sysinfo_thread: Optional[threading.Thread] = None
    _instances: weakref.WeakSet[Pool] = weakref.WeakSet()
    _instance_config: Dict[Type[Pool], Dict[str, Tuple[Tuple[Any, ...], Dict[str, Any]]]] = defaultdict(dict)
    _instance_dict: Dict[Type[Pool], Dict[str, Pool]] = defaultdict(dict)

    def __init__(
        self, max_workers: int,

        initializer: Optional[Callable[..., Any]],
        initargs: Tuple[Any, ...],
        initkwargs: Optional[Dict[str, Any]],

        result_queue_cls: Type[QueueLike],
        result_queue_args: Tuple[Any, ...]=(),
        result_queue_kwargs: Dict[str, Any]=None,

        *,

        max_tasks_per_child: Optional[int]=None,
        worker_cls: Type[Worker],
        use_torch: bool = False,
        use_onnx: bool = False,
        need_module_deps: bool = False
    ):
        self._init_sys_info()

        import os
        import threading

        if use_torch:
            import torch
            self._torch_gpu_available = False
            if torch.cuda.is_available() or getattr(torch, "hip", None) and torch.hip.is_available() or getattr(torch, "xpu", None) and torch.xpu.is_available():
                self._torch_gpu_available = True
        else:
            self._torch_gpu_available = False

        if not max_workers:
            max_workers = os.cpu_count()

        self._max_tasks_per_child: int = max_tasks_per_child
        self._need_module_deps: bool = need_module_deps
        self._use_torch: bool = use_torch
        self._max_workers: int = max_workers
        self._initializer: Optional[Callable[..., Any]] = initializer
        self._initargs: Tuple[Any, ...] = initargs
        self._initkwargs: Optional[Dict[str, Any]] = initkwargs
        self._workers_working_count: int = 0
        self._worker_cls: Type[Worker] = worker_cls
        self._workers: List[Worker] = []
        self._tasks: Dict[str, Task] = {}
        self._delayed_tasks: List[Task] = []
        self._lock: threading.Lock = threading.Lock()
        self._shutdown: bool = False
        self._result_thread: Optional[threading.Thread] = None
        self._use_onnx: bool = use_onnx

        if result_queue_cls is not None:
            if result_queue_kwargs is None:
                result_queue_kwargs = {}

            self._result_queue:Optional[QueueLike[Tuple[str, bool, Any]]] = result_queue_cls(*result_queue_args, **result_queue_kwargs)
        else:
            self._result_queue:Optional[QueueLike[Tuple[str, bool, Any]]] = None

        Pool._instances.add(self)

    def submit(
        self, func: Optional[Callable[..., Any]],
        args: Optional[Tuple[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        use_torch: Optional[bool] = None,
        device_changeable: bool = False
    )->Future:
        if self._shutdown:
            raise RuntimeError("cannot submit after shutdown")
        
        if args is None:
            args = []

        if kwargs is None:
            kwargs = {}

        self._check_args(func, args, kwargs)

        if cpu_mode_res is None:
            cpu_mode_res = Resource(cpu_cores_in_python=1)

        if gpu_mode_res is None:
            gpu_mode_res = cpu_mode_res

        if use_torch is None:
            use_torch = self._use_torch

        from .task import Task

        task = Task(
            pool=self,
            func=func,
            args=args,
            kwargs=kwargs,
            cpu_mode_res=cpu_mode_res,
            gpu_mode_res=gpu_mode_res,
            use_torch=use_torch,
            device_changeable=device_changeable,
            calculate_module_deps=self._need_module_deps
        )
        self._validate_resource_feasibility(task)

        with self._lock:
            self._tasks[task.id] = task
            return self._submit(task, append_to_delay=True)

    def _submit(self, task: Task, append_to_delay: bool) -> Future:
        if task.future.done():
            if task.id in self._tasks:
                del self._tasks[task.id]
                
            return task.future
        
        if task.worker is not None:
            return task.future
        
        if not self._try_assign_task(task):            
            if append_to_delay:
                self._delayed_tasks.append(task)

            return task.future

        self._put_task(task)
        try:
            self._delayed_tasks.remove(task)
        except ValueError:
            pass

        if self._result_queue is not None and self._result_thread is None:
            import threading
            self._result_thread = threading.Thread(target=self._collecting_result, daemon=True, name="collecting_result")
            self._result_thread.start()

        return task.future

    @classmethod
    def config(cls, pool_name:str, *args, **kwargs):
        Pool._instance_config[cls][pool_name] = (args, kwargs)

    @classmethod
    def instance(cls, pool_name: str):
        if pool_name in Pool._instance_dict[cls]:
            return Pool._instance_dict[cls][pool_name]

        if pool_name not in Pool._instance_config[cls]:
            args = ()
            kwargs = {}
        else:
            args, kwargs = Pool._instance_config[cls][pool_name]

        Pool._instance_dict[cls][pool_name] = cls(*args, **kwargs)
        return Pool._instance_dict[cls][pool_name]

    @classmethod
    def task(cls, func:Callable=None, *, pool_name:str="default", **submit_kwargs):

        def decorator(func):
            import functools
            from .task_wrapper import TaskWrapper

            wrapper = functools.wraps(func)(TaskWrapper(cls, func, pool_name, submit_kwargs))
            
            return wrapper
        
        if func is not None:
            return decorator(func)
        
        return decorator

    def task_count_on_device(self, device:str)->int:
        with self._lock:
            return sum(1 for task in self._tasks.values() if task.device == device)

    def task_count_with_provider(self, provider:Union[str, Tuple[str, Dict[str, Any]]])->int:
        with self._lock:
            count: int = 0
            for task in self._tasks.values():
                task_provider = task.onnx_provider
                if task_provider is None:
                    continue

                task_provider_name: str = task_provider[0]
                task_provider_device_id: int = 0
                if "device_id" in task_provider[1]:
                    task_provider_device_id: int = task_provider[1]["device_id"]

                if isinstance(provider, str):
                    if provider == task_provider_name:
                        count += 1
                else:
                    user_provider_name: str = provider[0]
                    user_device_id: int = 0
                    if "device_id" in provider[1]:
                        user_device_id: int = provider[1]["device_id"]

                    if user_provider_name == task_provider_name and user_device_id == task_provider_device_id:
                        count += 1

            return count

    def _check_args(self, func:Optional[Callable[..., Any]], args:Tuple[Any], kwargs:Dict[str, Any]) -> None:
        if func is None:
            return

        import inspect

        signature: inspect.Signature = inspect.signature(func)
        try:
            signature.bind(*args, **kwargs)
        except TypeError as e:
            formatted_msg = f"{func.__name__}() {str(e)}"
            raise TypeError(formatted_msg) from None

    def _try_assign_task(self, task: Task) -> bool:
        if not task.ready_to_run:
            return False
        
        gpu_res = task.gpu_mode_res
        if gpu_res.gpu_cores > 0 or gpu_res.gpu_mem > 0:
            self._choose_task_worker(task, gpu_res)
            if task.worker is not None:
                devices, should_kill_workers = self._choose_task_device(task, "gpu", kill_workers=False)
                if devices:
                    for idle_worker in should_kill_workers:
                        self._sys_info.cpu_mem_free += idle_worker.cached_rss
                        idle_worker.stop(wait=False, clear=True)

                    return True

                task.worker = None
                task.modules_overlap_ratio = 0.0

        cpu_res = task.cpu_mode_res
        self._choose_task_worker(task, cpu_res)
        if task.worker is not None:
            devices, _ = self._choose_task_device(task, "cpu", kill_workers=True)
            if devices:
                return True

        task.worker = None
        task.modules_overlap_ratio = 0.0
        return False

    def _validate_resource_feasibility(self, task: Task) -> None:
        cpu_res = task.cpu_mode_res
        gpu_res = task.gpu_mode_res

        with self._sys_info_lock:
            if (
                cpu_res.cpu_cores <= self._sys_info.cpu_cores_total
                and cpu_res.cpu_mem <= self._sys_info.cpu_mem_total
            ):
                return

            if (
                (gpu_res.gpu_cores > 0 or gpu_res.gpu_mem > 0) and (
                    gpu_res.cpu_cores <= self._sys_info.cpu_cores_total
                    and gpu_res.cpu_mem <= self._sys_info.cpu_mem_total
                )
            ):
                gpu_infos = task.filter_gpu_infos(self._sys_info.gpu_infos)
                for gpu in gpu_infos:
                    if (
                        gpu_res.gpu_cores <= gpu.n_cores
                        and gpu_res.gpu_mem <= gpu.mem_total
                    ):
                        return

            raise ValueError("task resources needed exceed system capacity")

    def _init_sys_info(self)->None:
        if Pool._sys_info is not None:
            return

        import threading

        from .sysinfo import SysInfo

        Pool._sys_info = SysInfo()
        Pool._sys_info_lock = threading.Lock()
        Pool._updating_sysinfo_thread = threading.Thread(target=Pool._updating_sysinfo, daemon=True)
        Pool._updating_sysinfo_thread.start()

    @staticmethod
    def _updating_sysinfo()->None:
        from .worker import Worker

        while True:
            Pool._sys_info.update_cpu_percent()

            if Worker.total_working_count() > 0:
                continue

            for pool in Pool._instances:
                if pool._shutdown or not pool._delayed_tasks:
                    continue

                with pool._lock:
                    pool._postprocess_after_task_done()

    @staticmethod
    def _result_iterator(futures:List[Future], end_time:Optional[float]):
        import time
        from concurrent.futures._base import _result_or_cancel

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
        cpu_mode_res:Optional[Union[Resource, Iterable[Resource]]] = None,
        gpu_mode_res:Optional[Union[Resource, Iterable[Resource]]] = None,
        timeout:Optional[Union[float, int]]=None,
        chunksize:int=1
    )->Iterable[Any]:
        import itertools
        import time
        from collections.abc import Iterable
        from concurrent.futures.process import (
            _chain_from_iterable_of_lists,
            _process_chunk,
        )
        from functools import partial

        from .utils import batched

        if cpu_mode_res is None:
            cpu_mode_res = Resource()

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
        cpu_mode_res:Optional[Union[Resource, Iterable[Resource]]] = None,
        gpu_mode_res:Optional[Union[Resource, Iterable[Resource]]] = None,
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

            if self._result_thread is not None and self._result_thread.is_alive() and self._result_queue is not None:
                self._result_queue.put(None)

            if cancel_futures:
                for task in self._tasks.values():
                    task.future.cancel()

            if wait:
                for worker in self._workers:
                    worker.join()

                if self._result_thread is not None and self._result_thread.is_alive():
                    self._result_thread.join()

    def __enter__(self)->Pool:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb)->None:
        self.shutdown(wait=True)

    def _on_task_done(self, task_id:str, success:bool, result:Any)->None:
        with self._lock:
            task = self._tasks.pop(task_id)
            worker: Worker = task.worker
            worker.is_working = False
            worker.n_finished_tasks += 1
            if self._max_tasks_per_child is not None and worker.n_finished_tasks >= self._max_tasks_per_child:
                worker.stop(wait=False, clear=True)

            self._release_resource(task)
            self._postprocess_after_task_done()

        try:
            if success:
                task.future.set_result(result)
            else:
                task.future.set_exception(result)
        except Exception:
            pass

        for pool in Pool._instances:
            if pool._shutdown or pool is self or pool._workers_working_count > 0 or not pool._delayed_tasks:
                continue

            with pool._lock:
                pool._postprocess_after_task_done()

    def _collecting_result(self)->None:
        while not self._shutdown:
            result_tuple = self._result_queue.get()
            if result_tuple is None:
                break

            task_id, success, result = result_tuple
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

        if not self._use_onnx:
            for task in self._tasks.values():
                self._try_move_to_gpu(task)

    def _sorted_idle_workers(self, exclude:Worker)->Tuple[List[Worker], int]:
        return [], 0

    def _choose_task_device(self, task:Task, mode:str, kill_workers: bool) -> Tuple[List[Union[GPUInfoSnapshot], str], List[Worker]]:
        from .worker import Worker

        res = (task.cpu_mode_res if mode == "cpu" else task.gpu_mode_res)

        with self._sys_info_lock:
            if Worker.total_working_count() == 0:
                self._sys_info.update()

            cpu_cores_needed = self._estimate_cpu_cores_needed(res)
            if cpu_cores_needed > self._sys_info.cpu_cores_free:
                task.device = None
                task.gpu_index = -1
                task.dml_id = -1
                return [], []

            should_kill_workers = []
            cpu_mem_needed = res.cpu_mem
            if cpu_mem_needed > max(0, self._sys_info.cpu_mem_free):
                if hasattr(task.worker, "cached_rss"):
                    idle_workers, total_hold_mem = self._sorted_idle_workers(exclude=task.worker)
                    if cpu_mem_needed > self._sys_info.cpu_mem_free + total_hold_mem:
                        task.device = None
                        task.gpu_index = -1
                        task.dml_id = -1
                        return [], []

                    if kill_workers:
                        for idle_worker in idle_workers:
                            self._sys_info.cpu_mem_free += idle_worker.cached_rss
                            idle_worker.stop(wait=False, clear=True)
                            if cpu_mem_needed <= self._sys_info.cpu_mem_free:
                                break
                    else:
                        temp_cpu_mem_free = self._sys_info.cpu_mem_free
                        for idle_worker in idle_workers:
                            temp_cpu_mem_free += idle_worker.cached_rss
                            should_kill_workers.append(idle_worker)
                            if cpu_mem_needed <= temp_cpu_mem_free:
                                break
                else:
                    task.device = None
                    task.gpu_index = -1
                    task.dml_id = -1
                    return [], []

            if mode == "cpu":
                task.device = "cpu"
                task.gpu_index = -1
                task.dml_id = -1
                return ["cpu"], should_kill_workers

            gpus = task.filter_gpu_infos(self._sys_info.gpu_infos)
            available_gpus = []
            for gpu in gpus:
                if gpu.mem_free < res.gpu_mem:
                    continue

                if gpu.n_cores_free < res.gpu_cores:
                    continue

                if self._use_onnx and not gpu.onnx_provider(task.user_providers):
                    continue

                available_gpus.append(gpu)

        if not available_gpus:
            task.device = None
            task.gpu_index = -1
            task.dml_id = -1
            return [], []

        available_gpus.sort(key=lambda gpu: gpu.n_cores_free, reverse=True)
        best_gpu: GPUInfoSnapshot = available_gpus[0]
        task.device = best_gpu.device
        task.gpu_index = best_gpu.index
        task.dml_id = best_gpu.dml_id
        return available_gpus, should_kill_workers

    def _choose_task_worker(self, task: Task, res: Resource) -> Optional[Worker]:
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
    def _estimate_cpu_cores_needed(self, res: Resource) -> float:
        pass

    @abstractmethod
    def _put_task(self, task:Task)->None:
        pass

    def _try_move_to_gpu(self, task:Task)->None:
        gpu_res = task.gpu_mode_res
        cpu_res = task.cpu_mode_res
        if (
            not task.device_changeable or
            task.device is None or
            task.device != "cpu" or
            gpu_res.gpu_cores == 0 or
            task.worker is None or
            not task.worker.is_working
        ):
            return

        extra_cpu_cores_needed = gpu_res.cpu_cores - cpu_res.cpu_cores
        extra_cpu_mem_needed = gpu_res.cpu_mem - cpu_res.cpu_mem
        with self._sys_info_lock:

            if extra_cpu_cores_needed > 0 and self._sys_info.cpu_cores_free < extra_cpu_cores_needed:
                return

            if extra_cpu_mem_needed > 0 and self._sys_info.cpu_mem_free < extra_cpu_mem_needed:
                return

            gpus = task.filter_gpu_infos(self._sys_info.gpu_infos)
            best_gpu = None
            for gpu in gpus:
                if gpu.mem_free < gpu_res.gpu_mem:
                    continue

                if gpu.n_cores_free < gpu_res.gpu_cores:
                    continue

                if best_gpu is None or gpu.n_cores_free > best_gpu.n_cores_free:
                    best_gpu = gpu

            if best_gpu is None:
                return

            worker:Worker = task.worker
            worker.change_device(best_gpu.device)
            task.device = best_gpu.device
            task.gpu_index = best_gpu.index
            task.dml_id = best_gpu.dml_id
            best_gpu.n_cores_free -= gpu_res.gpu_cores
            best_gpu.mem_free -= gpu_res.gpu_mem
            self._sys_info.cpu_cores_free -= extra_cpu_cores_needed
            self._sys_info.cpu_mem_free -= extra_cpu_mem_needed

    def _add_worker(self)->Worker:
        worker = self._worker_cls(self)
        self._workers.append(worker)
        return worker
