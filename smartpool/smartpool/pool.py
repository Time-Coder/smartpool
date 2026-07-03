from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
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

from .device import Device
from .resource import Resource

if TYPE_CHECKING:
    import threading

    from .chunk_task import ChunkTask
    from .futures import Future
    from .gpuinfo import GPUInfoSnapshot
    from .infer_session_pool.infer_session import InferSession
    from .sysinfo import SysInfo
    from .task import Task
    from .utils import QueueLike
    from .worker import Worker

class Pool(ABC):

    _sys_info_lock: Optional[threading.Lock] = None
    _sys_info: Optional[SysInfo] = None
    _housekeeping_thread: Optional[threading.Thread] = None
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
        chunk_timeout: float = 0.1,
        need_module_deps: bool = False
    ):
        self._init_sys_info()

        import threading

        if use_torch:
            import torch
            self._torch_gpu_available = False
            if torch.cuda.is_available() or getattr(torch, "hip", None) and torch.hip.is_available() or getattr(torch, "xpu", None) and torch.xpu.is_available():
                self._torch_gpu_available = True
        else:
            self._torch_gpu_available = False

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
        self._not_ready_tasks: Dict[str, Task] = {}
        self._delayed_tasks: Dict[str, Task] = {}
        self._lock: threading.RLock = threading.RLock()
        self._shutdown: bool = False
        self._chunk_timeout: float = chunk_timeout
        self._result_thread: Optional[threading.Thread] = None
        self._chunk_tasks: Dict[Tuple[Callable[..., Any], int], ChunkTask] = {}
        self._flush_chunk_thread: Optional[threading.Thread] = None
        self._flush_chunk_queue: Optional[QueueLike[Optional[ChunkTask]]] = None
        self._result_queue:Optional[QueueLike[Tuple[str, bool, Any]]] = None
        if result_queue_cls is not None:
            if result_queue_kwargs is None:
                result_queue_kwargs = {}

            self._result_queue:Optional[QueueLike[Tuple[str, bool, Any]]] = result_queue_cls(*result_queue_args, **result_queue_kwargs)

        Pool._instances.add(self)

    def submit(
        self, func: Optional[Callable[..., Any]],
        args: Optional[Tuple[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        check_args: bool = True,
        chunksize: int = 1,
        use_torch: Optional[bool] = None,
        device_changeable: bool = False
    )->Future:
        if self._shutdown:
            raise RuntimeError("cannot submit after shutdown")

        if args is None:
            args = []

        if kwargs is None:
            kwargs = {}

        if check_args:
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
            check_args=check_args,
            chunksize=chunksize,
            use_torch=use_torch,
            device_changeable=device_changeable,
            calculate_module_deps=self._need_module_deps
        )
        if check_args:
            self._validate_resource_feasibility(task)

        self._submit_or_chunk(task)
        return task.future

    @property
    def chunk_timeout(self)->float:
        return self._chunk_timeout

    @chunk_timeout.setter
    def chunk_timeout(self, timeout:float)->None:
        self._chunk_timeout = timeout

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def _flush(self) -> None:
        for chunk_task in list(self._chunk_tasks.values()):
            chunk_task.submit()

    def _flushing_chunk(self) -> None:
        import time
        while not self._shutdown:
            chunk_task: Optional[ChunkTask] = self._flush_chunk_queue.get()
            if chunk_task is None:
                break

            if chunk_task.submitted:
                continue

            time.sleep(self._chunk_timeout)

            if chunk_task.submitted:
                continue

            chunk_task.submit()

    def _put_in_chunk(self, task: Task) -> Future:
        import threading
        from queue import Queue

        from .chunk_task import ChunkTask

        key = (task.func, task.chunksize)
        if self._flush_chunk_thread is None:
            self._flush_chunk_queue = Queue()
            self._flush_chunk_thread = threading.Thread(target=self._flushing_chunk, daemon=True, name="flushing_chunk")
            self._flush_chunk_thread.start()

        if key not in self._chunk_tasks:
            chunk_task: ChunkTask = ChunkTask(self, task.func, task.chunksize)
            self._chunk_tasks[key] = chunk_task
            self._flush_chunk_queue.put(chunk_task)
        else:
            chunk_task: ChunkTask = self._chunk_tasks[key]

        chunk_task.add_task(task)

        return task.future

    def _submit(self, task: Task) -> Future:
        assert task.ready_to_run

        self._tasks[task.id] = task

        if task.id in self._not_ready_tasks:
            del self._not_ready_tasks[task.id]

        if task.future.done():
            if task.id in self._tasks:
                del self._tasks[task.id]

            if task.id in self._delayed_tasks:
                del self._delayed_tasks[task.id]

            return task.future

        if task.worker is not None:
            return task.future

        if not self._try_assign_task(task):
            self._delayed_tasks[task.id] = task
            return task.future

        self._put_task(task)
        if task.id in self._delayed_tasks:
            del self._delayed_tasks[task.id]

        if self._result_queue is not None and self._result_thread is None:
            import threading
            self._result_thread = threading.Thread(target=self._collecting_result, daemon=True, name="collecting_result")
            self._result_thread.start()

        return task.future

    def _submit_or_chunk(self, task: Task) -> Future:
        with self._lock:
            if not task.ready_to_run:
                self._not_ready_tasks[task.id] = task
                return

            if task.id in self._not_ready_tasks:
                del self._not_ready_tasks[task.id]

            if task.chunksize <= 1:
                return self._submit(task)
            else:
                return self._put_in_chunk(task)

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
        if self._workers_working_count >= self._max_workers:
            return False

        gpu_res = task.gpu_mode_res
        if gpu_res.gpu_cores > 0 or gpu_res.gpu_mem > 0:
            self._choose_task_worker(task, gpu_res)
            if task.worker is not None:
                devices, cpu_to_evict, gpu_to_evicts = self._choose_task_device(task, "gpu")
                if devices:
                    for idle_item in cpu_to_evict:
                        idle_item.stop()

                    for gpu_to_evict in gpu_to_evicts:
                        for idle_item in gpu_to_evict:
                            idle_item.stop()

                        break

                    return True

                task.worker = None
                task.modules_overlap_ratio = 0.0

        cpu_res = task.cpu_mode_res
        self._choose_task_worker(task, cpu_res)
        if task.worker is not None:
            devices, cpu_to_evict, gpu_to_evicts = self._choose_task_device(task, "cpu")
            if devices:
                for idle_item in cpu_to_evict:
                    idle_item.stop()

                for gpu_to_evict in gpu_to_evicts:
                    for idle_item in gpu_to_evict:
                        idle_item.stop()

                    break

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
        Pool._housekeeping_thread = threading.Thread(target=Pool._housekeeping, daemon=True)
        Pool._housekeeping_thread.start()

    @staticmethod
    def _housekeeping()->None:
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

    @staticmethod
    def _batch_resource(resources: List[Resource], aggregate_func: Callable[[Iterable[float]], float]) -> Resource:
        return Resource(
            cpu_cores_in_python=aggregate_func(res.cpu_cores_in_python for res in resources),
            cpu_cores_out_of_python=aggregate_func(res.cpu_cores_out_of_python for res in resources),
            cpu_mem=aggregate_func(res.cpu_mem for res in resources),
            gpu_cores=aggregate_func(res.gpu_cores for res in resources),
            gpu_mem=aggregate_func(res.gpu_mem for res in resources),
        )

    def map(
        self, func:Callable[..., Any],
        *iterable:Iterable[Any],
        cpu_mode_res:Optional[Union[Resource, Iterable[Resource]]] = None,
        gpu_mode_res:Optional[Union[Resource, Iterable[Resource]]] = None,
        timeout:Optional[Union[float, int]]=None,
        chunksize:int=1
    )->Iterable[Any]:
        import itertools
        import time
        from collections.abc import Iterable

        if cpu_mode_res is None:
            cpu_mode_res = Resource()

        if not isinstance(cpu_mode_res, Iterable):
            cpu_mode_res = itertools.repeat(cpu_mode_res)

        if gpu_mode_res is None:
            gpu_mode_res = cpu_mode_res
        elif not isinstance(gpu_mode_res, Iterable):
            gpu_mode_res = itertools.repeat(gpu_mode_res)

        end_time = None
        if timeout is not None:
            end_time = timeout + time.monotonic()

        futures:List[Future] = []
        for args, cpu_res, gpu_res in zip(zip(*iterable), cpu_mode_res, gpu_mode_res):
            future = self.submit(
                func, args=args,
                cpu_mode_res=cpu_res,
                gpu_mode_res=gpu_res,
                chunksize=chunksize
            )
            futures.append(future)

        self.flush()

        return Pool._result_iterator(futures, end_time)

    def shutdown(self, wait:bool=True, *, cancel_futures:bool=False)->None:
        with self._lock:
            if self._shutdown:
                return

            self._flush()
            self._shutdown = True

            if self._flush_chunk_thread is not None and self._flush_chunk_thread.is_alive() and self._flush_chunk_queue is not None:
                self._flush_chunk_queue.put(None)

            self._stop_feeding()

            if cancel_futures:
                for task in self._tasks.values():
                    task.future.cancel()

        if wait:
            self._join_feeding()

        with self._lock:
            for worker in self._workers:
                worker.stop(wait=False, clear=False)

        if wait:
            for worker in self._workers:
                worker.join()

            if self._result_thread is not None and self._result_thread.is_alive() and self._result_queue is not None:
                self._result_queue.put(None)

            if self._result_thread is not None and self._result_thread.is_alive():
                self._result_thread.join()

            if self._flush_chunk_thread is not None and self._flush_chunk_thread.is_alive():
                self._flush_chunk_thread.join()

    def _stop_feeding(self) -> None:
        pass

    def _join_feeding(self) -> None:
        pass

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
        while True:
            result_tuple = self._result_queue.get()
            if result_tuple is None:
                break

            task_id, success, result = result_tuple
            self._on_task_done(task_id, success, result)

    def _postprocess_after_task_done(self)->None:
        for task in self._tasks.values():
            self._try_move_to_gpu(task)

        if self._delayed_tasks:
            for task_id in list(self._delayed_tasks):
                delayed_task = self._delayed_tasks.get(task_id)
                if delayed_task is None:
                    continue

                if delayed_task.future.cancelled():
                    del self._delayed_tasks[task_id]
                    if task_id in self._tasks:
                        del self._tasks[task_id]
                    continue

                if not self._try_assign_task(delayed_task):
                    continue

                del self._delayed_tasks[task_id]
                self._put_task(delayed_task)

    def _sorted_idle_workers(self, exclude: Worker) -> Tuple[List[Worker], int]:
        workers: List[Worker] = []
        total_hold_mem:int = 0
        for worker in self._workers:
            if not worker.is_working and worker is not exclude and worker.memory > 0:
                workers.append(worker)
                total_hold_mem += worker.memory

        workers.sort(key=lambda worker: worker.memory, reverse=True)
        return workers, total_hold_mem

    def _choose_task_device(self, task: Task, mode: str) -> Tuple[List[Device], List[Union[Worker, InferSession]], List[List[Union[Worker, InferSession]]]]:
        from .worker import Worker

        res = (task.cpu_mode_res if mode == "cpu" else task.gpu_mode_res)

        with self._sys_info_lock:
            if Worker.total_working_count() == 0:
                self._sys_info.update()

            cpu_cores_needed = self._estimate_cpu_cores_needed(res)
            if cpu_cores_needed > self._sys_info.cpu_cores_free:
                task.device = None
                return [], [], []

            cpu_to_evict = []
            cpu_mem_needed = res.cpu_mem
            if cpu_mem_needed > max(0, self._sys_info.cpu_mem_free):
                cpu_to_evict = self._reclaim_cpu_memory(task, res)
                if not cpu_to_evict:
                    task.device = None
                    return [], [], []

            if mode == "cpu":
                task.device = Device("cpu")
                return [task.device], cpu_to_evict, []

            gpus = task.filter_gpu_infos(self._sys_info.gpu_infos)
            available_gpus: List[GPUInfoSnapshot] = []
            for gpu in gpus:
                if gpu.n_cores_free < res.gpu_cores:
                    continue

                if gpu.mem_free < res.gpu_mem:
                    continue

                available_gpus.append(gpu)

            gpu_to_evicts = []
            if not available_gpus:
                for gpu in gpus:
                    if res.gpu_cores > gpu.n_cores_free:
                        continue

                    to_evict = []
                    if res.gpu_mem > gpu.mem_free:
                        to_evict = self._reclaim_gpu_memory(task, res, gpu, cpu_to_evict)
                        if not to_evict:
                            continue

                    gpu_to_evicts.append(to_evict)
                    available_gpus.append(gpu)

            if available_gpus:
                available_gpus.sort(key=lambda gpu: gpu.n_cores_free, reverse=True)
                best_gpu = available_gpus[0]
                task.device = best_gpu.device
                return [gpu.device for gpu in available_gpus], cpu_to_evict, gpu_to_evicts

            task.device = None
            return [], [], []

    def _reclaim_cpu_memory(self, task: Task, res: Resource) -> List:
        if not hasattr(task.worker, "cached_rss"):
            return []

        idle_workers, total_hold_mem = self._sorted_idle_workers(exclude=task.worker)
        if res.cpu_mem > self._sys_info.cpu_mem_free + total_hold_mem:
            return []

        to_evict = []
        cpu_mem_needed = res.cpu_mem
        freed_cpu_mem = 0
        for idle_worker in idle_workers:
            to_evict.append(idle_worker)
            freed_cpu_mem += idle_worker.memory
            if cpu_mem_needed <= max(0, self._sys_info.cpu_mem_free) + freed_cpu_mem:
                break

        return to_evict

    def _reclaim_gpu_memory(self, task: Task, res: Resource, gpu: GPUInfoSnapshot, reclaim_items: List)->List:
        return []

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

    def _estimate_cpu_cores_needed(self, res: Resource) -> float:
        return res.cpu_cores

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
            best_gpu.n_cores_free -= gpu_res.gpu_cores
            best_gpu.mem_free -= gpu_res.gpu_mem
            self._sys_info.cpu_cores_free -= extra_cpu_cores_needed
            self._sys_info.cpu_mem_free -= extra_cpu_mem_needed

    def _add_worker(self)->Worker:
        worker = self._worker_cls(self)
        self._workers.append(worker)
        return worker
