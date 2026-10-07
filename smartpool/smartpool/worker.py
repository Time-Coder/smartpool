from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Tuple, Type, Union

if TYPE_CHECKING:
    import multiprocessing as mp

    from .device import Device
    from .pool import Pool
    from .task import Task
    from .utils import QueueLike


class Worker(ABC):

    _total_working_count_lock:threading.Lock = threading.Lock()
    _total_working_count:int = 0
    _last_all_idle_at:Optional[float] = None

    def __init__(
        self, pool:Pool,
        task_queue_cls:Type[QueueLike]=None,
        task_queue_args:Optional[Tuple[Any, ...]]=None,
        task_queue_kwargs:Optional[Dict[str, Any]]=None
    ):
        self.index: int = len(pool._workers)
        self.pool: Pool = pool
        self._is_working: bool = False
        self.imported_modules: Set[str] = set()
        self.n_finished_tasks: int = 0
        self.task_queue_cls: Type[QueueLike] = task_queue_cls
        self.task_queue_args: Optional[Tuple[Any, ...]] = task_queue_args
        self.task_queue_kwargs: Optional[Dict[str, Any]] = task_queue_kwargs
        self._task_queue: Optional[QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]]] = None
        self.executor: Optional[Union[mp.Process, threading.Thread]] = None
        self._memory_taken: bool = False
        self._memory_taken_amount: int = 0
        self._current_task_id: Optional[str] = None

    @property
    def task_queue(self)->Optional[QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]]]:
        if self._task_queue is not None:
            return self._task_queue

        if self.task_queue_cls is not None:
            if self.task_queue_args is None:
                self.task_queue_args = ()

            if self.task_queue_kwargs is None:
                self.task_queue_kwargs = {}

            self._task_queue: Optional[QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]]] = self.task_queue_cls(*self.task_queue_args, **self.task_queue_kwargs)

        return self._task_queue

    def add_task(self, task: Task)->None:
        self.start()
        task.future.set_running_or_notify_cancel()
        self._current_task_id = task.id
        self.task_queue.put(task.info())

    def is_alive(self)->bool:
        """Whether the executor backing this worker is still running.

        The pool polls this so a worker that died mid-task (segfault, OOM kill,
        unpicklable task payload) fails its future instead of leaving it pending
        forever. Thread-backed workers run user code in-process, so they cannot
        die this way and are always reported alive.
        """
        return True

    def abandon(self)->None:
        """Drop a worker whose executor died, so the next task gets a fresh one.

        Releases the worker's memory reservation and clears the dead executor.
        The caller is responsible for failing the task the worker was running.
        """
        self._release_worker_memory()
        self._clear()

    def _working_changed_hook(self):
        pass

    @property
    def is_working(self)->bool:
        return self._is_working

    @is_working.setter
    def is_working(self, is_working:bool)->None:
        if self._is_working == is_working:
            return

        self._is_working = is_working
        self._working_changed_hook()

        if is_working:
            with Worker._total_working_count_lock:
                Worker._total_working_count += 1
                self.pool._workers_working_count += 1
        else:
            with Worker._total_working_count_lock:
                Worker._total_working_count -= 1
                self.pool._workers_working_count -= 1
                if Worker._total_working_count == 0:
                    Worker._last_all_idle_at = time.monotonic()

    @staticmethod
    def total_working_count()->int:
        with Worker._total_working_count_lock:
            return Worker._total_working_count

    @staticmethod
    def recently_all_idle(seconds:float=1.0)->bool:
        with Worker._total_working_count_lock:
            ended_at = Worker._last_all_idle_at
        return ended_at is not None and time.monotonic() - ended_at < seconds

    def _clear(self)->None:
        self.executor = None
        self._current_task_id = None
        # Go through the property so the global and per-pool working counts stay
        # consistent if a working worker is ever cleared.
        self.is_working = False

    def change_device(self, device:Device, task_id:Optional[str]=None)->None:
        pass

    @property
    def memory(self) -> int:
        return 0

    def _take_worker_memory(self) -> None:
        if self._memory_taken:
            return
        self._memory_taken = True
        # Remember the exact amount charged. Refunding a freshly sampled RSS
        # instead would leak the difference whenever the worker grew after it was
        # charged, which inflates cpu_mem_free over a pool's lifetime.
        self._memory_taken_amount = self.memory
        with self.pool._sys_info_lock:
            self.pool._sys_info.cpu_mem_free -= self._memory_taken_amount

    def _release_worker_memory(self) -> None:
        if not self._memory_taken:
            return
        self._memory_taken = False
        amount = self._memory_taken_amount
        self._memory_taken_amount = 0
        with self.pool._sys_info_lock:
            self.pool._sys_info.cpu_mem_free += amount

    @abstractmethod
    def start(self):
        pass

    def stop(self, wait:bool=False, clear:bool=True)->None:
        if self.executor is None:
            return

        self._release_worker_memory()
        self.task_queue.put(None)
        if wait:
            self.join()

        elif clear:
            self._clear()

    @abstractmethod
    def join(self)->None:
        pass

    def overlap_modules_ratio(self, task:Task)->float:
        if not self.imported_modules:
            return 0

        return len(self.imported_modules & task.module_deps) / len(self.imported_modules)

    @staticmethod
    def _changing_device(cmd_queue:QueueLike, current_thread_id, state:dict):
        """Apply device-change commands that belong to the task now running.

        Commands carry the task id they were issued for. Without that check, a
        command addressed to a task that has already finished can be applied to
        whichever task the worker picks up next, so a task can begin on a device
        other than the one it was admitted on. A bare device string is still
        accepted and applies to whatever task is current.
        """
        from .utils import _set_best_device

        while True:
            item = cmd_queue.get()
            if item is None:
                break

            if isinstance(item, tuple):
                task_id, device = item
            else:
                task_id, device = None, item

            if task_id is not None and state.get("task_id") != task_id:
                # Stale command, addressed to a task that already finished.
                continue

            _set_best_device(device, current_thread_id)

    @staticmethod
    def run(
        task_queue:QueueLike[Optional[Tuple[str, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]]]],
        result_queue:QueueLike[Tuple[str, bool, Any]],
        change_device_cmd_queue:Optional[QueueLike[Optional[str]]],
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]]
    ):
        from .utils import _set_best_device


        if initializer is not None:
            if initkwargs is None:
                initkwargs = {}

            initializer(*initargs, **initkwargs)

        device_state:dict = {"task_id": None}

        if change_device_cmd_queue is not None:
            import threading

            current_thread_id = threading.get_ident()
            change_device_thread = threading.Thread(target=Worker._changing_device, args=(change_device_cmd_queue, current_thread_id, device_state), name="changing_device")
            change_device_thread.start()

        while True:
            task = task_queue.get()
            if task is None:
                break

            task_id, task_device, func, args, kwargs = task
            # Publish the task id before setting the device, so a command issued
            # for this task is applied and one issued for an earlier task is
            # discarded instead of leaking into this task.
            device_state["task_id"] = task_id
            _set_best_device(task_device)

            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = e
                success = False

            device_state["task_id"] = None
            result_queue.put((task_id, success, result))

        if change_device_cmd_queue is not None and change_device_thread.is_alive():
            change_device_cmd_queue.put(None)
            change_device_thread.join()
