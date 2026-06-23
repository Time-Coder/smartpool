from __future__ import annotations

import sys
import uuid
import threading
from functools import partial
from concurrent.futures import Future as _BaseFuture
from .futures import Future
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from .gpuinfo import GPUInfoSnapshot
    from .resource import Resource
    from .worker import Worker
    from .pool import Pool


class Task:

    _torch_gpu_backend: Optional[str] = None

    def __init__(
        self, pool: Pool, func: Union[Callable[..., Any], str], args: Tuple[Any, ...], kwargs: Dict[str, Any],
        cpu_mode_res: Resource, gpu_mode_res: Resource, check_args: bool,
        use_torch: bool, device_changeable: bool, calculate_module_deps: bool
    ):
        self.id: str = str(uuid.uuid4())
        self.func: Union[Callable[..., Any], str] = func

        if args is None:
            args = ()

        if kwargs is None:
            kwargs = {}

        self.args: List[Any] = list(args)
        self.kwargs: Dict[str, Any] = kwargs
        self.cpu_mode_res: Resource = cpu_mode_res
        self.gpu_mode_res: Resource = gpu_mode_res
        self.check_args: bool = check_args
        self.use_torch: bool = use_torch
        self.device_changeable: bool = device_changeable
        self.estimated_need_cpu_mem: float = 0.0
        self.modules_overlap_ratio: float = 0.0
        self.module_deps: Dict[str, int] = {}
        self.ready_to_run: bool = True

        if calculate_module_deps:
            from .module_deps import module_deps
            self.module_deps:Dict[str, int] = module_deps(sys.modules[func.__module__])

        self._device: Optional[str] = None
        self._device_prefix: Optional[str] = None
        self._device_id: Optional[int] = None
        self.gpu_index: int = -1
        self.dml_id: int = -1
        self.worker: Optional[Worker] = None
        self.mem_before_enter: int = 0
        self._future: Optional[Future] = None
        self.dep_futures: Dict[Future, Union[str, int]] = {}
        self.finished_dep_futures_count_lock: threading.Lock = threading.Lock()
        self.finished_dep_futures_count: int = 0

        for i, arg in enumerate(args):
            if isinstance(arg, _BaseFuture):
                self.dep_futures[arg] = i

        for key, arg in kwargs.items():
            if isinstance(arg, _BaseFuture):
                self.dep_futures[arg] = key

        if self.dep_futures:
            self.ready_to_run = False
            callback = partial(self._dep_future_done_callback, pool)
            for dep_future in self.dep_futures:
                dep_future.add_done_callback(callback)

    def can_use(self)->bool:
        return True

    def _dep_future_done_callback(self, pool: Pool, future: Future) -> None:
        try:
            with self.finished_dep_futures_count_lock:
                self.finished_dep_futures_count += 1

                result = future.result()
                future_key = self.dep_futures[future]
                if isinstance(future_key, int):
                    self.args[future_key] = result
                else:
                    self.kwargs[future_key] = result

                if self.finished_dep_futures_count == len(self.dep_futures):
                    self.ready_to_run = True
                    with pool._lock:
                        pool._submit(self, append_to_delay=False)
        except Exception as e:
            self.future.set_exception(e)
            with pool._lock:
                if self.id in pool._tasks:
                    pool._tasks.pop(self.id)

    @property
    def future(self)->Future:
        if self._future is None:
            self._future = Future(self.check_args)

        return self._future

    @property
    def device(self)->str:
        return self._device

    @device.setter
    def device(self, device:str)->None:
        if self._device == device:
            return

        self._device = device
        self._device_prefix = None
        self._device_id = None

    @property
    def effective_res(self) -> Resource:
        if self.device and self.device != "cpu":
            return self.gpu_mode_res

        return self.cpu_mode_res

    def info(self):
        return (self.id, self.device, self.func, self.args, self.kwargs)

    @property
    def device_id(self)->int:
        if self._device_id is not None:
            return self._device_id

        self._device_prefix = ""
        self._device_id = 0
        device = self._device
        if isinstance(device, str):
            if ":" in device:
                parts = device.split(":")
                if len(parts) == 2:
                    try:
                        self._device_prefix = parts[0]
                        self._device_id = int(parts[1])
                    except (ValueError, IndexError):
                        pass
            else:
                self._device_prefix = device

        return self._device_id

    @property
    def device_prefix(self)->str:
        if self._device_prefix is not None:
            return self._device_prefix

        self._device_prefix = ""
        self._device_id = -1
        device = self._device
        if isinstance(device, str):
            if ":" in device:
                parts = device.split(":")
                if len(parts) == 2:
                    try:
                        self._device_prefix = parts[0]
                        self._device_id = int(parts[1])
                    except (ValueError, IndexError):
                        pass
            else:
                self._device_prefix = device

        return self._device_prefix

    def exec(self)->Tuple[bool, Any]:
        try:
            result = self.func(*self.args, **self.kwargs)
            success = True
        except Exception as e:
            result = e
            success = False

        return success, result

    def filter_gpu_infos(self, gpu_infos: List[GPUInfoSnapshot]) -> Iterable[GPUInfoSnapshot]:
        if not self.use_torch or not gpu_infos:
            return gpu_infos

        if Task._torch_gpu_backend is None:
            import torch
            if torch.cuda.is_available():
                Task._torch_gpu_backend = "cuda"

            if getattr(torch, "hip", None) and torch.hip.is_available():
                Task._torch_gpu_backend = "hip"
            elif getattr(torch, "xpu", None) and torch.xpu.is_available():
                Task._torch_gpu_backend = "xpu"

        return filter(lambda gpu: gpu.device.startswith(Task._torch_gpu_backend), gpu_infos)
