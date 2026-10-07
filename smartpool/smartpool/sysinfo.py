from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import psutil

if TYPE_CHECKING:
    from .gpuinfo import GPUInfoSnapshot


class SysInfo:

    def __init__(self):
        self._cpu_cores_total:int = psutil.cpu_count()
        self._cpu_mem_total:int = psutil.virtual_memory().total
        self._last_cpu_percent:float = 0.0
        self._last_cpu_percent_at:Optional[float] = None

        self._cpu_cores_used:Optional[float] = None
        self._cpu_mem_used:Optional[int] = None
        self._gpu_infos:Optional[List[GPUInfoSnapshot]] = None
        # Logical GPU reservations currently outstanding, keyed by stable device
        # name. Held here because rebuilding the physical snapshot list must not
        # drop reservations made by admitted tasks or retained results.
        self._gpu_reserved:Dict[str, Tuple[int, int]] = {}

    @property
    def cpu_mem_free(self)->int:
        return self._cpu_mem_total - self.cpu_mem_used

    @cpu_mem_free.setter
    def cpu_mem_free(self, cpu_mem_free:float):
        self._cpu_mem_used:float = self._cpu_mem_total - cpu_mem_free

    @property
    def cpu_mem_used(self)->int:
        if self._cpu_mem_used is None:
            self._cpu_mem_used:float = min(self._cpu_mem_total, psutil.virtual_memory().used)

        return self._cpu_mem_used

    @cpu_mem_used.setter
    def cpu_mem_used(self, cpu_mem_used:float):
        self._cpu_mem_used:float = cpu_mem_used

    @property
    def gpu_infos(self)->List[GPUInfoSnapshot]:
        from .gpuinfos import GPUInfos

        if self._gpu_infos is None:
            self._gpu_infos:List[GPUInfoSnapshot] = GPUInfos.snapshot(
                n_cores=True, n_cores_used=True, mem_total=True, mem_used=True
            )
            self._restore_gpu_reserved()

        return self._gpu_infos

    @staticmethod
    def _gpu_key(gpu:GPUInfoSnapshot)->str:
        device = gpu.device
        name = getattr(device, "torch_device", None)
        if name:
            return str(name)

        return f"gpu_index:{getattr(device, 'gpu_index', -1)}"

    def _restore_gpu_reserved(self)->None:
        if not self._gpu_reserved:
            return

        for gpu in self._gpu_infos:
            reserved = self._gpu_reserved.get(self._gpu_key(gpu))
            if reserved is None:
                continue

            n_cores_reserved, mem_reserved = reserved
            if n_cores_reserved:
                gpu.n_cores_free = gpu.n_cores - gpu.n_cores_used - n_cores_reserved

            if mem_reserved:
                gpu.mem_free = gpu.mem_total - gpu.mem_used - mem_reserved

    def _remember_gpu_reserved(self)->None:
        if self._gpu_infos is None:
            return

        remembered:Dict[str, Tuple[int, int]] = {}
        for gpu in self._gpu_infos:
            if gpu.n_cores_reserved or gpu.mem_reserved:
                remembered[self._gpu_key(gpu)] = (gpu.n_cores_reserved, gpu.mem_reserved)

        self._gpu_reserved = remembered

    @property
    def cpu_cores_free(self)->float:
        return self._cpu_cores_total - self.cpu_cores_used

    @cpu_cores_free.setter
    def cpu_cores_free(self, cpu_cores_free:float):
        self._cpu_cores_used:float = self._cpu_cores_total - cpu_cores_free

    @property
    def cpu_cores_used(self)->float:
        if self._cpu_cores_used is None:
            if self._last_cpu_percent > 0:
                used_cpu_percent = self._last_cpu_percent
            else:
                used_cpu_percent:float = psutil.cpu_percent()

            self._cpu_cores_used:float = used_cpu_percent / 100 * self._cpu_cores_total

        return self._cpu_cores_used

    @cpu_cores_used.setter
    def cpu_cores_used(self, cpu_cores_used:float):
        self._cpu_cores_used:float = cpu_cores_used

    @property
    def cpu_cores_total(self)->int:
        return self._cpu_cores_total

    @property
    def cpu_mem_total(self)->int:
        return self._cpu_mem_total

    def update(self, refresh_cpu:bool=True)->None:
        if refresh_cpu:
            self._cpu_cores_used = None
        self._cpu_mem_used = None
        self._remember_gpu_reserved()
        self._gpu_infos = None

    def update_cpu_percent(self, interval: Optional[float] = 1)->None:
        self._last_cpu_percent = psutil.cpu_percent(interval=interval)
        self._last_cpu_percent_at = time.monotonic()

    @property
    def cpu_percent_age(self)->float:
        if self._last_cpu_percent_at is None:
            return float("inf")

        return time.monotonic() - self._last_cpu_percent_at

    def refresh_cpu_now(self, min_interval:float=0.05)->None:
        """Refresh the cached CPU percentage without blocking the caller.

        The housekeeping thread samples once per second, so a non-blocking
        psutil call measures the window since that last sample and is already
        fresh enough for an admission decision. Refreshing is skipped when the
        cached value is younger than ``min_interval``; besides avoiding redundant
        syscalls this keeps the submission path from asking psutil for a
        zero-length window when it races the housekeeping thread.
        """
        if self.cpu_percent_age < min_interval:
            return

        self.update_cpu_percent(interval=None)
