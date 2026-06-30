from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import psutil

if TYPE_CHECKING:
    from .gpuinfo import GPUInfoSnapshot


class SysInfo:

    def __init__(self):
        self._cpu_cores_total:int = psutil.cpu_count()
        self._cpu_mem_total:int = psutil.virtual_memory().total
        self._last_cpu_percent:float = 0.0

        self._cpu_cores_used:Optional[float] = None
        self._cpu_mem_used:Optional[float] = None
        self._gpu_infos:Optional[List[GPUInfoSnapshot]] = None

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

        return self._gpu_infos

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

    def update(self)->None:
        self._cpu_cores_used = None
        self._cpu_mem_used = None
        self._gpu_infos = None

    def update_cpu_percent(self)->None:
        self._last_cpu_percent = psutil.cpu_percent(interval=1)
