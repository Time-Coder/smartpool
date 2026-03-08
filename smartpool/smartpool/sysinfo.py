from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .gpuinfos import GPUInfoSnapshot


class SysInfo:

    def __init__(self):
        import psutil
        import threading

        self._cpu_cores_total = psutil.cpu_count()
        self._cpu_mem_total = psutil.virtual_memory().total
        self._last_cpu_percent = 0.0

        self._cpu_cores_used = None
        self._cpu_mem_used = None
        self._gpu_infos:List[GPUInfoSnapshot] = None
        self._get_cpu_percent_thread = threading.Thread(target=self._get_cpu_percent, daemon=True)
        self._get_cpu_percent_thread.start()

    @property
    def cpu_mem_free(self)->int:
        return self._cpu_mem_total - self.cpu_mem_used

    @cpu_mem_free.setter
    def cpu_mem_free(self, cpu_mem_free):
        self._cpu_mem_used = self._cpu_mem_total - cpu_mem_free

    @property
    def cpu_mem_used(self)->int:
        if self._cpu_mem_used is None:
            import psutil
            self._cpu_mem_used = min(self._cpu_mem_total, psutil.virtual_memory().used + 0.15 * self._cpu_mem_total)

        return self._cpu_mem_used

    @cpu_mem_used.setter
    def cpu_mem_used(self, cpu_mem_used):
        self._cpu_mem_used = cpu_mem_used

    @property
    def gpu_infos(self)->List[GPUInfoSnapshot]:
        from .gpuinfos import GPUInfos

        if self._gpu_infos is None:
            self._gpu_infos = GPUInfos.snapshot("n_cores", "n_cores_used", "mem_total", "mem_used")

        return self._gpu_infos

    @property
    def cpu_cores_free(self)->float:
        return self._cpu_cores_total - self.cpu_cores_used

    @cpu_cores_free.setter
    def cpu_cores_free(self, cpu_cores_free):
        self._cpu_cores_used = self._cpu_cores_total - cpu_cores_free

    @property
    def cpu_cores_used(self)->float:
        if self._cpu_cores_used is None:
            if self._last_cpu_percent > 0:
                used_cpu_percent = self._last_cpu_percent
            else:
                import psutil

                used_cpu_percent = psutil.cpu_percent(interval=0.1)

            self._cpu_cores_used = used_cpu_percent / 100 * self._cpu_cores_total

        return self._cpu_cores_used

    @cpu_cores_used.setter
    def cpu_cores_used(self, cpu_cores_used):
        self._cpu_cores_used = cpu_cores_used

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

    def _get_cpu_percent(self)->None:
        import psutil
        while True:
            self._last_cpu_percent = psutil.cpu_percent(interval=1)