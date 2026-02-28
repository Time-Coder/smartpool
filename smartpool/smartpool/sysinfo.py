import psutil
from threading import Lock


class SysInfo:

    def __init__(self):
        self._cpu_cores_total = psutil.cpu_count()
        mem_info = psutil.virtual_memory()
        self._cpu_mem_total = mem_info.total

        self._cpu_cores_used = None
        self._cpu_mem_used = None
        self._gpu_infos = None

        self._lock = Lock()

    @property
    def cpu_mem_free(self):
        return self._cpu_mem_total - self.cpu_mem_used
    
    @cpu_mem_free.setter
    def cpu_mem_free(self, cpu_mem_free):
        with self._lock:
            self._cpu_mem_used = self._cpu_mem_total - cpu_mem_free

    @property
    def cpu_mem_used(self):
        with self._lock:
            if self._cpu_mem_used is None:
                self._cpu_mem_used = psutil.virtual_memory().used

            return self._cpu_mem_used
        
    @cpu_mem_used.setter
    def cpu_mem_used(self, cpu_mem_used):
        with self._lock:
            self._cpu_mem_used = cpu_mem_used
        
    @property
    def gpu_infos(self):
        from .gpuinfos import GPUInfos

        with self._lock:
            if self._gpu_infos is None:
                self._gpu_infos = GPUInfos.snapshot("n_cores", "n_cores_used", "mem_total", "mem_used")

            return self._gpu_infos

    @property
    def cpu_cores_free(self):
        return self._cpu_cores_total - self.cpu_cores_used
    
    @cpu_cores_free.setter
    def cpu_cores_free(self, cpu_cores_free):
        with self._lock:
            self._cpu_cores_used = self._cpu_cores_total - cpu_cores_free

    @property
    def cpu_cores_used(self):
        with self._lock:
            if self._cpu_cores_used is None:
                self._cpu_cores_used = psutil.cpu_percent() / 100 * self._cpu_cores_total

            return self._cpu_cores_used
        
    @cpu_cores_used.setter
    def cpu_cores_used(self, cpu_cores_used):
        with self._lock:
            self._cpu_cores_used = cpu_cores_used

    @property
    def cpu_cores_total(self):
        return self._cpu_cores_total
        
    @property
    def cpu_mem_total(self):
        return self._cpu_mem_total

    def update(self):
        with self._lock:
            self._cpu_core_used = None
            self._cpu_mem_used = None
            self._gpu_infos = None