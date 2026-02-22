import psutil
from .gpuinfos import GPUInfos
from threading import Lock


class SysInfo:

    def __init__(self):
        self._cpu_cores_free = int((1 - psutil.cpu_percent() / 100) * psutil.cpu_count())
        self._cpu_mem_free = psutil.virtual_memory().available
        self._gpu_infos = GPUInfos.snapshot("n_cores_free", "memory_free")

        self._lock = Lock()

    @property
    def cpu_mem_free(self):
        with self._lock:
            return self._cpu_mem_free
    
    @cpu_mem_free.setter
    def cpu_mem_free(self, cpu_mem_free):
        with self._lock:
            self._cpu_mem_free = cpu_mem_free
        
    @property
    def gpu_infos(self):
        with self._lock:
            return self._gpu_infos
    
    @gpu_infos.setter
    def gpu_infos(self, gpu_infos):
        with self._lock:
            self._gpu_infos = gpu_infos

    @property
    def cpu_cores_free(self):
        with self._lock:
            return self._cpu_cores_free
    
    @cpu_cores_free.setter
    def cpu_cores_free(self, cpu_cores_free):
        with self._lock:
            self._cpu_cores_free = cpu_cores_free

    def update(self):
        with self._lock:
            self._cpu_core_free = (1 - psutil.cpu_percent() / 100) * psutil.cpu_count()
            self._cpu_mem_free = psutil.virtual_memory().available
            self._gpu_infos = GPUInfos.snapshot("n_cores_free", "memory_free")