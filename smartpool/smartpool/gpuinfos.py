from __future__ import annotations
import pynvml
import weakref
import uuid
import threading
from typing import List, Optional, Iterator


class GPUInfoSnapshot:

    def __init__(self):
        self.id:Optional[int] = 0
        self.name:Optional[str] = None
        self.uuid:Optional[uuid.UUID] = None
        self.serial:Optional[str] = None
        self.driver_version:Optional[str] = None
        self.mem_total:Optional[int] = None
        self.mem_used:Optional[int] = None
        self.load:Optional[float] = None
        self.temperature:Optional[int] = None
        self.display_active:Optional[bool] = None
        self.display_mode:Optional[bool] = None
        self.n_cores:Optional[int] = None
        self.n_cores_used:Optional[int] = None

    @property
    def device(self)->str:
        return f"cuda:{self.id}"
    
    @property
    def mem_free(self)->int:
        return self.mem_total - self.mem_used
    
    @mem_free.setter
    def mem_free(self, mem_free:int)->None:
        self.mem_used = self.mem_total - mem_free
    
    @property
    def n_cores_free(self)->int:
        return self.n_cores - self.n_cores_used
    
    @n_cores_free.setter
    def n_cores_free(self, n_cores_free:int)->None:
        self.n_cores_used = self.n_cores - n_cores_free


class GPUInfo:

    _lock = threading.Lock()

    def __init__(self, index:int) -> None:
        self._id:int = index
        with self._lock:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        self._name:Optional[str] = None
        self._uuid:Optional[uuid.UUID] = None
        self._serial:Optional[str] = None
        self._driver_version:Optional[str] = None
        self._mem_info = None
        self._mem_total:Optional[int] = None
        self._mem_used:Optional[int] = None
        self._load:Optional[float] = None
        self._temperature:Optional[int] = None
        self._display_active:Optional[bool] = None
        self._display_mode:Optional[bool] = None
        self._n_cores:Optional[int] = None
        self._n_cores_used:Optional[int] = None
        
    def update(self):
        self._mem_info = None
        self._mem_used:Optional[int] = None
        self._load:Optional[float] = None
        self._temperature:Optional[int] = None
        self._display_active:Optional[bool] = None
        self._display_mode:Optional[bool] = None
        self._n_cores_used:Optional[int] = None

    @property
    def id(self)->int:
        return self._id

    @property
    def device(self)->str:
        return f"cuda:{self.id}"

    @property
    def name(self)->str:
        if self._name is None:
            with self._lock:
                name_bytes = pynvml.nvmlDeviceGetName(self._handle)

            self._name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else name_bytes

        return self._name
    
    @property
    def uuid(self)->uuid.UUID:
        if self._uuid is None:
            with self._lock:
                uuid_bytes = pynvml.nvmlDeviceGetUUID(self._handle)

            self._uuid = uuid.UUID(bytes=uuid_bytes)

        return self._uuid
    
    @property
    def serial(self)->str:
        if self._serial is None:
            with self._lock:
                serial_bytes = pynvml.nvmlDeviceGetSerial(self._handle)

            self._serial = serial_bytes.decode('utf-8') if isinstance(serial_bytes, bytes) else serial_bytes

        return self._serial
    
    @property
    def driver_version(self)->str:
        if self._driver_version is None:
            with self._lock:
                driver_bytes = pynvml.nvmlSystemGetDriverVersion()

            self._driver_version = driver_bytes.decode('utf-8') if isinstance(driver_bytes, bytes) else driver_bytes

        return self._driver_version
    
    @property
    def n_cores(self)->int:
        if self._n_cores is None:
            with self._lock:
                self._n_cores = pynvml.nvmlDeviceGetNumGpuCores(self._handle)

        return self._n_cores
    
    @property
    def n_cores_free(self)->int:
        self._update_load()
        return self._n_cores - self._n_cores_used
    
    @property
    def n_cores_used(self)->int:
        self._update_load()
        return self._n_cores_used
    
    @property
    def mem_total(self)->int:
        if self._mem_total is None:
            if self._mem_info is None:
                with self._lock:
                    self._mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

            self._mem_total = self._mem_info.total

        return self._mem_total
    
    @property
    def mem_used(self)->int:
        if self._mem_used is None:
            if self._mem_info is None:
                with self._lock:
                    self._mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

            self._mem_used = self._mem_info.used

        return self._mem_used
    
    @property
    def mem_free(self)->int:
        return self.mem_total - self.mem_used
    
    @property
    def mem_util(self)->float:
        return self.mem_used / self.mem_total
    
    def _update_load(self)->None:
        if self._load is None:
            with self._lock:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                except:
                    pynvml.nvmlShutdown()
                    pynvml.nvmlInit()
                    self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)

            self._load = util.gpu / 100.0
            self._n_cores_used = int(self._load * self.n_cores)

    @property
    def load(self)->float:
        self._update_load()
        return self._load
    
    @property
    def temperature(self)->int:
        if self._temperature is None:
            with self._lock:
                self._temperature = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)

        return self._temperature

    @property
    def display_mode(self)->bool:
        if self._display_mode is None:
            with self._lock:
                self._display_mode = bool(pynvml.nvmlDeviceGetDisplayMode(self._handle))

        return self._display_mode
    
    @property
    def display_active(self)->bool:
        if self._display_active is None:
            with self._lock:
                self._display_active = bool(pynvml.nvmlDeviceGetDisplayActive(self._handle))

        return self._display_active
    
    def __repr__(self)->str:
        return f"GPUInfo(id={self.id}, name='{self.name}')"
    
    def snapshot(self, *args)->GPUInfoSnapshot:
        snapshot = GPUInfoSnapshot()
        snapshot.id = self.id

        if not args or "name" in args:
            snapshot.name = self.name

        if not args or "uuid" in args:
            snapshot.uuid = self.uuid

        if not args or "serial" in args:
            snapshot.serial = self.serial

        if not args or "driver_version" in args:
            snapshot.driver_version = self.driver_version

        if not args or "mem_total" in args or "mem_free" in args or "mem_used" in args:
            snapshot.mem_total = self.mem_total

        if not args or "mem_used" in args or "mem_free" in args:
            snapshot.mem_used = self.mem_used

        if not args or "load" in args:
            snapshot.load = self.load

        if not args or "temperature" in args:
            snapshot.temperature = self.temperature

        if not args or "display_active" in args:
            snapshot.display_active = self.display_active

        if not args or "display_mode" in args:
            snapshot.display_mode = self.display_mode

        if not args or "n_cores" in args or "n_cores_free" in args or "n_cores_used" in args:
            snapshot.n_cores = self.n_cores

        if not args or "n_cores_used" in args or "n_cores_free" in args:
            snapshot.n_cores_used = self.n_cores_used

        return snapshot


class NVMLConstructor:

    def __init__(self):
        pynvml.nvmlInit()
        self.finalizer = weakref.finalize(self, pynvml.nvmlShutdown)


class MetaGPUInfos(type):

    __has_gpu:bool = True
    __nvml_constructor:Optional[NVMLConstructor] = None
    __n_devices:Optional[int] = None
    __gpu_infos:Optional[List[GPUInfo]] = None

    @staticmethod
    def __init()->None:
        if not MetaGPUInfos.__has_gpu:
            return
        
        if MetaGPUInfos.__nvml_constructor is None:
            try:
                MetaGPUInfos.__nvml_constructor = NVMLConstructor()
            except:
                MetaGPUInfos.__has_gpu = False

    def __len__(self)->int:
        if not MetaGPUInfos.__has_gpu:
            return 0

        if MetaGPUInfos.__n_devices is None:
            MetaGPUInfos.__init()
            MetaGPUInfos.__n_devices = pynvml.nvmlDeviceGetCount()

        return MetaGPUInfos.__n_devices

    def __getitem__(self, index:int)->GPUInfo:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")

        if MetaGPUInfos.__gpu_infos is None:
            MetaGPUInfos.__gpu_infos = []
            for i in range(len(self)):
                MetaGPUInfos.__gpu_infos.append(GPUInfo(i))

        return MetaGPUInfos.__gpu_infos[index]
    
    def __iter__(self)->Iterator[GPUInfo]:
        if MetaGPUInfos.__gpu_infos is None:
            MetaGPUInfos.__gpu_infos = []
            for i in range(len(self)):
                MetaGPUInfos.__gpu_infos.append(GPUInfo(i))

        return iter(MetaGPUInfos.__gpu_infos)


class GPUInfos(metaclass=MetaGPUInfos):
    
    @staticmethod
    def update()->None:
        for gpu in GPUInfos:
            gpu.update()

    @staticmethod
    def snapshot(*args)->List[GPUInfoSnapshot]:
        result = []
        for gpu in GPUInfos:
            gpu.update()
            result.append(gpu.snapshot(*args))

        return result
