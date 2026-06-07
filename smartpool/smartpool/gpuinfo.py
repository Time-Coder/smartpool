from abc import ABC, abstractmethod
import uuid
from enum import Enum
from typing import Type, Optional


class GPUVendor(Enum):
    NVIDIA = "NVIDIA"
    INTEL = "INTEL"
    AMD = "AMD"
    UNKNOWN = "UNKNOWN"


class GPUInfoSnapshot:

    def __init__(self):
        self.index: Optional[int] = None
        self.device_id: Optional[int] = None
        self.dml_id: Optional[int] = None

        self.name: Optional[str] = None
        self.uuid: Optional[uuid.UUID] = None
        self.serial: Optional[str] = None
        self.driver_version: Optional[str] = None
        self.vendor: Optional[str] = None
        self.device: Optional[str] = None
        self.mem_total: Optional[int] = None
        self.mem_used: Optional[int] = None
        self.load: Optional[float] = None
        self.temperature: Optional[int] = None
        self.display_active: Optional[bool] = None
        self.display_mode: Optional[bool] = None
        self.n_cores: Optional[int] = None
        self.n_cores_used: Optional[int] = None

    @property
    def mem_free(self) -> Optional[int]:
        return self.mem_total - self.mem_used

    @mem_free.setter
    def mem_free(self, value: int) -> None:
        self.mem_used = self.mem_total - value

    @property
    def n_cores_free(self) -> Optional[int]:
        return self.n_cores - self.n_cores_used

    @n_cores_free.setter
    def n_cores_free(self, value: int) -> None:
        self.n_cores_used = self.n_cores - value


class GPUInfo(ABC):

    supported_onnx_providers = []

    def __init__(self, device_id: int, index: int):
        self._device_id: int = device_id
        self._index: int = index
        self._dml_id: int = -1

        self._name: Optional[str] = None
        self._uuid: Optional[uuid.UUID] = None
        self._serial: Optional[str] = None
        self._driver_version: Optional[str] = None
        self._mem_total: Optional[int] = None
        self._mem_used: Optional[int] = None
        self._load: Optional[float] = None
        self._temperature: Optional[int] = None
        self._display_active: Optional[bool] = None
        self._display_mode: Optional[bool] = None
        self._n_cores: Optional[int] = None
        self._n_cores_used: Optional[int] = None

    @abstractmethod
    def _fetch_name(self) -> str:
        pass

    @abstractmethod
    def _fetch_uuid(self) -> Optional[uuid.UUID]:
        pass

    @abstractmethod
    def _fetch_serial(self) -> Optional[str]:
        pass

    @abstractmethod
    def _fetch_driver_version(self) -> Optional[str]:
        pass

    @abstractmethod
    def _fetch_memory_info(self) -> tuple[Optional[int], Optional[int]]:
        pass

    @abstractmethod
    def _fetch_utilization(self) -> Optional[float]:
        pass

    @abstractmethod
    def _fetch_temperature(self) -> Optional[int]:
        pass

    @abstractmethod
    def _fetch_display_mode(self) -> Optional[bool]:
        pass

    @abstractmethod
    def _fetch_display_active(self) -> Optional[bool]:
        pass

    @abstractmethod
    def _fetch_num_cores(self) -> Optional[int]:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @classmethod
    def get_device_count(cls) -> int:
        return 0

    @property
    @abstractmethod
    def vendor(self) -> GPUVendor:
        pass

    @staticmethod
    def gpuinfo_class(device_prefix:str)->Type[GPUInfo]:
        if device_prefix == "cuda":
            from .nvidia_gpuinfo import NvidiaGPUInfo
            return NvidiaGPUInfo
        elif device_prefix == "xpu":
            from .intel_gpuinfo import IntelGPUInfo
            return IntelGPUInfo
        elif device_prefix == "hip":
            from .amd_gpuinfo import AMDGPUInfo
            return AMDGPUInfo
        else:
            raise ValueError(f"Unknown device prefix: {device_prefix}")

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def index(self) -> int:
        return self._index

    @property
    def dml_id(self) -> int:
        return self._dml_id

    @property
    def device(self) -> str:
        vendor_prefix = {
            GPUVendor.NVIDIA: "cuda",
            GPUVendor.INTEL: "xpu",
            GPUVendor.AMD: "hip",
        }
        prefix = vendor_prefix[self.vendor]
        return f"{prefix}:{self.device_id}"

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = self._fetch_name()

        return self._name

    @property
    def uuid(self) -> Optional[uuid.UUID]:
        if self._uuid is None:
            self._uuid = self._fetch_uuid()

        return self._uuid

    @property
    def serial(self) -> Optional[str]:
        if self._serial is None:
            self._serial = self._fetch_serial()

        return self._serial

    @property
    def driver_version(self) -> Optional[str]:
        if self._driver_version is None:
            self._driver_version = self._fetch_driver_version()

        return self._driver_version

    @property
    def mem_total(self) -> Optional[int]:
        if self._mem_total is None:
            self._mem_total, _ = self._fetch_memory_info()

        return self._mem_total

    @property
    def mem_used(self) -> Optional[int]:
        if self._mem_used is None:
            _, self._mem_used = self._fetch_memory_info()

        return self._mem_used

    @property
    def mem_free(self) -> Optional[int]:
        if self.mem_total is None or self.mem_used is None:
            return None
        return self.mem_total - self.mem_used

    @property
    def mem_util(self) -> Optional[float]:
        if self.mem_total is None or self.mem_used is None:
            return None
        
        if self.mem_total == 0:
            return None
        
        return self.mem_used / self.mem_total

    @property
    def load(self) -> Optional[float]:
        if self._load is None:
            self._load = self._fetch_utilization()

        return self._load

    @property
    def n_cores(self) -> Optional[int]:
        if self._n_cores is None:
            self._n_cores = self._fetch_num_cores()

        return self._n_cores

    @property
    def n_cores_used(self) -> Optional[int]:
        if self._n_cores_used is None and self.load is not None and self.n_cores is not None:
            self._n_cores_used = int(self.load * self.n_cores)

        return self._n_cores_used

    @property
    def n_cores_free(self) -> Optional[int]:
        if self.n_cores is None or self.n_cores_used is None:
            return None
        
        return self.n_cores - self.n_cores_used

    @property
    def temperature(self) -> Optional[int]:
        if self._temperature is None:
            self._temperature = self._fetch_temperature()

        return self._temperature

    @property
    def display_mode(self) -> Optional[bool]:
        if self._display_mode is None:
            self._display_mode = self._fetch_display_mode()

        return self._display_mode

    @property
    def display_active(self) -> Optional[bool]:
        if self._display_active is None:
            self._display_active = self._fetch_display_active()
            
        return self._display_active

    def update(self) -> None:
        self._mem_used = None
        self._load = None
        self._temperature = None
        self._display_active = None
        self._display_mode = None
        self._n_cores_used = None

    def snapshot(
        self,
        name: bool = False,
        uuid: bool = False,
        serial: bool = False,
        driver_version: bool = False,
        mem_total: bool = False,
        mem_used: bool = False,
        load: bool = False,
        temperature: bool = False,
        display_mode: bool = False,
        display_active: bool = False,
        n_cores: bool = False,
        n_cores_used: bool = False,
    ) -> GPUInfoSnapshot:
        if not any([
            name, uuid, serial, driver_version, mem_total, mem_used,
            load, temperature, display_mode, display_active,
            n_cores, n_cores_used
        ]):
            name = True
            uuid = True
            serial = True
            driver_version = True
            mem_total = True
            mem_used = True
            load = True
            temperature = True
            display_mode = True
            display_active = True
            n_cores = True
            n_cores_used = True

        snapshot = GPUInfoSnapshot()
        snapshot.index = self._index
        snapshot.device_id = self.device_id
        snapshot.vendor = self.vendor.value
        snapshot.device = self.device
        snapshot.dml_id = self.dml_id

        if name:
            snapshot.name = self.name

        if uuid:
            snapshot.uuid = self.uuid

        if serial:
            snapshot.serial = self.serial

        if driver_version:
            snapshot.driver_version = self.driver_version

        if mem_total:
            snapshot.mem_total = self.mem_total

        if mem_used:
            snapshot.mem_used = self.mem_used

        if load:
            snapshot.load = self.load

        if temperature:
            snapshot.temperature = self.temperature

        if display_mode:
            snapshot.display_mode = self.display_mode

        if display_active:
            snapshot.display_active = self.display_active

        if n_cores:
            snapshot.n_cores = self.n_cores

        if n_cores_used:
            snapshot.n_cores_used = self.n_cores_used

        return snapshot

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.device_id}, name='{self.name}')"
