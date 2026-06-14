import threading
import uuid
from typing import List, Optional

import pynvml

from .gpuinfo import GPUInfo, GPUVendor


class NvidiaGPUInfo(GPUInfo):

    _lock = threading.Lock()
    _initialized = False

    _supported_onnx_providers: List[str] = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider"
    ]

    def __init__(self, device_id: int, index: int):
        GPUInfo.__init__(self, device_id, index)
        self._handle = None
        self._ensure_init()

    @classmethod
    def _ensure_init(cls) -> None:
        if cls._initialized:
            return

        with cls._lock:
            try:
                pynvml.nvmlInit()
            finally:
                cls._initialized = True

    @classmethod
    def shutdown(cls) -> None:
        if not cls._initialized:
            return

        with cls._lock:
            try:
                pynvml.nvmlShutdown()
            finally:
                cls._initialized = False

    def _get_handle(self):
        if self._handle is None:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_id)

        return self._handle

    @property
    def vendor(self) -> GPUVendor:
        return GPUVendor.NVIDIA

    @classmethod
    def get_device_count(cls) -> int:
        cls._ensure_init()

        try:
            return pynvml.nvmlDeviceGetCount()
        except Exception:
            return 0

    def _fetch_name(self) -> str:
        with self._lock:
            name_bytes = pynvml.nvmlDeviceGetName(self._get_handle())
            return name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else name_bytes

    def _fetch_uuid(self) -> Optional[uuid.UUID]:
        with self._lock:
            uuid_bytes = pynvml.nvmlDeviceGetUUID(self._get_handle())

        if isinstance(uuid_bytes, bytes):
            try:
                uuid_str = uuid_bytes.decode('utf-8')
                if uuid_str.startswith('GPU-'):
                    uuid_str = uuid_str[4:]
                return uuid.UUID(uuid_str)
            except (ValueError, UnicodeDecodeError):
                try:
                    return uuid.UUID(bytes=uuid_bytes)
                except ValueError:
                    return None
        else:
            uuid_str = uuid_bytes
            if uuid_str.startswith('GPU-'):
                uuid_str = uuid_str[4:]
            try:
                return uuid.UUID(uuid_str)
            except ValueError:
                return None

    def _fetch_serial(self) -> Optional[str]:
        with self._lock:
            try:
                serial_bytes = pynvml.nvmlDeviceGetSerial(self._get_handle())
                return serial_bytes.decode('utf-8') if isinstance(serial_bytes, bytes) else serial_bytes
            except pynvml.NVMLError:
                return None

    def _fetch_driver_version(self) -> Optional[str]:
        with self._lock:
            driver_bytes = pynvml.nvmlSystemGetDriverVersion()
            return driver_bytes.decode('utf-8') if isinstance(driver_bytes, bytes) else driver_bytes

    def _fetch_memory_info(self) -> tuple[Optional[int], Optional[int]]:
        with self._lock:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._get_handle())

        return (mem_info.total, mem_info.used)

    def _fetch_utilization(self) -> Optional[float]:
        with self._lock:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._get_handle())
            except Exception:
                pynvml.nvmlShutdown()
                pynvml.nvmlInit()
                self._handle = None
                util = pynvml.nvmlDeviceGetUtilizationRates(self._get_handle())

            return util.gpu / 100.0

    def _fetch_temperature(self) -> Optional[int]:
        with self._lock:
            return pynvml.nvmlDeviceGetTemperature(self._get_handle(), pynvml.NVML_TEMPERATURE_GPU)

    def _fetch_display_mode(self) -> Optional[bool]:
        with self._lock:
            return bool(pynvml.nvmlDeviceGetDisplayMode(self._get_handle()))

    def _fetch_display_active(self) -> Optional[bool]:
        with self._lock:
            return bool(pynvml.nvmlDeviceGetDisplayActive(self._get_handle()))

    def _fetch_num_cores(self) -> Optional[int]:
        with self._lock:
            try:
                return pynvml.nvmlDeviceGetNumGpuCores(self._get_handle())
            except Exception as e:
                import pyopencl as cl
                platforms = cl.get_platforms()
                for plat in platforms:
                    vendor = plat.vendor.lower()
                    if 'nvidia' not in vendor:
                        continue

                    devices = plat.get_devices(device_type=cl.device_type.GPU)
                    if 0 <= self._device_id < len(devices):
                        device = devices[self._device_id]
                        return device.max_compute_units * device.max_work_group_size

                raise RuntimeError("cannot fetch num cores of current NVIDIA GPU") from e
