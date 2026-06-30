import threading
import uuid
from typing import Optional

import pynvml

from .gpuinfo import GPUInfo, GPUVendor


class NvidiaGPUInfo(GPUInfo):

    _lock = threading.Lock()
    _initialized = False

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
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.device_id)

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
            except Exception:
                pass

            return self._fetch_num_cores_from_compute_capability()

    @staticmethod
    def _fetch_num_cores_from_compute_capability() -> Optional[int]:
        try:
            cc = pynvml.nvmlDeviceGetCudaComputeCapability(
                pynvml.nvmlDeviceGetHandleByIndex(0)
            )
            cores_per_sm = NvidiaGPUInfo._CC_TO_CORES_PER_SM.get(cc)
            if cores_per_sm is None:
                return None

            name = pynvml.nvmlDeviceGetName(
                pynvml.nvmlDeviceGetHandleByIndex(0)
            )
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            sm_count = NvidiaGPUInfo._GPU_NAME_TO_SM_COUNT.get(name)
            if sm_count is None:
                return None

            return cores_per_sm * sm_count
        except Exception:
            return None

    _CC_TO_CORES_PER_SM = {
        (3, 0): 192, (3, 2): 192, (3, 5): 192, (3, 7): 192,
        (5, 0): 128, (5, 2): 128, (5, 3): 128,
        (6, 0): 64,
        (6, 1): 128, (6, 2): 128,
        (7, 0): 64, (7, 2): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128, (8, 7): 128, (8, 9): 128,
        (9, 0): 128,
    }

    _GPU_NAME_TO_SM_COUNT = {
        'NVIDIA GeForce RTX 4090': 128,
        'NVIDIA GeForce RTX 4080': 76,
        'NVIDIA GeForce RTX 4080 SUPER': 80,
        'NVIDIA GeForce RTX 4070 Ti': 60,
        'NVIDIA GeForce RTX 4070 Ti SUPER': 66,
        'NVIDIA GeForce RTX 4070': 36,
        'NVIDIA GeForce RTX 4070 SUPER': 46,
        'NVIDIA GeForce RTX 4060 Ti': 34,
        'NVIDIA GeForce RTX 4060': 24,
        'NVIDIA GeForce RTX 3090 Ti': 84,
        'NVIDIA GeForce RTX 3090': 82,
        'NVIDIA GeForce RTX 3080 Ti': 80,
        'NVIDIA GeForce RTX 3080': 68,
        'NVIDIA GeForce RTX 3070 Ti': 48,
        'NVIDIA GeForce RTX 3070': 46,
        'NVIDIA GeForce RTX 3060 Ti': 38,
        'NVIDIA GeForce RTX 3060': 28,
        'NVIDIA GeForce RTX 2080 Ti': 68,
        'NVIDIA GeForce RTX 2080 SUPER': 48,
        'NVIDIA GeForce RTX 2080': 46,
        'NVIDIA GeForce RTX 2070 SUPER': 40,
        'NVIDIA GeForce RTX 2070': 36,
        'NVIDIA GeForce RTX 2060 SUPER': 34,
        'NVIDIA GeForce RTX 2060': 30,
        'NVIDIA GeForce RTX 4090 Laptop GPU': 76,
        'NVIDIA GeForce RTX 4080 Laptop GPU': 58,
        'NVIDIA GeForce RTX 4070 Laptop GPU': 36,
        'NVIDIA GeForce RTX 4060 Laptop GPU': 24,
        'NVIDIA GeForce RTX 3080 Laptop GPU': 48,
        'NVIDIA GeForce RTX 3070 Laptop GPU': 32,
        'NVIDIA GeForce RTX 3060 Laptop GPU': 30,
        'NVIDIA GeForce RTX 2080 with Max-Q Design': 46,
        'NVIDIA GeForce RTX 2070 with Max-Q Design': 36,
        'NVIDIA GeForce RTX 2060 with Max-Q Design': 30,
        'NVIDIA GeForce GTX 1660 Ti': 24,
        'NVIDIA GeForce GTX 1660 SUPER': 22,
        'NVIDIA GeForce GTX 1660': 22,
        'NVIDIA GeForce GTX 1650 SUPER': 20,
        'NVIDIA GeForce GTX 1650': 14,
        'NVIDIA A100 80GB PCIe': 108,
        'NVIDIA A100 40GB PCIe': 108,
        'NVIDIA A100-SXM4-80GB': 108,
        'NVIDIA A100-SXM4-40GB': 108,
        'NVIDIA H100 80GB HBM3': 132,
        'NVIDIA H100 PCIe': 114,
        'NVIDIA L40S': 142,
        'NVIDIA L40': 142,
        'NVIDIA L4': 58,
        'NVIDIA A10': 72,
        'NVIDIA A30': 28,
        'NVIDIA A16': 40,
        'NVIDIA Tesla V100-SXM2-32GB': 80,
        'NVIDIA Tesla V100-SXM2-16GB': 80,
        'NVIDIA Tesla V100-PCIE-32GB': 80,
        'NVIDIA Tesla V100-PCIE-16GB': 80,
        'NVIDIA Tesla T4': 40,
    }
