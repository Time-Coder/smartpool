import contextlib
import platform
import subprocess
import threading
import uuid
import warnings
from typing import Dict, List, Optional, Tuple

from .gpuinfo import GPUInfo, GPUVendor


class AMDGPUInfo(GPUInfo):

    _lock = threading.Lock()
    _initialized = False
    _adlx_helper = None
    _system = None
    _gpu_list = None
    _perf_monitoring = None
    _vram_ranges: Dict[int, int] = {}
    _device_info: List[Dict] = []

    _supported_onnx_providers: List[str] = [
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider"
    ]

    def __init__(self, device_id: int, index: int):
        GPUInfo.__init__(self, device_id, index)
        self._ensure_init()

    @classmethod
    def _ensure_init(cls) -> None:
        if cls._initialized:
            return

        with cls._lock:
            try:
                system = platform.system()
                if system == "Windows":
                    cls._init_windows()
                elif system == "Linux":
                    cls._init_linux()
                elif system == "Darwin":
                    cls._init_mac()
            finally:
                cls._initialized = True

    @classmethod
    def _init_windows(cls) -> None:
        import ADLXPybind

        cls._adlx_helper = ADLXPybind.ADLXHelper()
        res = cls._adlx_helper.Initialize()
        if res != ADLXPybind.ADLX_RESULT.ADLX_OK:
            raise RuntimeError(f"ADLXPybind initialization failed: {res}")

        cls._system = cls._adlx_helper.GetSystemServices()
        gpu_holder = ADLXPybind.ADLXGPUHolder(cls._system)
        cls._gpu_list = gpu_holder.getGPUList()
        cls._perf_monitoring = cls._system.GetPerformanceMonitoringServices()

        for i in range(cls._gpu_list.Size()):
            gpu = cls._gpu_list.At(i)
            try:
                support = cls._perf_monitoring.GetSupportedGPUMetrics(gpu)
                if support:
                    _, max_vram = support.GetGPUVRAMRange()
                    cls._vram_ranges[i] = max_vram
            except Exception:
                cls._vram_ranges[i] = 0

    @classmethod
    def _init_linux(cls) -> None:
        cls._device_info = []

        import pyamdgpuinfo
        count = pyamdgpuinfo.detect_gpus()
        if count > 0:
            for i in range(count):
                gpu = pyamdgpuinfo.get_gpu(i)
                cls._device_info.append({
                    'name': getattr(gpu, 'name', None) or gpu.__class__.__name__,
                    'vram_size': getattr(gpu, 'vram_size', None),
                })
            return
        else:
            warnings.warn("ROCm is not installed, AMD GPU is not available now")

    @classmethod
    def _init_mac(cls) -> None:
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True, timeout=10
            )
            if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                cls._device_info = [{'name': 'AMD Radeon GPU'}]
            else:
                cls._device_info = []
        except Exception:
            cls._device_info = []

    @classmethod
    def shutdown(cls) -> None:
        if not cls._initialized:
            return

        with cls._lock:
            cls._initialized = False
            cls._vram_ranges.clear()
            cls._device_info.clear()

            if cls._adlx_helper:
                with contextlib.suppress(Exception):
                    cls._adlx_helper.Terminate()

                cls._adlx_helper = None
                cls._system = None
                cls._gpu_list = None
                cls._perf_monitoring = None

    @property
    def vendor(self) -> GPUVendor:
        return GPUVendor.AMD

    @classmethod
    def get_device_count(cls) -> int:
        cls._ensure_init()
        system = platform.system()
        if system == "Windows":
            if cls._gpu_list is None:
                return 0
            try:
                return cls._gpu_list.Size()
            except Exception:
                return 0
        return len(cls._device_info)

    def _get_device_info(self) -> Dict:
        if 0 <= self._device_id < len(self._device_info):
            return self._device_info[self._device_id]
        return {}

    def _fetch_name(self) -> str:
        system = platform.system()
        if system == "Windows":
            with self._lock:
                gpu = self._gpu_list.At(self._device_id)
                return str(gpu.Name())
        elif system == "Linux":
            with self._lock:
                import pyamdgpuinfo
                gpu = pyamdgpuinfo.get_gpu(self._device_id)
                return gpu.name
                
        info = self._get_device_info()
        return info.get('name', 'Unknown AMD GPU')

    def _fetch_uuid(self) -> Optional[uuid.UUID]:
        system = platform.system()
        if system == "Windows":
            with self._lock:
                gpu = self._gpu_list.At(self._device_id)
                uid = gpu.UniqueId()
                return uuid.UUID(int=uid)
        return None

    def _fetch_serial(self) -> Optional[str]:
        return None

    def _fetch_driver_version(self) -> Optional[str]:
        system = platform.system()
        if system == "Windows":
            with self._lock:
                gpu = self._gpu_list.At(self._device_id)
                return f"AMD {gpu.DriverPath()}"
        return None

    def _fetch_memory_info(self) -> Tuple[Optional[int], Optional[int]]:
        system = platform.system()
        if system == "Windows":
            with self._lock:
                gpu = self._gpu_list.At(self._device_id)
                total_mb = self._vram_ranges.get(self._device_id, 0)
                if total_mb == 0:
                    total_mb = gpu.TotalVRAM()
                metrics = self._perf_monitoring.GetCurrentGPUMetrics(gpu)
                used_mb = metrics.GPUVRAM()
                total = int(total_mb * 1024 * 1024)
                used = int(used_mb * 1024 * 1024)
                return (total, used)
        elif system == "Linux":
            with self._lock:
                import pyamdgpuinfo
                gpu = pyamdgpuinfo.get_gpu(self._device_id)
                used = int(gpu.query_vram_usage())
                total = int(getattr(gpu, 'vram_size', 0))
                return (total, used)

        info = self._get_device_info()
        vram_size = info.get('vram_size')
        if vram_size:
            return (int(vram_size), 0)
        
        mem_total = info.get('mem_total')
        if mem_total:
            return (mem_total, 0)
        
        return (None, None)

    def _fetch_utilization(self) -> Optional[float]:
        system = platform.system()

        if system == "Windows":
            with self._lock:
                gpu = self._gpu_list.At(self._device_id)
                metrics = self._perf_monitoring.GetCurrentGPUMetrics(gpu)
                usage = metrics.GPUUsage()
                return usage / 100.0
        elif system == "Linux":
            with self._lock:
                import pyamdgpuinfo
                gpu = pyamdgpuinfo.get_gpu(self._device_id)
                return float(gpu.query_load())
        elif system == "Darwin":
            return self._fetch_utilization_mac()
        else:
            raise RuntimeError(f"unsupported system {system}")

    def _fetch_utilization_mac(self) -> float:
        result = subprocess.run(
            ['powermetrics', '--samplers', 'gpu_power', '-i', '100', '-n', '1'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if 'GPU' not in line or '%' not in line:
                    continue

                import re
                match = re.search(r'(\d+(\.\d+)?)%', line)
                if not match:
                    continue

                return float(match.group(1)) / 100.0

        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True, text=True, timeout=10, check=True
        )
        if 'AMD' in result.stdout:
            pass

    def _fetch_temperature(self) -> Optional[int]:
        system = platform.system()
        if system == "Windows":
            with self._lock:
                gpu = self._gpu_list.At(self._device_id)
                metrics = self._perf_monitoring.GetCurrentGPUMetrics(gpu)
                temp = int(metrics.GPUTemperature())
                return temp
        elif system == "Linux":
            with self._lock:
                import pyamdgpuinfo
                gpu = pyamdgpuinfo.get_gpu(self._device_id)
                return float(gpu.query_temperature())
        else:
            raise RuntimeError(f"not supported system: {system}")

    def _fetch_display_mode(self) -> Optional[bool]:
        return None

    def _fetch_display_active(self) -> Optional[bool]:
        return None

    def _fetch_num_cores(self) -> int:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for plat in platforms:
            vendor = plat.vendor.lower()
            if 'amd' not in vendor and 'advanced micro devices' not in vendor:
                continue

            devices = plat.get_devices(device_type=cl.device_type.GPU)
            if 0 <= self._device_id < len(devices):
                device = devices[self._device_id]
                return device.max_compute_units * device.max_work_group_size

        raise RuntimeError("cannot fetch num cores of current AMD GPU")
