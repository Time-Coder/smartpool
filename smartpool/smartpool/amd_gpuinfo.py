import contextlib
import json
import platform
import re
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
    _wsl_fallback: bool = False

    _supported_ort_providers: List[str] = [
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider"
    ]

    # GPU model name -> Shader Processor (Stream Processor) count lookup table
    _GPU_NAME_TO_CU_COUNT: Dict[str, int] = {
        # RDNA 3 (Desktop)
        'AMD Radeon RX 7900 XTX': 6144,
        'AMD Radeon RX 7900 XT': 5376,
        'AMD Radeon RX 7900 GRE': 5120,
        'AMD Radeon RX 7800 XT': 3840,
        'AMD Radeon RX 7700 XT': 3456,
        'AMD Radeon RX 7600': 2048,
        # RDNA 3 (Laptop)
        'AMD Radeon RX 7900S': 2048,
        'AMD Radeon RX 7700S': 1792,
        'AMD Radeon RX 7600S': 1792,
        # RDNA 2 (Desktop)
        'AMD Radeon RX 6950 XT': 5120,
        'AMD Radeon RX 6900 XT': 5120,
        'AMD Radeon RX 6800 XT': 4608,
        'AMD Radeon RX 6800': 3840,
        'AMD Radeon RX 6750 XT': 2560,
        'AMD Radeon RX 6700 XT': 2560,
        'AMD Radeon RX 6700': 2304,
        'AMD Radeon RX 6650 XT': 2048,
        'AMD Radeon RX 6600 XT': 2048,
        'AMD Radeon RX 6600': 1792,
        'AMD Radeon RX 6500 XT': 1024,
        'AMD Radeon RX 6400': 768,
        # RDNA 2 (Laptop)
        'AMD Radeon RX 6850M XT': 2560,
        'AMD Radeon RX 6800M': 2560,
        'AMD Radeon RX 6700M': 2304,
        'AMD Radeon RX 6600M': 1792,
        'AMD Radeon RX 6500M': 1024,
        # RDNA 2 (Integrated)
        'AMD Radeon 680M': 768,
        'AMD Radeon 660M': 384,
        'AMD Radeon 610M': 128,
        # RDNA (Desktop)
        'AMD Radeon RX 5700 XT': 2560,
        'AMD Radeon RX 5700': 2304,
        'AMD Radeon RX 5600 XT': 2304,
        'AMD Radeon RX 5600M': 2304,
        # Vega
        'AMD Radeon VII': 3840,
        'AMD Radeon RX Vega 64': 4096,
        'AMD Radeon RX Vega 56': 3584,
        # Polaris
        'AMD Radeon RX 590': 2304,
        'AMD Radeon RX 580': 2304,
        'AMD Radeon RX 570': 2048,
        # Professional
        'AMD Radeon Pro W6800': 5120,
        'AMD Radeon Pro W6600': 2048,
        'AMD Radeon Pro W6400': 1536,
        'AMD Radeon Pro W5500': 1408,
        'AMD Radeon Pro W5700': 2304,
    }

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
        cls._wsl_fallback = False

        import pyamdgpuinfo
        try:
            count = pyamdgpuinfo.detect_gpus()
        except Exception:
            count = 0

        if count > 0:
            for i in range(count):
                gpu = pyamdgpuinfo.get_gpu(i)
                cls._device_info.append({
                    'name': getattr(gpu, 'name', None) or gpu.__class__.__name__,
                    'vram_size': getattr(gpu, 'vram_size', None),
                })
            return

        cls._init_wsl_fallback()

    @classmethod
    def _init_wsl_fallback(cls) -> None:
        try:
            import subprocess
            import json
            result = subprocess.run(
                ['powershell.exe', '-Command',
                 'Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, PNPDeviceID | ConvertTo-Json'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return

            data = json.loads(result.stdout)
            if isinstance(data, dict):
                data = [data]

            for gpu in data:
                name = gpu.get('Name', '')
                if not name or 'AMD' not in name.upper() and 'Radeon' not in name.upper():
                    continue

                vram = gpu.get('AdapterRAM')
                if vram is not None:
                    vram = int(vram)

                cls._device_info.append({
                    'name': name,
                    'vram_size': vram,
                    'pnp_device_id': gpu.get('PNPDeviceID', ''),
                })

            if cls._device_info:
                cls._wsl_fallback = True
        except Exception:
            pass

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
                try:
                    import pyamdgpuinfo
                    gpu = pyamdgpuinfo.get_gpu(self._device_id)
                    return gpu.name
                except Exception:
                    pass

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
                try:
                    import pyamdgpuinfo
                    gpu = pyamdgpuinfo.get_gpu(self._device_id)
                    used = int(gpu.query_vram_usage())
                    total = int(getattr(gpu, 'vram_size', 0))
                    return (total, used)
                except Exception:
                    pass

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
            if self._wsl_fallback:
                return self._fetch_utilization_wsl()
            with self._lock:
                try:
                    import pyamdgpuinfo
                    gpu = pyamdgpuinfo.get_gpu(self._device_id)
                    return float(gpu.query_load())
                except Exception:
                    return None
        elif system == "Darwin":
            return self._fetch_utilization_mac()
        else:
            raise RuntimeError(f"unsupported system {system}")

    def _fetch_utilization_wsl(self) -> Optional[float]:
        try:
            result = subprocess.run(
                ['powershell.exe', '-Command',
                 r"Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Select-Object InstanceName, CookedValue | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            if not data:
                return None

            # Get all unique LUIDs and their engine counts
            # Format: pid_12345_luid_0x00000000_0x0001272b_phys_0_eng_0_engtype_3d
            luid_engines = {}
            for item in data:
                match = re.search(r'luid_0x[0-9a-f]+_(0x[0-9a-f]+)_phys', item.get('InstanceName', ''))
                if match:
                    luid = match.group(1)
                    if luid not in luid_engines:
                        luid_engines[luid] = 0
                    luid_engines[luid] += 1

            # AMD GPUs typically have fewer engines than NVIDIA
            # Sort by engine count and pick the one with moderate count (AMD-like)
            # NVIDIA typically has 300+ engines, AMD typically has 100-200
            amd_luid = None
            for luid, count in luid_engines.items():
                if 50 < count < 300:  # AMD-like engine count
                    amd_luid = luid
                    break

            if not amd_luid:
                # Fallback: use the LUID with the most engines that isn't the first one
                sorted_luids = sorted(luid_engines.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_luids) > 1:
                    amd_luid = sorted_luids[1][0]  # Second highest (likely AMD)
                elif sorted_luids:
                    amd_luid = sorted_luids[0][0]

            if not amd_luid:
                return None

            # Get max utilization across main engine types (3d, videodecode, videoencode)
            # These represent actual GPU workload, not idle copy/security engines
            max_util = 0.0
            for item in data:
                if f'luid_0x00000000_{amd_luid}' in item.get('InstanceName', ''):
                    instance_name = item.get('InstanceName', '')
                    # Only consider main workload engines
                    if 'engtype_3d' in instance_name or 'engtype_videodecode' in instance_name or 'engtype_videoencode' in instance_name:
                        util = item.get('CookedValue', 0.0)
                        if util > max_util:
                            max_util = util

            return max_util / 100.0

        except Exception:
            pass

        return None

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
                try:
                    import pyamdgpuinfo
                    gpu = pyamdgpuinfo.get_gpu(self._device_id)
                    return float(gpu.query_temperature())
                except Exception:
                    return None
        else:
            return None

    def _fetch_display_mode(self) -> Optional[bool]:
        return None

    def _fetch_display_active(self) -> Optional[bool]:
        return None

    def _fetch_num_cores(self) -> Optional[int]:
        # Fall back to lookup table
        if self._device_info:
            info = self._get_device_info()
            gpu_name = info.get('name', '')
            if gpu_name:
                # Try exact match first
                cu_count = self._GPU_NAME_TO_CU_COUNT.get(gpu_name)
                if cu_count is not None:
                    return cu_count

                # Normalize name by removing common variations
                normalized = gpu_name.replace('(TM)', '').replace('(R)', '').replace('  ', ' ').strip()

                # Try exact match with normalized name
                cu_count = self._GPU_NAME_TO_CU_COUNT.get(normalized)
                if cu_count is not None:
                    return cu_count

                # Try partial match
                for name, count in self._GPU_NAME_TO_CU_COUNT.items():
                    if name in gpu_name or gpu_name in name:
                        return count
                    # Also try normalized partial match
                    norm_name = name.replace('(TM)', '').replace('(R)', '').replace('  ', ' ').strip()
                    if norm_name in normalized or normalized in norm_name:
                        return count

        return None
