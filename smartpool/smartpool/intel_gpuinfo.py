import os
import platform
import subprocess
import threading
import uuid
from typing import Dict, List, Optional

from .gpuinfo import GPUInfo, GPUVendor


class IntelGPUInfo(GPUInfo):

    _lock = threading.Lock()
    _initialized = False
    _device_info: List[Dict] = []

    supported_onnx_providers: List[str] = [
        "OpenVINOExecutionProvider",
        "DnnlExecutionProvider",
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
    def shutdown(cls) -> None:
        if not cls._initialized:
            return

        with cls._lock:
            cls._initialized = False
            cls._device_info.clear()

    @property
    def vendor(self) -> GPUVendor:
        return GPUVendor.INTEL

    @classmethod
    def is_available(cls) -> bool:
        system = platform.system()
        if system == "Windows":
            return cls._detect_windows()
        elif system == "Linux":
            return cls._detect_linux()
        elif system == "Darwin":
            return cls._detect_mac()
        return False

    @classmethod
    def get_device_count(cls) -> int:
        cls._ensure_init()
        return len(cls._device_info)

    @classmethod
    def _detect_windows(cls) -> bool:
        try:
            import wmi
            c = wmi.WMI()
            intel_gpus = [
                gpu for gpu in c.Win32_VideoController()
                if 'Intel' in (gpu.Name or '')
            ]
            return len(intel_gpus) > 0
        except Exception:
            try:
                result = subprocess.run(
                    ['powershell', '-Command',
                     'Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*Intel*"} | Select-Object -Property Name'],
                    capture_output=True, text=True, timeout=5
                )
                return 'Intel' in result.stdout
            except Exception:
                return False

    @classmethod
    def _detect_linux(cls) -> bool:
        try:
            drm_path = '/sys/class/drm'
            if os.path.exists(drm_path):
                for device in os.listdir(drm_path):
                    device_path = os.path.join(drm_path, device)
                    vendor_path = os.path.join(device_path, 'device', 'vendor')
                    if os.path.exists(vendor_path):
                        with open(vendor_path) as f:
                            if f.read().strip() == '0x8086':
                                return True
            return False
        except Exception:
            return False

    @classmethod
    def _detect_mac(cls) -> bool:
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True, timeout=10
            )
            return 'Intel' in result.stdout
        except Exception:
            return False

    @classmethod
    def _init_windows(cls) -> None:
        import wmi
        c = wmi.WMI()
        intel_gpus = [
            gpu for gpu in c.Win32_VideoController()
            if 'Intel' in (gpu.Name or '')
        ]

        cls._device_info = []
        for gpu in intel_gpus:
            mem_raw = gpu.AdapterRAM
            mem_total = None
            if mem_raw is not None:
                try:
                    val = int(mem_raw)
                    if val > 0:
                        mem_total = val
                except (ValueError, TypeError):
                    pass

            cls._device_info.append({
                'name': gpu.Name,
                'driver_version': gpu.DriverVersion,
                'mem_total': mem_total,
            })

    @classmethod
    def _init_linux(cls) -> None:
        try:
            drm_path = '/sys/class/drm'
            cls._device_info = []

            for device in os.listdir(drm_path):
                device_path = os.path.join(drm_path, device)
                vendor_path = os.path.join(device_path, 'device', 'vendor')

                if os.path.exists(vendor_path):
                    with open(vendor_path) as f:
                        if f.read().strip() == '0x8086':
                            name = "Intel GPU"
                            uevent_path = os.path.join(device_path, 'device', 'uevent')
                            if os.path.exists(uevent_path):
                                with open(uevent_path) as f:
                                    for line in f:
                                        if line.startswith('PCI_SLOT_NAME'):
                                            name = f"Intel GPU ({line.split('=')[1].strip()})"
                                            break

                            cls._device_info.append({'name': name})
        except Exception:
            cls._device_info = []

    @classmethod
    def _init_mac(cls) -> None:
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True, timeout=10
            )

            if 'Intel' in result.stdout:
                cls._device_info = [{'name': 'Intel GPU'}]
            else:
                cls._device_info = []
        except Exception:
            cls._device_info = []

    def _get_device_info(self) -> Dict:
        if 0 <= self._device_id < len(self._device_info):
            return self._device_info[self._device_id]
        return {}

    def _fetch_name(self) -> str:
        info = self._get_device_info()
        return info.get('name', 'Unknown Intel GPU')

    def _fetch_uuid(self) -> Optional[uuid.UUID]:
        return None

    def _fetch_serial(self) -> Optional[str]:
        return None

    def _fetch_driver_version(self) -> Optional[str]:
        info = self._get_device_info()
        return info.get('driver_version')

    def _fetch_memory_info(self) -> tuple[Optional[int], Optional[int]]:
        info = self._get_device_info()
        mem_total = info.get('mem_total')
        if mem_total:
            return (mem_total, 0)
        return (None, None)

    def _fetch_utilization(self) -> Optional[float]:
        system = platform.system()

        if system == "Windows":
            try:
                import wmi
                c = wmi.WMI()
                for adapter in c.Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapter():
                    name = adapter.Name or ''
                    if 'Intel' in name:
                        util = getattr(adapter, 'UtilizationPercentage', None)
                        if util is not None:
                            return int(util) / 100.0
            except Exception:
                pass

        return None

    def _fetch_temperature(self) -> Optional[int]:
        return None

    def _fetch_display_mode(self) -> Optional[bool]:
        return None

    def _fetch_display_active(self) -> Optional[bool]:
        return None

    def _fetch_num_cores(self) -> int:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for platform in platforms:
            vendor = platform.vendor.lower()
            if 'intel' not in vendor:
                continue

            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if 0 <= self._device_id < len(devices):
                device = devices[self._device_id]
                return device.max_compute_units * device.max_work_group_size
            
        raise RuntimeError("cannot fetch num cores of current Intel GPU")
