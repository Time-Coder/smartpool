from typing import Iterator, List, Optional, Type, Dict

from .gpuinfo import GPUInfo, GPUVendor, GPUInfoSnapshot

from .nvidia_gpuinfo import NvidiaGPUInfo
from .intel_gpuinfo import IntelGPUInfo
from .amd_gpuinfo import AMDGPUInfo


class MetaGPUInfos(type):

    __gpu_infos: Optional[List[GPUInfo]] = None
    __n_devices: Optional[int] = None

    @staticmethod
    def _dxgi_enumerate_name_map() -> Dict[str, int]:
        import platform
        if platform.system() != "Windows":
            return {}
        
        import wmi
        c = wmi.WMI()
        adapters = list(c.Win32_VideoController())
        return {
            (getattr(a, 'Name', None) or getattr(a, 'Description', '') or '').lower().strip(): i
            for i, a in enumerate(adapters)
        }

    @staticmethod
    def __init() -> None:
        if MetaGPUInfos.__gpu_infos is not None:
            return
        
        MetaGPUInfos.__gpu_infos = []
        MetaGPUInfos.__n_devices = 0

        gpu_classes:List[Type[GPUInfo]] = [
            NvidiaGPUInfo,
            IntelGPUInfo,
            AMDGPUInfo,
        ]

        index = 0
        dml_map = MetaGPUInfos._dxgi_enumerate_name_map()
        for gpu_class in gpu_classes:
            if not gpu_class.is_available():
                continue

            count = gpu_class.get_device_count()
            for i in range(count):
                gpu = gpu_class(i, index)
                MetaGPUInfos.__gpu_infos.append(gpu)
                index += 1

                if not dml_map:
                    continue

                gpu_name = gpu.name.lower().strip()
                if gpu_name in dml_map:
                    gpu._dml_id = dml_map[gpu_name]
                else:
                    for dml_name, dml_id in dml_map.items():
                        if gpu_name and (gpu_name in dml_name or dml_name in gpu_name):
                            gpu._dml_id = dml_id
                            break

            MetaGPUInfos.__n_devices += count

    def __len__(self) -> int:
        if MetaGPUInfos.__n_devices is None:
            MetaGPUInfos.__init()

        return MetaGPUInfos.__n_devices

    def __getitem__(self, index: int) -> GPUInfo:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")

        return MetaGPUInfos.__gpu_infos[index]

    def __iter__(self) -> Iterator[GPUInfo]:
        if MetaGPUInfos.__gpu_infos is None:
            MetaGPUInfos.__init()

        return iter(MetaGPUInfos.__gpu_infos)


class GPUInfos(metaclass=MetaGPUInfos):

    @staticmethod
    def update() -> None:
        for gpu in GPUInfos:
            gpu.update()

    @staticmethod
    def snapshot(
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
    ) -> List[GPUInfoSnapshot]:
        if not any([name, uuid, serial, driver_version, mem_total, mem_used,
                     load, temperature, display_mode, display_active,
                     n_cores, n_cores_used]):
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

        result = []
        for gpu in GPUInfos:
            gpu.update()
            result.append(gpu.snapshot(
                name=name, uuid=uuid, serial=serial,
                driver_version=driver_version, mem_total=mem_total,
                mem_used=mem_used, load=load, temperature=temperature,
                display_mode=display_mode, display_active=display_active,
                n_cores=n_cores, n_cores_used=n_cores_used,
            ))
        return result

    @staticmethod
    def get_by_vendor(vendor: GPUVendor) -> List[GPUInfo]:
        return [gpu for gpu in GPUInfos if gpu.vendor == vendor]

    @staticmethod
    def get_nvidia_gpus() -> List[GPUInfo]:
        return GPUInfos.get_by_vendor(GPUVendor.NVIDIA)

    @staticmethod
    def get_intel_gpus() -> List[GPUInfo]:
        return GPUInfos.get_by_vendor(GPUVendor.INTEL)

    @staticmethod
    def get_amd_gpus() -> List[GPUInfo]:
        return GPUInfos.get_by_vendor(GPUVendor.AMD)
