from typing import Iterator, List, Optional

from .gpuinfo import GPUInfo, GPUVendor
from .nvidia_gpu import NvidiaGPUInfo
from .intel_gpu import IntelGPUInfo
from .amd_gpu import AMDGPUInfo


class MetaGPUInfos(type):

    __gpu_infos: Optional[List[GPUInfo]] = None
    __n_devices: Optional[int] = None

    @staticmethod
    def __init() -> None:
        if MetaGPUInfos.__gpu_infos is None:
            MetaGPUInfos.__gpu_infos = []
            MetaGPUInfos.__n_devices = 0

            gpu_classes = [
                NvidiaGPUInfo,
                IntelGPUInfo,
                AMDGPUInfo,
            ]

            for gpu_class in gpu_classes:
                if not gpu_class.is_available():
                    continue

                count = gpu_class.get_device_count()
                for i in range(count):
                    MetaGPUInfos.__gpu_infos.append(gpu_class(i))

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
    def snapshot() -> List[dict]:
        result = []
        for gpu in GPUInfos:
            gpu.update()
            snapshot = {
                'vendor': gpu.vendor.value,
                'id': gpu.device_id,
                'name': gpu.name,
                'device': gpu.device,
                'uuid': str(gpu.uuid) if gpu.uuid else None,
                'serial': gpu.serial,
                'driver_version': gpu.driver_version,
                'mem_total': gpu.mem_total,
                'mem_used': gpu.mem_used,
                'mem_free': gpu.mem_free,
                'load': gpu.load,
                'temperature': gpu.temperature,
                'n_cores': gpu.n_cores,
                'n_cores_used': gpu.n_cores_used,
                'n_cores_free': gpu.n_cores_free,
            }
            result.append(snapshot)
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
