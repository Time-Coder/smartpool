from typing import Dict, Iterator, List, Optional, Type

from .amd_gpuinfo import AMDGPUInfo
from .gpuinfo import GPUInfo, GPUInfoSnapshot, GPUVendor
from .intel_gpuinfo import IntelGPUInfo
from .nvidia_gpuinfo import NvidiaGPUInfo


class MetaGPUInfos(type):

    __gpu_infos: Optional[List[GPUInfo]] = None
    __n_devices: Optional[int] = None

    @staticmethod
    def _dxgi_enumerate_name_map() -> Dict[str, List[int]]:
        import platform
        if platform.system() != "Windows":
            return {}

        import ctypes
        from ctypes import wintypes

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", wintypes.BYTE * 8),
            ]

        class _DXGI_ADAPTER_DESC(ctypes.Structure):
            _fields_ = [
                ("Description", wintypes.WCHAR * 128),
                ("VendorId", wintypes.UINT),
                ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT),
                ("Revision", wintypes.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", wintypes.LARGE_INTEGER),
            ]

        IID_IDXGIFactory = _GUID(
            0x7b7166ec, 0x21c7, 0x44ae,
            (wintypes.BYTE * 8)(0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69),
        )

        try:
            dxgi_lib = ctypes.WinDLL("dxgi.dll")
        except Exception:
            return {}

        CreateDXGIFactory = dxgi_lib.CreateDXGIFactory
        CreateDXGIFactory.restype = wintypes.LONG
        CreateDXGIFactory.argtypes = [
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]

        factory = ctypes.c_void_p()
        hr = CreateDXGIFactory(ctypes.byref(IID_IDXGIFactory), ctypes.byref(factory))
        if hr != 0 or not factory:
            return {}

        result: Dict[str, List[int]] = {}
        ptr_size = ctypes.sizeof(ctypes.c_void_p)
        fac_vtable = ctypes.c_void_p.from_address(factory.value).value

        EnumAdaptersFunc = ctypes.WINFUNCTYPE(
            wintypes.LONG, ctypes.c_void_p, wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
        )
        GetDescFunc = ctypes.WINFUNCTYPE(
            wintypes.LONG, ctypes.c_void_p,
            ctypes.POINTER(_DXGI_ADAPTER_DESC),
        )
        ReleaseFunc = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)

        # IDXGIFactory vtable: IUnknown(0-2) + IDXGIObject(3-6) + IDXGIFactory(7+)
        enum_adapters = EnumAdaptersFunc(
            ctypes.c_void_p.from_address(fac_vtable + 7 * ptr_size).value,
        )
        factory_release = ReleaseFunc(
            ctypes.c_void_p.from_address(fac_vtable + 2 * ptr_size).value,
        )

        i = 0
        while i < 64:
            adapter = ctypes.c_void_p()
            hr = enum_adapters(factory, i, ctypes.byref(adapter))
            if hr != 0:
                break

            if adapter:
                try:
                    ad_vtable = ctypes.c_void_p.from_address(adapter.value).value
                    # IDXGIAdapter vtable: IUnknown(0-2) + IDXGIObject(3-6) + EnumOutputs(7) + GetDesc(8)
                    get_desc = GetDescFunc(
                        ctypes.c_void_p.from_address(ad_vtable + 8 * ptr_size).value,
                    )
                    adapter_release = ReleaseFunc(
                        ctypes.c_void_p.from_address(ad_vtable + 2 * ptr_size).value,
                    )

                    desc = _DXGI_ADAPTER_DESC()
                    if get_desc(adapter, ctypes.byref(desc)) == 0:
                        name = desc.Description
                        if name:
                            result.setdefault(name.lower().strip(), []).append(i)

                    adapter_release(adapter)
                except Exception:
                    pass

            i += 1

        factory_release(factory)
        return result

    @staticmethod
    def __init() -> None:
        if MetaGPUInfos.__gpu_infos is not None:
            return

        MetaGPUInfos.__gpu_infos = []
        MetaGPUInfos.__n_devices = 0

        gpu_classes:List[Type[GPUInfo]] = [
            NvidiaGPUInfo,
            AMDGPUInfo,
            IntelGPUInfo,
        ]

        index = 0
        dml_map = MetaGPUInfos._dxgi_enumerate_name_map()
        for gpu_class in gpu_classes:
            count = gpu_class.get_device_count()
            if count == 0:
                continue

            for i in range(count):
                gpu = gpu_class(i, index)
                MetaGPUInfos.__gpu_infos.append(gpu)
                index += 1

                if not dml_map:
                    continue

                gpu_name = gpu.name.lower().strip()
                if gpu_name in dml_map and dml_map[gpu_name]:
                    gpu.dml_id = dml_map[gpu_name][0]
                else:
                    for dml_name, dml_ids in dml_map.items():
                        if dml_ids and gpu_name and (
                            gpu_name in dml_name or dml_name in gpu_name
                        ):
                            gpu.dml_id = dml_ids[0]
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
