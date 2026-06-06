class DataSize:
    B = 1
    KB = 1024 * B
    MB = 1024 * KB
    GB = 1024 * MB
    TB = 1024 * GB
    PB = 1024 * TB
    

class Resource:

    def __init__(self, cpu_cores_in_python: float = 0, cpu_cores_out_of_python: float = 0,
                 cpu_mem: int = 0, gpu_cores: float = 0, gpu_mem: int = 0):
        self.cpu_cores_in_python = cpu_cores_in_python
        self.cpu_cores_out_of_python = cpu_cores_out_of_python
        self.cpu_mem = cpu_mem
        self.gpu_cores = gpu_cores
        self.gpu_mem = gpu_mem

    @property
    def cpu_cores(self) -> float:
        return self.cpu_cores_in_python + self.cpu_cores_out_of_python