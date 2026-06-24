from typing import TYPE_CHECKING
from .task import Task

if TYPE_CHECKING:
    from .pool import Pool



class ChunkTask(Task):

    def __init__(self, pool: Pool, chunksize: int):
        Task.__init__(
            self, pool=pool, func=None, args=None, kwargs=None,
            cpu_mode_res=None, gpu_mode_res=None, check_args=False, chunksize=chunksize,
            use_torch=False, device_changeable=False, calculate_module_deps=False
        )