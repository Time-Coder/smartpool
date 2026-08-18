from __future__ import annotations
from typing import Optional, List, Union, Tuple, Dict, TYPE_CHECKING

from ..resource import Resource
from .model_info import ModelInfo
from ..futures import Future

if TYPE_CHECKING:
    import onnxruntime as ort


class InferModel:

    def __init__(
        self, model_path:str,
        pool_name: str = "default",
        output_names: Optional[List[str]] = None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
        check_args: bool = True,
        providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
        run_options: Optional[ort.RunOptions] = None,
        use_io_binding: bool = False,
        copy_outputs_to_cpu: bool = True,
        chunksize: int = 1
    ):
        self.model_path: str = model_path
        self.pool_name: str = pool_name
        self.output_names: Optional[List[str]] = output_names
        self.cpu_mode_res: Optional[Resource] = cpu_mode_res
        self.gpu_mode_res: Optional[Resource] = gpu_mode_res
        self.check_args: bool = check_args
        self.providers: Optional[List[Union[str, Tuple[str, Dict]]]] = providers
        self.run_options: Optional[ort.RunOptions] = run_options
        self.use_io_binding: bool = use_io_binding
        self.copy_outputs_to_cpu: bool = copy_outputs_to_cpu
        self.chunksize: int = chunksize

    @property
    def model_info(self)->ModelInfo:
        from .infer_session_pool import InferSessionPool
        return InferSessionPool.model_info(self.model_path)

    def __call__(self, *args, **kwargs)->Future:
        from .infer_session_pool import InferSessionPool

        pool = InferSessionPool.instance(self.pool_name)
        future = pool.submit(
            model_path=self.model_path,
            args=args, kwargs=kwargs,
            output_names=self.output_names,
            cpu_mode_res=self.cpu_mode_res,
            gpu_mode_res=self.gpu_mode_res,
            check_args=self.check_args,
            providers=self.providers,
            run_options=self.run_options,
            use_io_binding=self.use_io_binding,
            copy_outputs_to_cpu=self.copy_outputs_to_cpu,
            chunksize=self.chunksize
        )
        return future
