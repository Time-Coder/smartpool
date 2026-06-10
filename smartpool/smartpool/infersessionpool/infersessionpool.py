from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Set

from ..pool import Pool

if TYPE_CHECKING:
    from concurrent.futures import Future
    import onnxruntime as ort
    from ..resource import Resource
    from .model_info import ModelInfo


class InferSessionPool(Pool):

    _thread_safe_providers: Set[str] = {
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    }

    _model_infos: Dict[str, ModelInfo] = {}
    _session_options: Optional[ort.SessionOptions] = None

    def __init__(self, model_path:str, max_workers:int=0):
        from .model_info import ModelInfo
        
        model_path: str = os.path.abspath(model_path).replace("\\", "/")
        if model_path not in InferSessionPool._model_infos:
            InferSessionPool._model_infos[model_path] = ModelInfo(model_path)
        
        from .infersessionworker import InferSessionWorker
        
        Pool.__init__(
            self, max_workers = max_workers,
            initializer = None,
            initargs = (),
            initkwargs = None,
            result_queue_cls = None,
            worker_cls = InferSessionWorker,
            use_onnx = True,
        )
        self.print_info: bool = False
        self._model_info: ModelInfo = InferSessionPool._model_infos[model_path]
        self._session_lock: threading.Lock = threading.Lock()
        self._sessions: Dict[threading.Thread, Dict[Tuple[str, str, str], ort.InferenceSession]] = {}
        self._cpu_session: Optional[ort.InferenceSession] = None

    def submit(
        self, args:Optional[Tuple[Any]]=None, kwargs:Optional[Dict[str, Any]]=None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None,
    )->Future:
        if self._cpu_session is None:
            import onnxruntime as ort
            self._cpu_session = ort.InferenceSession(self._model_info.model_path, InferSessionPool._get_session_options(), providers=[("CPUExecutionProvider", {})])

        kwargs = self._model_info.check_args(self._cpu_session, args, kwargs)

    @staticmethod
    def _get_session_options()->ort.SessionOptions:
        if InferSessionPool._session_options is None:
            InferSessionPool._session_options = ort.SessionOptions()
            InferSessionPool._session_options.enable_cpu_mem_arena = True
            InferSessionPool._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return InferSessionPool._session_options
