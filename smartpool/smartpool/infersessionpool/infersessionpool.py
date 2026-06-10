from __future__ import annotations

import os
import threading
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Set, List

from ..pool import Pool

if TYPE_CHECKING:
    from concurrent.futures import Future
    import onnxruntime as ort
    from ..resource import Resource

@dataclass
class ModelInfo:
    file_size: int
    model_name: str
    inputs: List[ort.NodeArg]
    signature: inspect.Signature

class InferSessionPool(Pool):

    _thread_safe_providers: Set[str] = {
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    }

    _model_infos: Dict[str, ModelInfo] = {}
    _session_options: Optional[ort.SessionOptions] = None

    def __init__(self, model_path:str, max_workers:int=0):
        model_path: str = os.path.abspath(model_path).replace("\\", "/")
        if model_path not in InferSessionPool._valid_model_paths:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(model_path)

            _, ext = os.path.splitext(model_path)
            if ext != ".onnx":
                raise ValueError(f"only support .onnx model, {ext} were given")

            file_size = os.path.getsize(model_path)
            if file_size <= 0:
                raise ValueError("model file size is zero")
            
            import onnx
            onnx.checker.check_model(model_path, full_check=True)

            InferSessionPool._valid_model_file_sizes[model_path] = file_size
        
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
        self._model_path: str = model_path
        self._model_file_size: int = InferSessionPool._valid_model_file_sizes[model_path]
        self._min_mem_size: int = int(1.5 * self._model_file_size)
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
            self._cpu_session = ort.InferenceSession(self._model_path, InferSessionPool._get_session_options(), providers=[("CPUExecutionProvider", {})])

        
    def _get_model_info(self)->ModelInfo:
        if self._model_path in InferSessionPool._model_infos:
            return InferSessionPool._model_infos[self._model_path]
        
    @staticmethod
    def _get_session_options()->ort.SessionOptions:
        if InferSessionPool._session_options is None:
            InferSessionPool._session_options = ort.SessionOptions()
            InferSessionPool._session_options.enable_cpu_mem_arena = True 
            InferSessionPool._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return InferSessionPool._session_options

    @staticmethod
    def _get_session(model_path: str, provider: Tuple[str, Dict[str, Any]]):
        from cachetools import LRUCache
        import json

        model_path = os.path.abspath(model_path).replace("\\", "/")
        provider_name = provider[0]
        provider_options_str = json.dumps(provider[1], sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        if provider_name in InferSessionPool._thread_safe_providers:
            thread = None
        else:
            thread = threading.current_thread()
        key = (model_path, provider_name, provider_options_str)

        with InferSessionPool._session_lock:
            if thread not in InferSessionPool._sessions:
                InferSessionPool._sessions[thread] = LRUCache(maxsize=10)

            if key not in InferSessionPool._sessions[thread]:
                import onnxruntime as ort
                InferSessionPool._sessions[thread][key] = ort.InferenceSession(model_path, InferSessionPool._get_session_options(), providers=[provider])

            return InferSessionPool._sessions[thread][key]

    def submit(
        self, model_path:str,
        args:Optional[Tuple[Any]]=None,
        kwargs:Optional[Dict[str, Any]]=None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None
    )->Future:
        self._modify_res(model_path, cpu_mode_res, gpu_mode_res)

        return Pool.submit(
            self, func=model_path,
            args=args, kwargs=kwargs,
            cpu_mode_res=cpu_mode_res,
            gpu_mode_res=gpu_mode_res,
            use_torch=False,
            device_changeable=False
        )
