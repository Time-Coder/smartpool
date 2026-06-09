from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from ..pool import Pool
from ..threadpool.threadpool import ThreadPool

if TYPE_CHECKING:
    from concurrent.futures import Future
    
    from onnxruntime import InferenceSession
    from ..resource import Resource


class InferSessionPool(ThreadPool):

    _session_lock:threading.Lock = threading.Lock()
    _sessions:Dict[int, Dict[Tuple[str, str, str], InferenceSession]] = {}

    def __init__(
        self, max_workers:int=0, thread_name_prefix:str="InferSessionPool.worker:",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
    ):
        from .infersessionworker import InferSessionWorker
        
        ThreadPool.__init__(
            self, max_workers=max_workers, thread_name_prefix=thread_name_prefix,
            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,
            max_tasks_per_child=max_tasks_per_child,
            use_torch=False
        )
        self._worker_cls = InferSessionWorker
        self._use_onnx: bool = True
        self.print_info: bool = False

    def _check_model_path(self, model_path:str)->None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        _, ext = os.path.splitext(model_path)
        if ext != ".onnx":
            raise ValueError(f"only support .onnx model, {ext} were given")

        if os.path.getsize(model_path) <= 0:
            raise ValueError("model file size is zero")
        
    @staticmethod
    def _get_session(model_path:str, provider:tuple):
        from cachetools import LRUCache
        import json

        model_path = os.path.abspath(model_path).replace("\\", "/")
        provider_name = provider[0]
        provider_options_str = json.dumps(provider[1], sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        tid = 0
        key = (model_path, provider_name, provider_options_str)

        with InferSessionPool._session_lock:
            if tid not in InferSessionPool._sessions:
                InferSessionPool._sessions[tid] = LRUCache(maxsize=10)

            if key not in InferSessionPool._sessions[tid]:
                from onnxruntime import InferenceSession
                InferSessionPool._sessions[tid][key] = InferenceSession(model_path, providers=[provider])

            return InferSessionPool._sessions[tid][key]

    def submit(
        self, model_path:str,
        args:Optional[Tuple[Any]]=None,
        kwargs:Optional[Dict[str, Any]]=None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None
    )->Future:
        self._check_model_path(model_path)

        return Pool.submit(
            self, func=model_path,
            args=args, kwargs=kwargs,
            cpu_mode_res=cpu_mode_res,
            gpu_mode_res=gpu_mode_res,
            use_torch=False,
            device_changeable=False
        )
