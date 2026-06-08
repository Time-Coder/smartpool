from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from ..pool import Pool
from ..threadpool.threadpool import ThreadPool

if TYPE_CHECKING:
    from concurrent.futures import Future

    from ..resource import Resource
    from .infersessionworker import InferSessionWorker


class InferSessionPool(ThreadPool):

    def __init__(
        self, max_workers:int=0, thread_name_prefix:str="InferSessionPool.worker:",
        initializer:Optional[Callable[..., Any]]=None,
        initargs:Tuple[Any, ...]=(),
        initkwargs:Optional[Dict[str, Any]]=None,
        *,
        max_tasks_per_child:Optional[int]=None,
    ):
        ThreadPool.__init__(
            self, max_workers=max_workers, thread_name_prefix=thread_name_prefix,
            initializer=initializer,
            initargs=initargs,
            initkwargs=initkwargs,
            max_tasks_per_child=max_tasks_per_child,
            use_torch=False
        )
        self._use_onnx: bool = True
        self.print_info: bool = False

    def _add_worker(self)->InferSessionWorker:
        from .infersessionworker import InferSessionWorker

        worker = InferSessionWorker(
            len(self._workers), self._thread_name_prefix,
            thread_pool=self,
            initializer=self._initializer,
            initargs=self._initargs,
            initkwargs=self._initkwargs
        )
        self._workers.append(worker)
        return worker

    def submit(
        self, model_path:str,
        args:Optional[Tuple[Any]]=None,
        kwargs:Optional[Dict[str, Any]]=None,
        cpu_mode_res: Optional[Resource] = None,
        gpu_mode_res: Optional[Resource] = None
    )->Future:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        _, ext = os.path.splitext(model_path)
        if ext != ".onnx":
            raise ValueError(f"only support .onnx model, {ext} were given")

        if os.path.getsize(model_path) <= 0:
            raise ValueError("model file size is zero")

        return Pool.submit(
            self, func=model_path,
            args=args, kwargs=kwargs,
            cpu_mode_res=cpu_mode_res,
            gpu_mode_res=gpu_mode_res,
            use_torch=False
        )
