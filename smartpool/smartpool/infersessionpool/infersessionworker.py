from __future__ import annotations

import json
import os
import threading
from typing import TYPE_CHECKING, Dict, Tuple

from cachetools import LRUCache

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

    from ..task import Task
    from .infersessionpool import InferSessionPool

from ..threadpool.threadworker import ThreadWorker


class InferSessionWorker(ThreadWorker):

    _session_lock:threading.Lock = threading.Lock()
    _sessions:Dict[int, Dict[Tuple[str, str, str], InferenceSession]] = {}

    @staticmethod
    def _get_session(model_path:str, provider:tuple):
        model_path = os.path.abspath(model_path).replace("\\", "/")
        provider_name = provider[0]
        provider_options_str = json.dumps(provider[1], sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        tid = 0
        if provider_name == "DmlExecutionProvider":
            tid = threading.get_ident()
        key = (model_path, provider_name, provider_options_str)

        with InferSessionWorker._session_lock:
            if tid not in InferSessionWorker._sessions:
                InferSessionWorker._sessions[tid] = LRUCache(maxsize=10)

            if key not in InferSessionWorker._sessions[tid]:
                from onnxruntime import InferenceSession
                InferSessionWorker._sessions[tid][key] = InferenceSession(model_path, providers=[provider])

            return InferSessionWorker._sessions[tid][key]

    def _clear(self)->None:
        if self.process_or_thread is not None and self.process_or_thread.ident in InferSessionWorker._sessions:
            del InferSessionWorker._sessions[self.process_or_thread.ident]

        ThreadWorker._clear(self)

    def run(self):
        infersession_pool: InferSessionPool = self.pool

        if infersession_pool._initializer is not None:
            infersession_pool._initializer(*infersession_pool._initargs, **infersession_pool._initkwargs)

        while True:
            task:Task = self.task_queue.get()
            if task is None:
                break

            model_path = task.func
            args = task.args
            kwargs = task.kwargs
            provider = task.onnx_provider
            session = self._get_session(model_path, provider)
            if infersession_pool.print_info:
                if "device_id" in provider[1]:
                    print(f"infer with {provider[0]}(device_id={provider[1]['device_id']}) in session {id(session)}")
                else:
                    print(f"infer with {provider[0]} in session {id(session)}")

            try:
                inputs = kwargs
                if args:
                    inputs = dict(zip((node.name for node in session.get_inputs()), args))
                    inputs.update(kwargs)

                result = session.run(None, input_feed=inputs)
                success = True
            except BaseException as e:
                result = e
                success = False

            infersession_pool._on_task_done(task.id, success, result)
