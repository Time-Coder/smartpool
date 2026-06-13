from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Dict, Any, Optional

from ..worker import Worker

if TYPE_CHECKING:
    import onnxruntime as ort
    import numpy as np
    from ..task import Task
    from .infersessionpool import InferSessionPool


class InferSessionWorker(Worker):

    _session_options: Optional[ort.SessionOptions] = None

    def __init__(self, infer_session_pool:InferSessionPool, provider: Tuple[str, Dict[str, Any]], gpu_index: int):
        Worker.__init__(
            self, infer_session_pool,
            task_queue_cls=None,
            task_queue_args=(),
            task_queue_kwargs={},
        )

        self.provider: Tuple[str, Dict[str, Any]] = provider
        self.running_count: int = 0
        self._size_scale: int = 1
        self._gpu_index: int = gpu_index
        if gpu_index < 0:
            self._size_scale = 2
        else:
            self._size_scale = 20
        self._model_size: int = self._size_scale * infer_session_pool._model_info.file_size

        self._take_resource()

    @property
    def session(self)->ort.InferenceSession:
        return self.executor
    
    @property
    def infer_session_pool(self)->InferSessionPool:
        return self.pool

    @property
    def is_working(self)->bool:
        return self._is_working

    @is_working.setter
    def is_working(self, is_working:bool)->None:
        is_working = (self.running_count > 0)

        if self._is_working == is_working:
            return

        self._is_working = is_working

        if is_working:
            with Worker._total_working_count_lock:
                Worker._total_working_count += 1
                self.pool._workers_working_count += 1
        else:
            with Worker._total_working_count_lock:
                Worker._total_working_count -= 1
                self.pool._workers_working_count -= 1

    def _take_resource(self)->None:
        with self.infer_session_pool._sys_info_lock:
            self.infer_session_pool._sys_info.cpu_mem_free -= self._model_size
            if self._gpu_index >= 0:
                self.infer_session_pool._sys_info.gpu_infos[self._gpu_index].mem_free -= self._model_size

    def _release_resource(self)->None:
        with self.infer_session_pool._sys_info_lock:
            self.infer_session_pool._sys_info.cpu_mem_free += self._model_size
            if self._gpu_index >= 0:
                self.infer_session_pool._sys_info.gpu_infos[self._gpu_index].mem_free += self._model_size

    @staticmethod
    def _get_session_options() -> ort.SessionOptions:
        if InferSessionWorker._session_options is None:
            import onnxruntime as ort
            InferSessionWorker._session_options = ort.SessionOptions()
            InferSessionWorker._session_options.enable_cpu_mem_arena = True
            InferSessionWorker._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return InferSessionWorker._session_options

    def add_task(self, task:Task)->None:
        self.start()
        if self.infer_session_pool.print_info:
            print("infer with provider", self.provider)

        self.session.run_async(task.output_names, input_feed=task.kwargs, callback=InferSessionWorker.callback, user_data=(self, task.id))
        with self.infer_session_pool._running_count_lock:
            self.running_count += 1
            self.infer_session_pool._running_count += 1

        task.future.set_running_or_notify_cancel()

    def start(self)->None:
        if self.executor is not None:
            return

        import onnxruntime as ort

        infer_session_pool: InferSessionPool = self.pool
        model_path: str = infer_session_pool._model_info.model_path
        option: ort.SessionOptions = self._get_session_options()
        self.executor = ort.InferenceSession(model_path, option, providers=[self.provider])
        self.executor.disable_fallback()

    def _clear(self)->None:
        if self.executor is not None:
            self._release_resource()

        Worker._clear(self)
        self.provider = None
        self._gpu_index = -1
        self.running_count = 0
        self._model_size = 0
        self._size_scale = 1

    def join(self)->None:
        self._clear()

    def callback(results:np.ndarray, user_data: Tuple[InferSessionWorker, str], error_str: str) -> None:
        self, task_id = user_data
        if error_str:
            success = False
            result = RuntimeError(error_str)
        else:
            success = True
            result = results

        with self.infer_session_pool._running_count_lock:
            self.running_count -= 1
            self.infer_session_pool._running_count -= 1

        self.infer_session_pool._on_task_done(task_id, success, result)
