from __future__ import annotations
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .infersessiontask import InfersessionTask
from ..resource import Resource

if TYPE_CHECKING:
    from .infersessionpool import InferSessionPool
    from ..futures import Future
    from ..task import Task
    from .model_info import ModelInfo


class InferChunkTask(InfersessionTask):

    def __init__(
        self, pool: InferSessionPool,
        model_path: str,
        model_info: ModelInfo,
        output_names: Optional[List[str]],
        run_options: Any,
        use_io_binding: bool,
        copy_outputs_to_cpu: bool,
        chunksize: int,
        cpu_mode_res: Resource,
        gpu_mode_res: Resource,
        user_providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]]
    ):
        self._sub_tasks: List[Task] = []
        self._kwargs_list: List[Dict[str, Any]] = []
        self._model_info: ModelInfo = model_info
        self._submitted: bool = False
        self.last_add_time: float = 0.0
        self._key: Tuple[str, int] = (model_path, chunksize)

        super().__init__(
            infer_session_pool=pool,
            model_path=model_path,
            kwargs={},
            cpu_mode_res=cpu_mode_res,
            gpu_mode_res=gpu_mode_res,
            check_args=False,
            output_names=output_names,
            user_providers=user_providers,
            run_options=run_options,
            use_io_binding=True,
            copy_outputs_to_cpu=copy_outputs_to_cpu,
            chunksize=chunksize
        )
        self.future.add_done_callback(self._fan_out_batch_results)

    def add_task(self, sub_task: Task) -> None:
        if self._submitted:
            return

        self._sub_tasks.append(sub_task)
        self._kwargs_list.append(sub_task.kwargs)
        self._update_res(self.cpu_mode_res, sub_task.cpu_mode_res)
        self._update_res(self.gpu_mode_res, sub_task.gpu_mode_res)

        if sub_task.use_torch:
            self.use_torch = True
        if sub_task.module_deps:
            self.module_deps.update(sub_task.module_deps)

        try:
            self.pool._validate_resource_feasibility(self)
        except Exception as e:
            for st in self._sub_tasks:
                st.future.set_exception(e)
            if self._key in self.pool._chunk_tasks:
                del self.pool._chunk_tasks[self._key]
            return

        self.last_add_time = time.time()
        if len(self._sub_tasks) >= self.chunksize:
            self.submit()

    def submit(self) -> None:
        if self._submitted:
            return
        self._submitted = True
        if self._key in self.pool._chunk_tasks:
            del self.pool._chunk_tasks[self._key]
        self.pool._submit(self)

    def exec(self) -> Tuple[bool, Any]:
        first_node = next(iter(self._model_info.inputs.values()))
        first_dim = first_node.shape[0] if first_node.shape else None
        batch_dynamic = (
            first_dim is None or
            not isinstance(first_dim, (int, float)) or
            first_dim <= 0
        )

        if batch_dynamic and not self._has_ortvalue_inputs():
            merged = {}
            for name in self._model_info.inputs:
                parts = [self._to_numpy(kw[name]) for kw in self._kwargs_list]
                merged[name] = np.concatenate(parts, axis=0)

            batched = self.session_info.run_with_iobinding(
                self.output_names, merged, self.run_options, self.copy_outputs_to_cpu
            )

            results = []
            n = len(self._sub_tasks)
            for i in range(n):
                item = [out[i:i+1] for out in batched]
                results.append((True, item))
            return True, results
        else:
            results = []
            for kw in self._kwargs_list:
                try:
                    out = self.session_info.run_with_iobinding(
                        self.output_names, kw, self.run_options, self.copy_outputs_to_cpu
                    )
                    results.append((True, out))
                except Exception as e:
                    results.append((False, e))
            return True, results

    def _has_ortvalue_inputs(self) -> bool:
        for kw in self._kwargs_list:
            for v in kw.values():
                if v.__class__.__name__ == "OrtValue":
                    return True
        return False

    @staticmethod
    def _to_numpy(v: Any) -> np.ndarray:
        cls = v.__class__.__name__
        if cls == "Tensor":
            import torch
            return v.detach().cpu().numpy()
        if cls == "OrtValue":
            return v.numpy()
        return v

    def _fan_out_batch_results(self, future: Future) -> None:
        if future.cancelled():
            for task in self._sub_tasks:
                task.future.cancel()
            return

        try:
            results = future.result()
        except Exception as exc:
            for task in self._sub_tasks:
                task.future.set_exception(exc)
            return

        for (success, result), task in zip(results, self._sub_tasks):
            if success:
                task.future.set_result(result)
            else:
                task.future.set_exception(result)

    @staticmethod
    def _update_res(src_res: Resource, target_res: Resource) -> None:
        if target_res.cpu_cores_in_python > src_res.cpu_cores_in_python:
            src_res.cpu_cores_in_python = target_res.cpu_cores_in_python
        if target_res.cpu_cores_out_of_python > src_res.cpu_cores_out_of_python:
            src_res.cpu_cores_out_of_python = target_res.cpu_cores_out_of_python
        if target_res.cpu_mem > src_res.cpu_mem:
            src_res.cpu_mem = target_res.cpu_mem
        if target_res.gpu_cores > src_res.gpu_cores:
            src_res.gpu_cores = target_res.gpu_cores
        if target_res.gpu_mem > src_res.gpu_mem:
            src_res.gpu_mem = target_res.gpu_mem
