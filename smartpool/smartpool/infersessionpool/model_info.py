from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    import onnxruntime as ort


class ModelInfo:
    model_path: str
    model_name: str
    file_size: int
    inputs: Dict[str, ort.NodeArg]
    outputs: Dict[str, ort.NodeArg]
    signature: inspect.Signature

    _onnx_type_map: Optional[Dict[str, type]] = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = os.path.basename(self.model_path).split(".")[0]
        self.file_size = None
        self.inputs = None
        self.outputs = None
        self.signature = None

        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        _, ext = os.path.splitext(model_path)
        if ext != ".onnx":
            raise ValueError(f"only support .onnx model, {ext} were given")

        self.file_size = os.path.getsize(model_path)
        if self.file_size <= 0:
            raise ValueError("model file size is zero")

        import onnx
        onnx.checker.check_model(model_path, full_check=True)

    def _fetch_info(self, session: ort.InferenceSession) -> None:
        if self.inputs is not None:
            return

        params = []
        self.inputs = {}
        for node in session.get_inputs():
            params.append(inspect.Parameter(node.name, inspect.Parameter.POSITIONAL_OR_KEYWORD))
            self.inputs[node.name] = node

        self.outputs = {}
        for node in session.get_outputs():
            self.outputs[node.name] = node

        self.signature = inspect.Signature(params)

    @staticmethod
    def _get_onnx_to_numpy_map() -> Dict[str, type]:
        if ModelInfo._onnx_type_map is not None:
            return ModelInfo._onnx_type_map

        from onnx import TensorProto, helper
        ModelInfo._onnx_type_map = {}
        for attr_name in dir(TensorProto):
            if attr_name.isupper() and not attr_name.startswith('_'):
                try:
                    onnx_enum = getattr(TensorProto, attr_name)
                    np_dtype = helper.tensor_dtype_to_np_dtype(onnx_enum)
                    ModelInfo._onnx_type_map[f'tensor({attr_name.lower()})'] = np_dtype
                except Exception:
                    continue

        return ModelInfo._onnx_type_map

    def check_args(self, session: ort.InferenceSession, args: Tuple[Any] = (), kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._fetch_info(session)

        if kwargs is None:
            kwargs = {}

        try:
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = bound_args.arguments
        except TypeError as e:
            formatted_msg = f"model '{self.model_name}' arguments error: {str(e)}"
            raise TypeError(formatted_msg) from None

        if self.inputs:
            type_map = ModelInfo._get_onnx_to_numpy_map()

            import numpy as np
            for name, value in kwargs.items():
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"model '{self.model_name}' input node '{name}' need type np.ndarray, {type(value)} were given")

                node = self.inputs[name]

                expected_type = type_map[node.type]
                actual_type = value.dtype
                if actual_type != expected_type:
                    raise TypeError(f"model '{self.model_name}' input node '{name}' need dtype {expected_type}, {actual_type} were given")

                expected_shape = node.shape
                actual_shape = value.shape
                if len(expected_shape) != len(actual_shape):
                    raise ValueError(f"model '{self.model_name}' input node '{name}' need shape {expected_shape}, {actual_shape} were given")

                for i, dim in enumerate(expected_shape):
                    if isinstance(dim, str) or dim is None or dim == -1:
                        continue

                    if actual_shape[i] != dim:
                        raise ValueError(f"model '{self.model_name}' input node '{name}' need shape {expected_shape}, {actual_shape} were given")

        return kwargs

    def check_outputs(self, output_names):
        if output_names is None:
            return
        for name in output_names:
            if name not in self.outputs:
                valid = list(self.outputs.keys())
                raise ValueError(f"model '{self.model_name}' has no output named '{name}'. valid outputs: {valid}")
