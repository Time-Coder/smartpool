from __future__ import annotations

import inspect
import os
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np
    import onnx


class NodeInfo:

    def __init__(self, node: onnx.ValueInfoProto)->None:
        import onnx

        self.name = node.name
        self.dtype = onnx.helper.tensor_dtype_to_np_dtype(node.type.tensor_type.elem_type)
        dims = node.type.tensor_type.shape.dim
        shape = []
        for dim in dims:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(None)
        self.shape = tuple(shape)


class ModelInfo:

    def __init__(self, model_path: str):
        self._model_path: str = model_path
        self._model_name: str = os.path.basename(model_path).split(".")[0]
        self._file_size: Optional[int] = None
        self._file_mtime: Optional[float] = None
        self._inputs: Optional[Dict[str, NodeInfo]] = None
        self._outputs: Optional[Dict[str, NodeInfo]] = None
        self._signature: Optional[inspect.Signature] = None
        self._support_batch_input: Optional[bool] = None
        self.load()

    @property
    def model_path(self)->str:
        return self._model_path

    @property
    def model_name(self)->str:
        return self._model_name

    @property
    def file_size(self)->int:
        self.load()
        return self._file_size

    @property
    def file_mtime(self)->float:
        self.load()
        return self._file_mtime

    @property
    def inputs(self)->Dict[str, NodeInfo]:
        self.load()
        return self._inputs

    @property
    def outputs(self)->Dict[str, NodeInfo]:
        self.load()
        return self._outputs

    @property
    def signature(self)->inspect.Signature:
        self.load()
        return self._signature

    @property
    def support_batch_input(self)->bool:
        self.load()
        return self._support_batch_input

    def load(self):
        if (
            self._file_size is not None and
            self._file_mtime is not None and
            self._inputs is not None and
            self._outputs is not None and
            self._signature is not None and
            self._support_batch_input is not None
        ):
            mtime = 0
            if os.path.isfile(self._model_path):
                mtime = os.path.getmtime(self._model_path)

            if self._file_mtime == mtime:
                return
            else:
                self._file_size: Optional[int] = None
                self._file_mtime: Optional[float] = None
                self._inputs: Optional[Dict[str, NodeInfo]] = None
                self._outputs: Optional[Dict[str, NodeInfo]] = None
                self._signature: Optional[inspect.Signature] = None
                self._support_batch_input: Optional[bool] = None

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(self._model_path)

        _, ext = os.path.splitext(self._model_path)
        if ext != ".onnx":
            raise ValueError(f"only support .onnx model, {ext} were given")

        self._file_size = os.path.getsize(self._model_path)
        if self._file_size <= 0:
            raise ValueError("model file size is zero")

        self._file_mtime = os.path.getmtime(self._model_path)

        import onnx

        model = onnx.load(self._model_path)
        onnx.checker.check_model(model, full_check=True)

        params = []
        inputs = {}
        outputs = {}
        for node in model.graph.input:
            params.append(inspect.Parameter(node.name, inspect.Parameter.POSITIONAL_OR_KEYWORD))
            node_info = NodeInfo(node)
            inputs[node.name] = node_info
            if self._support_batch_input is None:
                first_dim = node_info.shape[0] if node_info.shape else None
                self._support_batch_input = (
                    first_dim is None or
                    not isinstance(first_dim, (int, float)) or
                    first_dim <= 0
                )

        for node in model.graph.output:
            outputs[node.name] = NodeInfo(node)

        self._inputs = inputs
        self._outputs = outputs
        self._signature = inspect.Signature(params)

    @staticmethod
    def get_dtype(value: Any) -> np.dtype:
        cls_name = value.__class__.__name__

        if cls_name == "Tensor":
            import torch
            return torch.empty(0, dtype=value.dtype).numpy().dtype

        if cls_name == "OrtValue":
            import onnx
            return onnx.helper.tensor_dtype_to_np_dtype(value.dtype())

        return value.dtype

    @staticmethod
    def get_shape(value: Any) -> Tuple[int, ...]:
        cls_name = value.__class__.__name__

        if cls_name == "OrtValue":
            return tuple(value.shape())

        if cls_name == "Tensor":
            return tuple(value.shape)

        return value.shape

    def merge_args(self, args: Tuple[Any] = (), kwargs: Optional[Dict[str, Any]] = None, check: bool = True) -> Tuple[Dict[str, Any], bool]:
        if args is None:
            args = ()

        if kwargs is None:
            kwargs = {}

        try:
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = bound_args.arguments
        except TypeError as e:
            formatted_msg = f"model '{self.model_name}' arguments error: {str(e)}"
            raise TypeError(formatted_msg) from None

        has_ortvalue: bool = False
        for name, value in kwargs.items():
            if isinstance(value, Future):
                continue

            cls_name = value.__class__.__name__
            if check and cls_name not in ("ndarray", "OrtValue", "Tensor"):
                raise TypeError(f"model '{self.model_name}' input node '{name}' need type (numpy.ndarray, onnxruntime.OrtValue, torch.Tensor), {type(value)} were given")

            if cls_name == "OrtValue":
                has_ortvalue: bool = True
                if not check:
                    break

            if check:
                node = self.inputs[name]

                expected_type = node.dtype
                actual_type = self.get_dtype(value)
                if actual_type != expected_type:
                    raise TypeError(f"model '{self.model_name}' input node '{name}' need dtype {expected_type}, {actual_type} were given")

                expected_shape = node.shape
                actual_shape = self.get_shape(value)
                if len(expected_shape) != len(actual_shape):
                    raise ValueError(f"model '{self.model_name}' input node '{name}' need shape {expected_shape}, {actual_shape} were given")

                for i, dim in enumerate(expected_shape):
                    if isinstance(dim, str) or dim is None or dim == -1:
                        continue

                    if actual_shape[i] != dim:
                        raise ValueError(f"model '{self.model_name}' input node '{name}' need shape {expected_shape}, {actual_shape} were given")

        return kwargs, has_ortvalue

    def check_outputs(self, output_names):
        if output_names is None:
            return

        for name in output_names:
            if name not in self.outputs:
                raise ValueError(f"model '{self.model_name}' has no output named '{name}'. valid outputs: {self.outputs.keys()}")
