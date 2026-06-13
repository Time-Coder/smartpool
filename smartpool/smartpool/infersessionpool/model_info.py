from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Optional, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
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
        self.model_path = model_path
        self.model_name = os.path.basename(self.model_path).split(".")[0]
        self.file_size = None
        self.input_nodes: Dict[str, NodeInfo] = {}
        self.output_names: Set[str] = set()
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

        model = onnx.load(model_path)
        onnx.checker.check_model(model, full_check=True)

        params = []
        for node in model.graph.input:
            params.append(inspect.Parameter(node.name, inspect.Parameter.POSITIONAL_OR_KEYWORD))
            self.input_nodes[node.name] = NodeInfo(node)

        for node in model.graph.output:
            self.output_names.add(node.name)

        self.signature = inspect.Signature(params)

    def check_args(self, args: Tuple[Any] = (), kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if kwargs is None:
            kwargs = {}

        try:
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = bound_args.arguments
        except TypeError as e:
            formatted_msg = f"model '{self.model_name}' arguments error: {str(e)}"
            raise TypeError(formatted_msg) from None

        if self.input_nodes:
            import numpy as np
            for name, value in kwargs.items():
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"model '{self.model_name}' input node '{name}' need type np.ndarray, {type(value)} were given")

                node = self.input_nodes[name]

                expected_type = node.dtype
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
            if name not in self.output_names:
                raise ValueError(f"model '{self.model_name}' has no output named '{name}'. valid outputs: {self.output_names}")
