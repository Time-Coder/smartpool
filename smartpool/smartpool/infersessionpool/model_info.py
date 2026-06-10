from __future__ import annotations

import os
from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import onnxruntime as ort
    import inspect


class ModelInfo:
    model_path: str
    model_name: str
    file_size: int
    inputs: Dict[str, ort.NodeArg]
    signature: inspect.Signature

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = os.path.basename(self.model_path).split(".")[0]
        self.file_size = None
        self.inputs = None
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

    def fetch_info(self, session: ort.InferenceSession)->None:
        if self.inputs is not None:
            return
        
        model_name = session.get_modelmeta().name
        if model_name:
            self.model_name = model_name

        params = []
        self.inputs = {}
        inputs = session.get_inputs()
        for node in inputs:
            params.append(inspect.Parameter(node.name, inspect.Parameter.POSITIONAL_OR_KEYWORD))
            self.inputs[node.name] = node

        self.signature = inspect.Signature(params)
        
    def check_args(self, args: Tuple[Any] = (), kwargs: Optional[Dict[str, Any]] = None)->Dict[str, Any]:
        if kwargs is None:
            kwargs = {}

        try:
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = bound_args.arguments
        except TypeError as e:
            formatted_msg = f"onnx model '{self.model_name}' arguments error: {str(e)}"
            raise TypeError(formatted_msg) from None
        
        if self.inputs:
            import numpy as np
            for name, value in kwargs.items():
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"onnx model '{self.model_name}' input node '{name}' need type np.ndarray, {type(value)} were given")
                
                node = self.inputs[name]
                if node.shape != value.shape:
                    raise ValueError(f"onnx model '{self.model_name}' input node '{name}' need shape {node.shape}, {value.shape} were given")

                if node.type != value.dtype:
                    raise TypeError(f"onnx model '{self.model_name}' input node '{name}' need dtype {node.type}, {value.dtype} were given")

        return kwargs