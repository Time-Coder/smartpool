from __future__ import annotations
from typing import TYPE_CHECKING, Dict
from collections import defaultdict


if TYPE_CHECKING:
    import onnxruntime as ort


class SessionInfo:

    def __init__(self, *args, **kwargs)->None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(*args, **kwargs)
        self._io_bindings: Dict[int, ort.IOBinding] = defaultdict(self.session.io_binding)

    @property
    def io_binding(self) -> ort.IOBinding:
        import threading
        return self._io_bindings[threading.current_thread().ident]
    