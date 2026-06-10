from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..task import Task
    from .infersessionpool import InferSessionPool

from ..worker import Worker


class InferSessionWorker(Worker):
    pass
