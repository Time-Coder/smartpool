from __future__ import annotations

from concurrent.futures import Future as _BaseFuture
from typing import Any, List


class Future(_BaseFuture):
    def __getitem__(self, key: Any) -> Future:
        child: Future = Future()

        def callback(fut: _BaseFuture) -> None:
            try:
                child.set_result(fut.result()[key])
            except BaseException as e:
                child.set_exception(e)

        self.add_done_callback(callback)
        return child

    def unpack(self, n: int) -> List[Future]:
        children: List[Future] = [Future() for _ in range(n)]

        def callback(fut: _BaseFuture) -> None:
            try:
                result = fut.result()
                for i, child in enumerate(children):
                    child.set_result(result[i])
            except BaseException as e:
                for child in children:
                    child.set_exception(e)

        self.add_done_callback(callback)
        return children
