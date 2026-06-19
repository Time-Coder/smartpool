from __future__ import annotations

import inspect
import traceback
from concurrent.futures import Future as _BaseFuture
from concurrent.futures._base import PENDING, CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED
from typing import Any, List, Callable


class Future(_BaseFuture):
    
    def __init__(self):
        _BaseFuture.__init__(self)
        self._start_callbacks: List[Callable[[], None]] = []

    def _invoke_callbacks(self):
        for callback in self._done_callbacks:
            try:
                callback(self)
            except Exception as e:
                traceback.print_stack()

    def _invoke_start_callbackes(self):
        for callback in self._start_callbacks:
            try:
                callback()
            except Exception as e:
                traceback.print_stack()

    def add_start_callback(self, fn: Callable[[], None]):
        signature: inspect.Signature = inspect.signature(fn)
        try:
            signature.bind()
        except TypeError as e:
            formatted_msg = f"{fn.__name__}() {str(e)}"
            raise TypeError(formatted_msg) from None
        
        with self._condition:
            if self._state == PENDING:
                self._start_callbacks.append(fn)
                return
        
        fn()

    def add_done_callback(self, fn: Callable[[_BaseFuture], None]):
        signature: inspect.Signature = inspect.signature(fn)
        try:
            signature.bind(self)
        except TypeError as e:
            formatted_msg = f"{fn.__name__}() {str(e)}"
            raise TypeError(formatted_msg) from None

        with self._condition:
            if self._state not in [CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED]:
                self._done_callbacks.append(fn)
                return
            
        fn(self)

    def set_running_or_notify_cancel(self)->bool:
        if _BaseFuture.set_running_or_notify_cancel(self):
            self._invoke_start_callbackes()

    def __getitem__(self, key: Any) -> Future:
        child: Future = Future()

        def callback(fut: _BaseFuture) -> None:
            try:
                child.set_result(fut.result()[key])
            except Exception as e:
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
            except Exception as e:
                for child in children:
                    child.set_exception(e)

        self.add_done_callback(callback)
        return children
