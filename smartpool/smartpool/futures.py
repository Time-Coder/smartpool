from __future__ import annotations

import inspect
import traceback
import itertools
from concurrent.futures import Future as _BaseFuture
from concurrent.futures._base import PENDING, RUNNING, CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED
from typing import Any, List, Set, Callable, Dict, Union


class Future(_BaseFuture):
    
    def __init__(self, check_args: bool = True):
        _BaseFuture.__init__(self)
        self._sub_futures: Dict[Union[int, str], _BaseFuture] = {}
        self._attached_futures: Set[_BaseFuture] = set()
        self._start_callbacks: List[Callable[[], None]] = []
        self._check_args: bool = check_args

    def _invoke_callbacks(self):
        for callback in self._done_callbacks:
            try:
                callback(self)
            except Exception as e:
                traceback.print_exc()

    def _invoke_start_callbackes(self):
        for callback in self._start_callbacks:
            try:
                callback()
            except Exception as e:
                traceback.print_exc()

    def add_start_callback(self, fn: Callable[[], None]):
        if self._check_args:
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
        if self._check_args:
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

    def add_future(self, future: _BaseFuture)->None:
        with self._condition:
            future._state = self._state
            self._attached_futures.add(future)

    def __getitem__(self, key: Any) -> Future:
        with self._condition:
            if key not in self._sub_futures:
                self._sub_futures[key] = Future(self._check_args)
                self._sub_futures[key]._state = self._state

            return self._sub_futures[key]
    
    def unpack(self, n: int) -> List[Future]:
        with self._condition:
            children = []
            for i in range(n):
                if i not in self._sub_futures:
                    self._sub_futures[i] = Future(self._check_args)
                    self._sub_futures[i]._state = self._state

                children.append(self._sub_futures[i])

            return children

    def set_running_or_notify_cancel(self)->bool:
        with self._condition:
            if self._state != PENDING:
                return self._state == RUNNING

        result: bool = _BaseFuture.set_running_or_notify_cancel(self)

        if result:
            self._invoke_start_callbackes()

        with self._condition:
            for sub_future in itertools.chain(self._sub_futures.values(), self._attached_futures):
                sub_future.set_running_or_notify_cancel()

        return result

    def cancel(self)->bool:
        result: bool = _BaseFuture.cancel(self)
        with self._condition:
            for sub_future in itertools.chain(self._sub_futures.values(), self._attached_futures):
                sub_future.cancel()

        return result

    def set_result(self, result: Any)->None:
        _BaseFuture.set_result(self, result)

        with self._condition:
            for key, sub_future in self._sub_futures.items():
                try:
                    sub_future.set_result(result[key])
                except Exception as e:
                    sub_future.set_exception(e)

    def set_exception(self, exception: BaseException)->None:
        _BaseFuture.set_exception(self, exception)

        with self._condition:
            for sub_future in itertools.chain(self._sub_futures.values(), self._attached_futures):
                sub_future.set_exception(exception)
    