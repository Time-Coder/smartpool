from .futures import Future
from concurrent.futures import Future as _BaseFuture
from typing import Callable
import inspect


class TaskWrapper:

    def __init__(self, cls, func, pool, submit_kwargs):
        self._cls = cls
        self._func = func
        self._pool = pool
        self._submit_kwargs = submit_kwargs
        self._start_callbacks = []
        self._done_callbacks = []

    def __call__(self, *args, **kwargs) -> Future:
        pool_instance = self._cls.instance(self._pool)
        future = pool_instance.submit(self._func, args=args, kwargs=kwargs, **self._submit_kwargs)
        for callback in self._start_callbacks:
            future.add_start_callback(callback)

        for callback in self._done_callbacks:
            future.add_done_callback(callback)
            
        return future

    def add_start_callback(self, fn: Callable[[], None])->None:
        signature: inspect.Signature = inspect.signature(fn)
        try:
            signature.bind()
        except TypeError as e:
            formatted_msg = f"{fn.__name__}() {str(e)}"
            raise TypeError(formatted_msg) from None
        
        self._start_callbacks.append(fn)

    def add_done_callback(self, fn: Callable[[_BaseFuture], None])->None:
        signature: inspect.Signature = inspect.signature(fn)
        try:
            signature.bind(1)
        except TypeError as e:
            formatted_msg = f"{fn.__name__}() {str(e)}"
            raise TypeError(formatted_msg) from None
        
        self._done_callbacks.append(fn)