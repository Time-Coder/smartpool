from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..task import Task
    from .infersessionpool import InferSessionPool

from ..threadpool.threadworker import ThreadWorker


class InferSessionWorker(ThreadWorker):

    def _clear(self)->None:
        if self.process_or_thread is not None:
            with InferSessionPool._session_lock:
                if self.process_or_thread.ident in InferSessionPool._sessions:
                    del InferSessionPool._sessions[self.process_or_thread.ident]

        ThreadWorker._clear(self)

    def run(self):
        infersession_pool: InferSessionPool = self.pool

        if infersession_pool._initializer is not None:
            infersession_pool._initializer(*infersession_pool._initargs, **infersession_pool._initkwargs)

        while True:
            task:Task = self.task_queue.get()
            if task is None:
                break

            model_path = task.func
            args = task.args
            kwargs = task.kwargs
            provider = task.onnx_provider
            session = infersession_pool._get_session(model_path, provider)
            if infersession_pool.print_info:
                if "device_id" in provider[1]:
                    print(f"infer with {provider[0]}(device_id={provider[1]['device_id']}) in session {id(session)}")
                else:
                    print(f"infer with {provider[0]} in session {id(session)}")

            try:
                inputs = kwargs
                if args:
                    inputs = dict(zip((node.name for node in session.get_inputs()), args))
                    inputs.update(kwargs)

                result = session.run(None, input_feed=inputs)
                success = True
            except BaseException as e:
                result = e
                success = False

            infersession_pool._on_task_done(task.id, success, result)
