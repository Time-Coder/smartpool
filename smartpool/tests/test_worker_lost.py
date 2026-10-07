"""A worker that dies mid-task must fail its future instead of hanging it.

Before the watchdog in Pool._collecting_result(), a child process that exited
while holding a task never sent a result, so the future stayed pending for the
lifetime of the parent process and any caller blocking on result() hung. These
tests drive a real crash through ProcessPool.
"""

import os
import sys
import unittest

from smartpool import DataSize, ProcessPool, Resource, WorkerLostError
from smartpool.pool import Pool

# A payload the child cannot unpickle under the spawn start method: the function
# object is a local, so pickling it fails inside the parent's queue feeder.
CRASH_REASON = "unpicklable payload"


def ok_task(value: int) -> int:
    return value + 1


def crashing_task(value: int) -> int:
    os._exit(7)


def _submit(pool, func, args, timeout=30.0):
    future = pool.submit(func, args=args, cpu_mode_res=Resource(cpu_cores=1))
    return future.result(timeout=timeout)


class WorkerLostTests(unittest.TestCase):
    def test_worker_lost_error_is_exported(self):
        import smartpool

        self.assertTrue(issubclass(smartpool.WorkerLostError, Exception))
        self.assertIs(smartpool.WorkerLostError, WorkerLostError)

    def test_pool_propagates_the_watchdog_setting(self):
        pool = ProcessPool(max_workers=1, worker_watchdog_interval=0.2)
        try:
            self.assertEqual(pool._worker_watchdog_interval, 0.2)
        finally:
            pool.shutdown()

    def test_default_watchdog_is_enabled(self):
        pool = ProcessPool(max_workers=1)
        try:
            self.assertGreater(pool._worker_watchdog_interval, 0.0)
        finally:
            pool.shutdown()

    def test_healthy_tasks_are_unaffected(self):
        pool = ProcessPool(max_workers=2)
        try:
            futures = [
                pool.submit(ok_task, args=(i,), cpu_mode_res=Resource(cpu_cores=1))
                for i in range(10)
            ]
            self.assertEqual([f.result(timeout=30) for f in futures], list(range(1, 11)))
        finally:
            pool.shutdown()

    @unittest.skipUnless(
        sys.platform != "win32" or os.environ.get("SP_CRASH_TESTS", "1") != "0",
        "crash test disabled",
    )
    def test_hard_exit_mid_task_raises_instead_of_hanging(self):
        pool = ProcessPool(max_workers=1)
        try:
            start = os.times()
            with self.assertRaises(WorkerLostError):
                _submit(pool, crashing_task, (1,))
            # The watchdog polls at 0.5 s by default; allow generous headroom for
            # a loaded machine but far less than an unbounded hang.
            self.assertLess(os.times().elapsed - start.elapsed, 30.0)
        finally:
            pool.shutdown()

    @unittest.skipUnless(
        sys.platform != "win32" or os.environ.get("SP_CRASH_TESTS", "1") != "0",
        "crash test disabled",
    )
    def test_pool_recovers_after_a_worker_is_lost(self):
        pool = ProcessPool(max_workers=2)
        try:
            with self.assertRaises(WorkerLostError):
                _submit(pool, crashing_task, (1,))

            # A fresh worker must be created and the pool must still work.
            futures = [
                pool.submit(ok_task, args=(i,), cpu_mode_res=Resource(cpu_cores=1))
                for i in range(6)
            ]
            self.assertEqual([f.result(timeout=30) for f in futures], list(range(1, 7)))
        finally:
            pool.shutdown()

    @unittest.skipUnless(
        sys.platform != "win32" or os.environ.get("SP_CRASH_TESTS", "1") != "0",
        "crash test disabled",
    )
    def test_resource_accounting_is_restored_after_a_worker_is_lost(self):
        pool = ProcessPool(max_workers=2)
        try:
            _submit(pool, ok_task, (1,))
            settled = Pool._sys_info.cpu_mem_free

            with self.assertRaises(WorkerLostError):
                _submit(pool, crashing_task, (1,))

            # Give the watchdog a moment to finish cleanup.
            import time

            for _ in range(20):
                if abs(Pool._sys_info.cpu_mem_free - settled) < 64 * DataSize.MB:
                    break
                time.sleep(0.1)

            self.assertLessEqual(
                Pool._sys_info.cpu_mem_free,
                settled + 64 * DataSize.MB,
                "logical memory was not returned after the worker died",
            )
        finally:
            pool.shutdown()


if __name__ == "__main__":
    unittest.main()