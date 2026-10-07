import unittest
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from smartpool.pool import Pool


class IdleRetryTests(unittest.TestCase):
    def test_idle_pool_refreshes_cpu_sample_before_admission(self):
        class FakeSysInfo:
            cpu_mem_free = 1024

            def __init__(self):
                self.refreshed = False

            def refresh_cpu_now(self):
                self.refreshed = True

            def update(self, refresh_cpu=True):
                pass

            @property
            def cpu_cores_free(self):
                # A stale sample can admit one task but still throttle a batch.
                return 32 if self.refreshed else 6

        info = FakeSysInfo()
        pool = SimpleNamespace(
            _sys_info_lock=threading.Lock(),
            _sys_info=info,
            _estimate_cpu_cores_needed=lambda resource: 1,
        )
        task = SimpleNamespace(cpu_mode_res=SimpleNamespace(cpu_mem=0), device=None)
        with patch("smartpool.worker.Worker.total_working_count", return_value=0), \
             patch("smartpool.worker.Worker.recently_all_idle", return_value=False):
            devices, _, _ = Pool._choose_task_device(pool, task, "cpu")

        self.assertTrue(info.refreshed)
        self.assertEqual(len(devices), 1)

    def test_recently_finished_work_keeps_released_logical_cpu_capacity(self):
        class FakeSysInfo:
            cpu_mem_free = 1024
            cpu_cores_free = 32

            def __init__(self):
                self.samples = 0
                self.refresh_cpu = None

            def refresh_cpu_now(self):
                self.samples += 1

            def update(self, refresh_cpu=True):
                self.refresh_cpu = refresh_cpu
                if refresh_cpu:
                    self.cpu_cores_free = 6

        info = FakeSysInfo()
        pool = SimpleNamespace(
            _sys_info_lock=threading.Lock(),
            _sys_info=info,
            _estimate_cpu_cores_needed=lambda resource: 1,
        )
        task = SimpleNamespace(cpu_mode_res=SimpleNamespace(cpu_mem=0), device=None)
        with patch("smartpool.worker.Worker.total_working_count", return_value=0), \
             patch("smartpool.worker.Worker.recently_all_idle", return_value=True):
            devices, _, _ = Pool._choose_task_device(pool, task, "cpu")

        self.assertEqual(len(devices), 1)
        self.assertEqual(info.samples, 0)
        self.assertFalse(info.refresh_cpu)

    def test_delayed_task_retries_when_cached_cpu_capacity_is_zero(self):
        class FakePool:
            _postprocessing = False
            _can_move_to_gpu_tasks = {}
            _workers_working_count = 0
            _max_workers = 1
            _tasks = {}
            _sys_info = SimpleNamespace(cpu_cores_free=0)

            def __init__(self):
                self.task = type("Task", (), {"future": Future()})()
                self._delayed_tasks = {"task": self.task}
                self.assignments = 0
                self.dispatched = []

            def _try_assign_task(self, task):
                self.assignments += 1
                return True

            def _put_task(self, task):
                self.dispatched.append(task)

        pool = FakePool()
        Pool._postprocess_after_task_done(pool)
        self.assertEqual(pool.assignments, 1)
        self.assertEqual(pool.dispatched, [pool.task])
        self.assertFalse(pool._delayed_tasks)


if __name__ == "__main__":
    unittest.main()
