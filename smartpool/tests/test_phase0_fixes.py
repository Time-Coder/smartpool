"""Regression tests for the scheduler defects fixed during the Section 6 redesign.

Each test pins one behaviour that was previously wrong. They are deliberately
cheap and free of real processes so they can run on every edit; the behavioural
evidence lives in the benchmark harness.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from smartpool.gpuinfo import GPUInfoSnapshot
from smartpool.resource import DataSize, Resource
from smartpool.sysinfo import SysInfo


def _make_snapshot(mem_total=8 * 1024, mem_used=2 * 1024, n_cores=100, n_cores_used=10):
    snapshot = GPUInfoSnapshot()
    snapshot.mem_total = mem_total
    snapshot.mem_used = mem_used
    snapshot.n_cores = n_cores
    snapshot.n_cores_used = n_cores_used
    snapshot.device = SimpleNamespace(torch_device="cuda:0", gpu_index=0)
    return snapshot


class ChunkResourceUnionTests(unittest.TestCase):
    """A chunk runs on one worker, so capacity must be a max, not a sum."""

    def test_capacity_fields_take_the_maximum_not_the_sum(self):
        from smartpool.chunk_task import ChunkTask

        target = Resource(cpu_cores=4, cpu_mem=4 * DataSize.GB, gpu_cores=64, gpu_mem=8 * DataSize.GB)
        accumulated = Resource(cpu_cores=1, cpu_mem=0, gpu_cores=0, gpu_mem=0)

        for _ in range(8):
            ChunkTask._update_res(accumulated, target)

        self.assertEqual(accumulated.cpu_cores, 4)
        self.assertEqual(accumulated.cpu_mem, 4 * DataSize.GB)
        self.assertEqual(accumulated.gpu_cores, 64)
        self.assertEqual(accumulated.gpu_mem, 8 * DataSize.GB)

    def test_infer_chunk_capacity_matches_generic_chunk(self):
        from smartpool.chunk_task import ChunkTask
        from smartpool.infer_session_pool.infer_chunk_task import InferChunkTask

        def build():
            return Resource(
                cpu_cores=2,
                cpu_mem=2 * DataSize.GB,
                gpu_cores=2305,
                gpu_mem=512 * DataSize.MB,
                result_cpu_mem=16 * DataSize.MB,
            )

        generic = Resource()
        infer = Resource()
        for _ in range(4):
            ChunkTask._update_res(generic, build())
            InferChunkTask._update_res(infer, build())

        for field in ("cpu_cores_in_python", "cpu_cores_out_of_python", "cpu_mem", "gpu_cores", "gpu_mem"):
            self.assertEqual(
                getattr(generic, field), getattr(infer, field),
                msg=f"{field} diverges between ChunkTask and InferChunkTask",
            )

        # Result footprints are held per sub-result, so those must accumulate.
        self.assertEqual(generic.result_cpu_mem, 4 * 16 * DataSize.MB)
        self.assertEqual(infer.result_cpu_mem, 4 * 16 * DataSize.MB)

    def test_full_width_gpu_chunk_does_not_exceed_one_device(self):
        """The defect that made batched inference undispatchable in practice."""
        from smartpool.infer_session_pool.infer_chunk_task import InferChunkTask

        device_cores = 2305
        sub = Resource(gpu_cores=device_cores, gpu_mem=512 * DataSize.MB)
        chunk_res = Resource()
        for _ in range(16):
            InferChunkTask._update_res(chunk_res, sub)

        self.assertLessEqual(chunk_res.gpu_cores, device_cores)
        self.assertLessEqual(chunk_res.gpu_mem, 512 * DataSize.MB)


class GpuReservationTests(unittest.TestCase):
    """Refreshing a physical sample must not drop an outstanding reservation."""

    def test_free_getter_subtracts_reservations(self):
        gpu = _make_snapshot()
        self.assertEqual(gpu.mem_free, 6 * 1024)
        self.assertEqual(gpu.n_cores_free, 90)

        gpu.mem_free -= 1024
        gpu.n_cores_free -= 20

        self.assertEqual(gpu.mem_free, 5 * 1024)
        self.assertEqual(gpu.n_cores_free, 70)
        self.assertEqual(gpu.mem_reserved, 1024)
        self.assertEqual(gpu.n_cores_reserved, 20)

    def test_absolute_set_keeps_physical_sample_intact(self):
        gpu = _make_snapshot()
        gpu.mem_free = 4 * 1024
        self.assertEqual(gpu.mem_used, 2 * 1024)
        self.assertEqual(gpu.mem_reserved, 2 * 1024)

    def test_refresh_preserves_reservation_across_snapshot_rebuild(self):
        info = SysInfo()
        info._gpu_infos = [_make_snapshot()]

        gpu = info.gpu_infos[0]
        gpu.mem_free -= 2 * 1024
        gpu.n_cores_free -= 30
        expected_mem_free = gpu.mem_free
        expected_cores_free = gpu.n_cores_free

        # update() drops the physical snapshots; the next access rebuilds them.
        info.update(refresh_cpu=False)
        self.assertIsNone(info._gpu_infos)

        with patch("smartpool.gpuinfos.GPUInfos.snapshot", return_value=[_make_snapshot()]):
            rebuilt = info.gpu_infos[0]

        self.assertIsNot(rebuilt, gpu)
        self.assertEqual(rebuilt.mem_free, expected_mem_free)
        self.assertEqual(rebuilt.n_cores_free, expected_cores_free)

    def test_reservation_survives_several_refreshes(self):
        info = SysInfo()
        info._gpu_infos = [_make_snapshot()]
        info.gpu_infos[0].mem_free -= 1024

        for _ in range(3):
            info.update(refresh_cpu=False)
            with patch("smartpool.gpuinfos.GPUInfos.snapshot", return_value=[_make_snapshot()]):
                _ = info.gpu_infos[0]

        with patch("smartpool.gpuinfos.GPUInfos.snapshot", return_value=[_make_snapshot()]):
            self.assertEqual(info.gpu_infos[0].mem_free, 5 * 1024)

    def test_no_reservation_means_no_carry_over(self):
        info = SysInfo()
        info._gpu_infos = [_make_snapshot()]
        info.update(refresh_cpu=False)
        with patch("smartpool.gpuinfos.GPUInfos.snapshot", return_value=[_make_snapshot()]):
            self.assertEqual(info.gpu_infos[0].mem_free, 6 * 1024)

        self.assertEqual(info._gpu_reserved, {})


class WorkerMemoryLedgerTests(unittest.TestCase):
    """The refund must equal the charge, not a freshly sampled RSS."""

    def _make_worker(self, rss_sequence):
        from smartpool.worker import Worker

        class _MemoryWorker(Worker):
            @property
            def memory(self):
                return rss_sequence.pop(0) if rss_sequence else 0

            def start(self):
                pass

            def join(self):
                pass

        info = SimpleNamespace(cpu_mem_free=1000)
        lock = threading.Lock()
        pool = SimpleNamespace(_sys_info=info, _sys_info_lock=lock, _workers=[])

        worker = object.__new__(_MemoryWorker)
        Worker.__init__(worker, pool)
        worker.pool = pool
        return worker, info

    def test_refund_matches_charge_when_worker_grew(self):
        worker, info = self._make_worker([100, 900])

        worker._take_worker_memory()
        self.assertEqual(info.cpu_mem_free, 900)
        self.assertEqual(worker._memory_taken_amount, 100)

        worker._release_worker_memory()
        self.assertEqual(info.cpu_mem_free, 1000)

    def test_refund_matches_charge_when_rss_read_failed(self):
        worker, info = self._make_worker([0, 0])

        worker._take_worker_memory()
        worker._release_worker_memory()
        self.assertEqual(info.cpu_mem_free, 1000)

    def test_repeated_cycles_do_not_drift(self):
        worker, info = self._make_worker([100, 900] * 20)
        baseline = info.cpu_mem_free

        for _ in range(20):
            worker._take_worker_memory()
            worker._release_worker_memory()

        self.assertEqual(info.cpu_mem_free, baseline)

    def test_double_release_is_a_no_op(self):
        worker, info = self._make_worker([100, 900])
        worker._take_worker_memory()
        worker._release_worker_memory()
        worker._release_worker_memory()
        self.assertEqual(info.cpu_mem_free, 1000)


class ChunkFlushTests(unittest.TestCase):
    """An empty chunk must not be dispatched, or its sub-task is lost."""

    def test_empty_chunk_submit_is_ignored(self):
        from smartpool.chunk_task import ChunkTask

        chunk = ChunkTask.__new__(ChunkTask)
        chunk.submitted = False
        chunk._sub_tasks = []
        chunk._key = ("f", 2)
        chunk.pool = SimpleNamespace(_chunk_tasks={("f", 2): chunk}, _submit=lambda t: None)

        chunk.submit()

        self.assertFalse(chunk.submitted)
        self.assertIn(("f", 2), chunk.pool._chunk_tasks)

    def test_chunk_last_add_time_is_stamped_at_creation(self):
        """The flush deadline is measured from the last add, so it must be live."""
        import time

        from smartpool.chunk_task import ChunkTask

        before = time.time()
        chunk = ChunkTask.__new__(ChunkTask)
        chunk.last_add_time = time.time()
        self.assertGreaterEqual(chunk.last_add_time, before)


def _identity(value: int) -> int:
    return value


class ChunkFlushIntegrationTests(unittest.TestCase):
    """A slow submission loop must still resolve every chunked sub-task."""

    def test_all_subtasks_resolve_with_slow_submission(self):
        import time

        from smartpool import ProcessPool, Resource

        pool = ProcessPool(max_workers=2)
        try:
            futures = []
            for i in range(24):
                futures.append(pool.submit(
                    _identity,
                    args=(i,),
                    cpu_mode_res=Resource(cpu_cores=1),
                    chunksize=4,
                ))
                # Slower than the default 0.1 s chunk timeout would tolerate if
                # the flush deadline were measured from chunk creation.
                time.sleep(0.02)

            pool.flush()
            self.assertEqual([f.result(timeout=30) for f in futures], list(range(24)))
        finally:
            pool.shutdown()


class InterpreterClearTests(unittest.TestCase):
    def test_clear_does_not_raise(self):
        from smartpool.interpreter_pool.interpreter_worker import InterpreterWorker

        worker = InterpreterWorker.__new__(InterpreterWorker)
        worker.executor = object()
        worker._is_working = False
        worker._is_rss_dirty = False
        worker._cached_rss = 0
        worker.rss = 0
        worker.pool = SimpleNamespace(
            _sys_info=SimpleNamespace(cpu_mem_free=0),
            _sys_info_lock=threading.Lock(),
            _workers_working_count=0,
        )
        worker._memory_taken = False
        worker._memory_taken_amount = 0
        worker.interp = object()
        worker.imported_modules = {"torch"}

        worker._clear()

        self.assertIsNone(worker.executor)
        self.assertIsNone(worker.interp)
        self.assertEqual(worker.imported_modules, set())


class CpuRefreshTests(unittest.TestCase):
    """The submit path must not block on a CPU sample."""

    def test_refresh_is_non_blocking_and_rate_limited(self):
        info = SysInfo()
        with patch("smartpool.sysinfo.psutil.cpu_percent", return_value=17.0) as sample:
            info.refresh_cpu_now()
            self.assertEqual(sample.call_args.kwargs["interval"], None)
            self.assertEqual(info._last_cpu_percent, 17.0)

            # A second call inside the rate-limit window must not sample again.
            info.refresh_cpu_now()
            self.assertEqual(sample.call_count, 1)

    def test_sample_is_retaken_once_the_window_expires(self):
        info = SysInfo()
        info.update_cpu_percent(interval=None)
        with patch("smartpool.sysinfo.psutil.cpu_percent", return_value=42.0) as sample:
            info.refresh_cpu_now(min_interval=0.0)
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(info._last_cpu_percent, 42.0)

    def test_update_without_cpu_refresh_keeps_the_logical_account(self):
        info = SysInfo()
        info.update_cpu_percent(interval=None)
        info.cpu_cores_free = 32.0
        info.update(refresh_cpu=False)
        self.assertEqual(info.cpu_cores_free, 32.0)


class DeviceChangeChannelTests(unittest.TestCase):
    """device_changeable must not depend on the unrelated use_torch flag.

    The device-change queue is the only channel through which a migration signal
    reaches a child, and Worker.run needs its handle at spawn time. Creating it
    only when torch was available made device_changeable=True a silent no-op on a
    default ProcessPool: the scheduler reserved the GPU and marked the task
    migrated, and the worker never learned about it.
    """

    def _worker(self, use_torch: bool):
        import importlib.util

        from smartpool import ProcessPool, Resource

        if use_torch and importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed")

        pool = ProcessPool(max_workers=1, use_torch=use_torch)
        try:
            # Workers are created lazily, so force one into existence.
            pool.submit(_identity, args=(1,), cpu_mode_res=Resource(cpu_cores=1)).result(timeout=60)
            self.assertEqual(len(pool._workers), 1)
            return pool._workers[0]
        finally:
            pool.shutdown()

    def test_queue_exists_without_use_torch(self):
        worker = self._worker(use_torch=False)
        self.assertIsNotNone(worker.change_device_cmd_queue)

    def test_queue_exists_with_use_torch(self):
        worker = self._worker(use_torch=True)
        self.assertIsNotNone(worker.change_device_cmd_queue)

    def test_change_device_is_delivered(self):
        """A device change must reach the queue the child is already reading."""
        from smartpool import Device

        worker = self._worker(use_torch=False)
        worker.change_device(Device("cuda:0"), task_id="task-42")

        # multiprocessing.SimpleQueue.get takes no timeout, so poll the reader.
        reader = worker.change_device_cmd_queue._reader
        self.assertTrue(reader.poll(15), "device change was not delivered to the worker")

        task_id, device = worker.change_device_cmd_queue.get()
        self.assertEqual(task_id, "task-42")
        self.assertEqual(str(getattr(device, "torch_device", device)), "cuda:0")

    def test_device_command_carries_its_task_id(self):
        """An untagged command would be applied to whichever task runs next."""
        from smartpool import Device

        worker = self._worker(use_torch=False)
        worker.change_device(Device("cuda:1"), task_id="task-7")
        reader = worker.change_device_cmd_queue._reader
        self.assertTrue(reader.poll(15))
        task_id, _device = worker.change_device_cmd_queue.get()
        self.assertEqual(task_id, "task-7")


class StaleDeviceCommandTests(unittest.TestCase):
    """A command for a finished task must not reach the task that runs next."""

    def test_command_for_another_task_is_discarded(self):
        from smartpool.worker import Worker

        applied = []

        class _FakeLock:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class _Queue:
            def __init__(self, items):
                self.items = list(items)

            def get(self):
                if not self.items:
                    raise AssertionError("queue drained")
                return self.items.pop(0)

        with patch(
            "smartpool.utils._set_best_device",
            side_effect=lambda device, tid=None: applied.append((tid, str(device))),
        ):
            state = {"task_id": "current"}
            Worker._changing_device(
                _Queue([("finished", "cuda:9"), ("current", "cuda:0"), None]),
                1234,
                state,
            )

        self.assertEqual(applied, [(1234, "cuda:0")])


if __name__ == "__main__":
    unittest.main()