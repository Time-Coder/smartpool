import unittest
from types import SimpleNamespace

from smartpool.pool import Pool


class WorkerAffinityTests(unittest.TestCase):
    def test_loaded_dependencies_win_over_unrelated_large_rss(self):
        class FakeWorker:
            is_working = False

            def __init__(self, modules, rss):
                self.imported_modules = modules
                self.cached_rss = rss

            def overlap_modules_ratio(self, task):
                return len(self.imported_modules & task.module_deps) / len(self.imported_modules)

        large_wrong_worker = FakeWorker({"shared", "torch"}, 1000)
        matching_worker = FakeWorker({"shared", "scipy"}, 50)
        pool = SimpleNamespace(
            _workers_working_count=0, _workers=[large_wrong_worker, matching_worker],
            _max_workers=2, _need_module_deps=True,
        )
        task = SimpleNamespace(module_deps={"shared", "scipy"},
                               modules_overlap_ratio=0.0, worker=None)
        chosen = Pool._choose_task_worker(pool, task, SimpleNamespace(cpu_mem=1))

        self.assertIs(chosen, matching_worker)
        self.assertIs(task.worker, matching_worker)


if __name__ == "__main__":
    unittest.main()
