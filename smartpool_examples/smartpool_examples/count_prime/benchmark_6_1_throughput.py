"""Section 6.1 benchmark: raw submission throughput of count_prime tasks.

Compares smartpool.ProcessPool against concurrent.futures.ProcessPoolExecutor
by sweeping max_workers from 1 to the machine's core count.  Measures the
end-to-end wall-clock time of submitting a fixed batch of N near-identical,
CPU-bound count_prime tasks and collecting all results.

Output: a JSON file (raw per-repetition times) consumed by
`make_6_1_figures.py` to produce the paper figures.

Run:  python benchmark_6_1_throughput.py [--out path/to/results.json]
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time

from concurrent.futures import ProcessPoolExecutor as CFFutureProcessPool

_here = os.path.dirname(os.path.abspath(__file__))
_sp_pkg_dir = os.path.abspath(os.path.join(_here, "..", "..", "..", "smartpool"))
if os.path.isdir(os.path.join(_sp_pkg_dir, "smartpool")):
    sys.path.insert(0, _sp_pkg_dir)


def count_prime(start: int, stop: int) -> int:
    """Number of primes in the half-open interval [start, stop)."""
    import math

    count = 0
    for num in range(start, stop):
        if num < 2:
            continue
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                break
        else:
            count += 1
    return count


BASE = 1_000_000          # start of the searched number band
STEP = 500                # length of each task's disjoint interval
N_TASKS = int(os.environ.get("SP_BENCH_N_TASKS", "3000"))  # batch size ("thousands of tasks")


def build_tasks(n_tasks: int = N_TASKS, step: int = STEP, base: int = BASE):
    return [(base + i * step, base + (i + 1) * step) for i in range(n_tasks)]


def run_batch(pool_cls, tasks, max_workers: int):
    """Submit `tasks` on a fresh pool and return the wall-clock seconds."""
    args_param = inspect.signature(pool_cls.submit).parameters.get("args")
    smartpool_style = args_param is not None and args_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

    def do_submit(pool, t):
        if smartpool_style:
            return pool.submit(count_prime, args=t)
        return pool.submit(count_prime, *t)

    t0 = time.perf_counter()
    with pool_cls(max_workers=max_workers) as pool:
        futures = [do_submit(pool, t) for t in tasks]
        total = sum(f.result() for f in futures)
    elapsed = time.perf_counter() - t0
    return elapsed, total


def run_sequential(tasks):
    t0 = time.perf_counter()
    total = sum(count_prime(*t) for t in tasks)
    elapsed = time.perf_counter() - t0
    return elapsed, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per config")
    parser.add_argument("--workers", type=int, default=None, help="Max worker count to sweep up to")
    args = parser.parse_args()

    from smartpool import ProcessPool as SPProcessPool

    workers_max = args.workers or os.cpu_count() or 8
    workers = list(range(1, workers_max + 1))
    tasks = build_tasks()

    expected = None
    try:
        from sympy import primepi
        expected = int(primepi(BASE + N_TASKS * STEP) - primepi(BASE))
    except Exception:
        pass

    print(f"machine: {platform.platform()}")
    print(f"cpu logical: {os.cpu_count()}")
    print(f"tasks: {N_TASKS}, each counting primes in a {STEP}-number interval in [{BASE}, {BASE + N_TASKS * STEP})")
    print(f"expected prime count in band: {expected}")
    print(f"sweeping workers 1..{workers_max}, {args.reps} reps per config")

    # warm up both libraries and the OS
    print("warming up ...", flush=True)
    run_batch(SPProcessPool, tasks[:256], 4)
    run_batch(CFFutureProcessPool, tasks[:256], 4)
    run_sequential(tasks[:256])

    results = {
        "benchmark": "count_prime throughput sweep (Section 6.1)",
        "machine": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "n_tasks": N_TASKS,
        "task_band": [BASE, BASE + N_TASKS * STEP],
        "step": STEP,
        "expected_total_primes": expected,
        "reps": args.reps,
        "workers": workers,
        "sequential_sec": [],
        "pools": {},
    }

    out_path = args.out
    if out_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(here, "..", "..", "..", "doc", "experiments",
                                "6_1_count_prime_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def dump():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("sequential baseline ...", flush=True)
    for _ in range(args.reps):
        elapsed, total = run_sequential(tasks)
        assert expected is None or total == expected, (total, expected)
        results["sequential_sec"].append(elapsed)
        print(f"  seq: {elapsed:.3f} s  (total={total})", flush=True)
    dump()

    for name, cls in [("smartpool.ProcessPool", SPProcessPool),
                      ("concurrent.futures.ProcessPoolExecutor", CFFutureProcessPool)]:
        print(f"pool: {name}", flush=True)
        results["pools"][name] = {"times": {}}
        for w in workers:
            runs = []
            for rep in range(args.reps):
                elapsed, total = run_batch(cls, tasks, w)
                assert expected is None or total == expected, (total, expected)
                runs.append(elapsed)
                print(f"  workers={w:>2} rep={rep}: {elapsed:.3f} s  (total={total})", flush=True)
            results["pools"][name]["times"][str(w)] = runs
            dump()
            time.sleep(0.3)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()