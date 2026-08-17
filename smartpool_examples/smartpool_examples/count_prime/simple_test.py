import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
sys.path.append(target_folder)

from smartpool import ProcessPool

def count_prime(start: int, stop: int) -> int:
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


BASE = 1_000
STEP = 500
N_TASKS = 3000
USE_SMARTPOOL = False


def build_tasks(n_tasks: int = N_TASKS, step: int = STEP, base: int = BASE):
    return [(base + i * step, base + (i + 1) * step) for i in range(n_tasks)]

if __name__ == "__main__":
    tasks = build_tasks()

    if USE_SMARTPOOL:
        pool = ProcessPool()
    else:
        pool = ProcessPoolExecutor()

    t1 = time.perf_counter()
    futures = []
    for task in tasks:
        if USE_SMARTPOOL:
            future = pool.submit(count_prime, args=task)
        else:
            future = pool.submit(count_prime, *task)

        futures.append(future)

    for future in futures:
        future.result()

    t2 = time.perf_counter()
    pool.shutdown()

    print("run time:", t2-t1)
