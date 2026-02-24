import sys
import os
parent_folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(parent_folder)

import math
from smartprocesspool import SmartProcessPool


def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


def compute_task(n):
    count = 0
    for i in range(2, n):
        if is_prime(i):
            count += 1
    return count


if __name__ == "__main__":
    pool = SmartProcessPool(max_workers=4, use_torch=False)
    
    tasks = [1000, 2000, 3000, 4000, 5000]
    
    futures = []
    for task_param in tasks:
        future = pool.submit(compute_task, args=(task_param,))
        futures.append(future)
    
    results = [future.result() for future in futures]
    print(results)
    
    pool.shutdown()