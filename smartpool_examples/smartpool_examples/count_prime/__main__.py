import os
import sys

self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.append(self_folder)

if __name__ == "__main__":
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)

from smartpool import ProcessPool

from count_prime import count_prime


if __name__ == "__main__":
    print("Use ProcessPool to count prime numbers lower than 10000.")
    print(f"See source code at folder {self_folder}")
    
    tasks = []
    start = 0
    while start < 10000:
        stop = start + 1000
        tasks.append((start, stop))
        start = stop
        
    with ProcessPool() as pool:
        futures = []
        for task in tasks:
            future = pool.submit(count_prime, args=task)
            futures.append(future)
        
        total_primes_count = sum(future.result() for future in futures)
        print(total_primes_count)
    