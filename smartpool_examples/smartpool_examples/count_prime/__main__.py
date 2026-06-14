import os
import sys

self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.append(self_folder)

if __name__ == "__main__":
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)

import typer
from count_prime import count_prime

from smartpool import InterpreterPool, ProcessPool, ThreadPool

app = typer.Typer()

POOL_MAP = {
    "ThreadPool": ThreadPool,
    "ProcessPool": ProcessPool,
    "InterpreterPool": InterpreterPool,
}


@app.command()
def main(pool: str = typer.Option("ProcessPool", "--pool", help="Pool type to use")):
    pool_cls = POOL_MAP.get(pool)
    if pool_cls is None:
        print(f"Unknown pool: {pool}. Choose from {list(POOL_MAP.keys())}")
        raise typer.Exit(code=1)

    print(f"Use {pool} to count prime numbers lower than 10000.")
    print(f"See source code at folder {self_folder}")

    tasks = []
    start = 0
    while start < 10000:
        stop = start + 1000
        tasks.append((start, stop))
        start = stop

    with pool_cls() as pool:
        futures = []
        for task in tasks:
            future = pool.submit(count_prime, args=task)
            futures.append(future)

        total_primes_count = sum(future.result() for future in futures)
        print(total_primes_count)


if __name__ == "__main__":
    app()
