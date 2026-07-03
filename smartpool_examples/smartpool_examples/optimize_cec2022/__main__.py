if __name__ == "__main__":
    import os
    import sys
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    target_folder = os.path.abspath(self_folder + "/../../../smartpool").replace("\\", "/")
    sys.path.append(target_folder)


from smartpool import limit_num_single_thread

limit_num_single_thread()

from typing import Literal, TypeAlias

import typer

app = typer.Typer(help="Use different parallel backends to optimize CEC2022 benchmark functions via opfunu + mealpy.")

PoolChoice: TypeAlias = Literal[
    "smartpool",
    "multiprocessing",
    "concurrent",
    "joblib",
    "ray",
    "sequentially"
]


@app.command()
def main(
    pool: PoolChoice = typer.Option(
        "smartpool",
        "--pool",
        help="choose parallel backend"
    ),
    max_workers: int = typer.Option(
        0,
        "--max_workers",
        help="max number of workers, 0 to use all available cores"
    )
):
    import os
    import sys
    import time

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    sys.path.append(self_folder)

    from config import ALGORITHMS, N_RUNS
    from functions import FUNC_CLASSES
    from progress_info import ProgressInfo
    from visualization import plot_convergence, print_results_table

    n_funcs = len(FUNC_CLASSES)
    n_algos = len(ALGORITHMS)
    total_tasks = n_funcs * n_algos * N_RUNS

    print(f"Use {pool} to optimize {n_funcs} CEC2022 functions with {n_algos} algorithms, {N_RUNS} runs each ({total_tasks} tasks).")
    print(f"See source code at folder {self_folder}")
    print()

    if max_workers == 0:
        max_workers = os.cpu_count()

    task_templates = []
    for func_idx in range(n_funcs):
        for algo_name in ALGORITHMS:
            for run_idx in range(N_RUNS):
                task_templates.append((algo_name, func_idx, run_idx))

    start_time = time.perf_counter()
    progress_info = ProgressInfo(len(task_templates))

    with progress_info:
        if pool == "smartpool":
            from optimize_with_smartpool import optimize_with_smartpool
            results = optimize_with_smartpool(task_templates, max_workers, progress_info)
        elif pool == "concurrent":
            from optimize_with_concurrent import optimize_with_concurrent
            results = optimize_with_concurrent(task_templates, max_workers, progress_info)
        elif pool == "multiprocessing":
            from optimize_with_multiprocessing import optimize_with_multiprocessing
            results = optimize_with_multiprocessing(task_templates, max_workers, progress_info)
        elif pool == "joblib":
            from optimize_with_joblib import optimize_with_joblib
            results = optimize_with_joblib(task_templates, max_workers, progress_info)
        elif pool == "ray":
            from optimize_with_ray import optimize_with_ray
            results = optimize_with_ray(task_templates, progress_info)
        elif pool == "sequentially":
            from optimize_sequentially import optimize_sequentially
            results = optimize_sequentially(task_templates, progress_info)

    elapsed = time.perf_counter() - start_time
    print(f"\noptimization completed in {elapsed:.2f} seconds")

    print_results_table(results)
    plot_convergence(results)


if __name__ == "__main__":
    app()
