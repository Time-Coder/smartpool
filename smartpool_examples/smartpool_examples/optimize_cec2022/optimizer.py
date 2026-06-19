import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from mealpy import DE, PSO, FloatVar
from mealpy.utils.problem import Problem

from config import DIM, MAX_ITER, POP_SIZE
from functions import FUNC_CLASSES, FUNC_NAMES


@dataclass
class OptimizationResult:
    algo_name: str
    func_name: str
    run_idx: int
    best_fitness: float
    convergence: List[float] = field(default_factory=list)


@dataclass
class ProgressInfo:
    algo_name: str
    func_name: str
    run_idx: int
    iteration: int
    total_iterations: int
    best_fitness: float


@dataclass
class ErrorInfo:
    exception: BaseException
    traceback: str


class FitnessProblem(Problem):
    def __init__(self, bounds, minmax="min", fit_func=None, **kwargs):
        super().__init__(bounds, minmax, **kwargs)
        self._fit_func = fit_func

    def obj_func(self, x):
        return self._fit_func(x)


class _ObservableDE(DE.OriginalDE):
    def __init__(self, progress_callback: Optional[Callable] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_callback = progress_callback

    def track_optimize_step(self, population, epoch, runtime):
        super().track_optimize_step(population, epoch, runtime)
        if self._progress_callback:
            self._progress_callback(epoch, self.g_best.target.fitness)


class _ObservablePSO(PSO.OriginalPSO):
    def __init__(self, progress_callback: Optional[Callable] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_callback = progress_callback

    def track_optimize_step(self, population, epoch, runtime):
        super().track_optimize_step(population, epoch, runtime)
        if self._progress_callback:
            self._progress_callback(epoch, self.g_best.target.fitness)


def optimize_single(algo_name, func_idx, run_idx, progress_queue, device=None):
    try:
        return _optimize_single(algo_name, func_idx, run_idx, progress_queue)
    except Exception as e:
        error_info = ErrorInfo(e, traceback.format_exc())
        progress_queue.put(error_info)
        raise e


def _optimize_single(algo_name, func_idx, run_idx, progress_queue):
    func_class = FUNC_CLASSES[func_idx]
    func_name = FUNC_NAMES[func_idx]
    opf = func_class(ndim=DIM)

    problem = FitnessProblem(
        bounds=FloatVar(lb=opf.lb, ub=opf.ub),
        minmax="min",
        fit_func=opf.evaluate,
        log_to="None",
    )

    def on_epoch(epoch, best_fitness):
        info = ProgressInfo(algo_name, func_name, run_idx, epoch, MAX_ITER, best_fitness)
        progress_queue.put(info)

    cls_map = {"DE": _ObservableDE, "PSO": _ObservablePSO}
    model = cls_map[algo_name](progress_callback=on_epoch, epoch=MAX_ITER, pop_size=POP_SIZE, log_to="None")

    initial_info = ProgressInfo(algo_name, func_name, run_idx, 0, MAX_ITER, float("inf"))
    progress_queue.put(initial_info)

    model.solve(problem)

    convergence = list(model.history.list_global_best_fit)
    best_fitness = model.g_best.target.fitness
    info = ProgressInfo(algo_name, func_name, run_idx, MAX_ITER, MAX_ITER, best_fitness)
    progress_queue.put(info)

    return OptimizationResult(algo_name, func_name, run_idx, best_fitness, convergence)
