"""Compare optimizers at an equal number of objective evaluations.

Iteration counts are not a fair basis: one iteration of fireworks or
bacterial foraging can cost many times the evaluations of one iteration of
grey wolf. :func:`budgeted_run` stops every method after the same number of
evaluations and reports the best value it had found by then.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .registry import FAMILIES, OPTIMIZERS, get_optimizer

Objective = Callable[[Sequence[float]], float]


class _BudgetSpent(Exception):
    pass


class BudgetedObjective:
    """Counts evaluations, keeps the best, and stops the run at the budget."""

    def __init__(self, objective: Objective, budget: int) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.objective = objective
        self.budget = int(budget)
        self.evaluations = 0
        self.best_value = math.inf
        self.best_position: List[float] = []
        self.__qualname__ = getattr(objective, "__qualname__", type(objective).__qualname__)
        self.__module__ = getattr(objective, "__module__", "?")

    def __call__(self, position: Sequence[float]) -> float:
        if self.evaluations >= self.budget:
            raise _BudgetSpent
        self.evaluations += 1
        value = float(self.objective(position))
        if value < self.best_value:
            self.best_value = value
            self.best_position = list(position)
        return value


def budgeted_run(
    name: str,
    objective: Objective,
    bounds: Sequence[Tuple[float, float]],
    budget: int,
    *,
    population_size: int = 30,
    seed: Optional[int] = None,
    **params,
) -> Tuple[List[float], float, int]:
    """Run optimizer ``name`` until ``budget`` evaluations are spent.

    Returns ``(best_position, best_value, evaluations_used)``.

    Many methods schedule their behaviour on progress through the run
    (``iteration / iterations``), so the iteration count matters. It is set
    from a probe: the evaluations one initialisation and one step cost. A run
    that still has budget left when its iterations end reports the
    evaluations it actually used; one that runs out stops at the budget.
    """
    from .contract import drive

    cls = get_optimizer(name)

    def build(func):
        return cls(objective=func, bounds=bounds, population_size=population_size,
                   seed=seed, **params)

    probe = BudgetedObjective(objective, 10**9)
    trial = build(probe)
    trial.iterations = 2
    trial.initialise()
    initial = probe.evaluations
    trial.iteration = 0
    trial.step()
    per_step = max(1, probe.evaluations - initial)
    iterations = max(1, (budget - initial) // per_step)

    counted = BudgetedObjective(objective, budget)
    try:
        drive(build(counted), iterations=iterations)
    except _BudgetSpent:
        pass
    return counted.best_position, counted.best_value, counted.evaluations


@dataclass
class Result:
    name: str
    family: str
    objective: str
    median: float
    best: float
    worst: float
    evaluations: int


def compare(
    objectives: Dict[str, Objective],
    bounds: Sequence[Tuple[float, float]],
    budget: int,
    *,
    names: Optional[Iterable[str]] = None,
    seeds: Sequence[int] = (0, 1, 2),
    population_size: int = 30,
) -> List[Result]:
    """Median, best and worst value per method and objective over ``seeds``."""
    results = []
    for name in sorted(names or OPTIMIZERS):
        for label, objective in objectives.items():
            values, used = [], 0
            for seed in seeds:
                _, value, evaluations = budgeted_run(
                    name, objective, bounds, budget,
                    population_size=population_size, seed=seed)
                values.append(value)
                used = max(used, evaluations)
            results.append(Result(name, FAMILIES[name], label, statistics.median(values),
                                  min(values), max(values), used))
    return results
