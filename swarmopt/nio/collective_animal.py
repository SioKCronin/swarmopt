"""Collective Animal Behavior.

Reference:
Cuevas, E., González, M., Zaldivar, D., Pérez-Cisneros, M., & García, G. (2012).
An algorithm for global optimization inspired by collective animal behavior.
Discrete Dynamics in Nature and Society.
"""

from __future__ import annotations

import copy

from .base import PopulationOptimizer


class CollectiveAnimalBehavior(PopulationOptimizer):
    """Animals move toward a small memory of dominant neighbors, then the herd is replaced."""

    def step(self) -> None:
        # The memory holds copies of the dominant animals, so moving the herd
        # below doesn't move the memory, and the rebuilt herd has no duplicates.
        memory = [copy.deepcopy(agent) for agent in
                  self.sorted_population()[: max(2, self.population_size // 4)]]
        for agent in self.population:
            nearest = min(memory, key=lambda other: self.distance(agent.position, other.position))
            if self.rand() < 0.5:
                trial = self.mix(agent.position, nearest.position, self.rand())
            else:
                trial = self.mix(agent.position, self.best_position, self.rand())
            if self.rand() < 0.1:
                trial = self.perturb(trial, 0.1)
            position, fitness = self.place(trial)
            self.adopt(agent, position, fitness)
        pool = self.population + memory
        self.population = sorted(pool, key=lambda agent: agent.fitness)[: self.population_size]
