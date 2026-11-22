"""Communication graph for a swarm that cannot see every peer.

``topology='knn-oracle'`` is the historical behaviour: every particle reads
every other particle's state. The other topologies give each particle a
neighbour list and a message channel. A message carries a claimed best
position and cost. Loss and delay apply to that channel. A liar can put a
false claim on it. A spot-check defence re-evaluates the claimed point
instead of trusting the reported cost.
"""

from __future__ import annotations

import numpy as np

TOPOLOGIES = ("knn-oracle", "ring", "random-k", "geometric", "dynamic")
DEFENSES = ("trust", "spot-check")

# Lower than any cost the benchmark functions produce, so a trusted lie
# beats every honest claim.
LIE_COST = -1e300


class MessageChannel:
    def __init__(self, swarm):
        self.swarm = swarm
        self.neighbors: list[list[int]] = []
        self.liars: set[int] = set()
        self._colluding_pos = None
        self._pending: list[tuple] = []
        self.sent = 0
        self.delivered = 0
        self.lost = 0
        self.spot_checks = 0

    def build(self):
        n = self.swarm.n_particles
        for i, particle in enumerate(self.swarm.swarm):
            particle.index = i
        count = int(round(self.swarm.liar_fraction * n))
        if count:
            chosen = self.swarm.rng.choice(n, size=count, replace=False)
            self.liars = {int(i) for i in np.atleast_1d(chosen)}
            if self.swarm.liar_mode == "colluding":
                self._colluding_pos = self.swarm.rng.uniform(
                    self.swarm.val_min, self.swarm.val_max, self.swarm.dims
                )
        if self.swarm.topology != "dynamic":
            self.neighbors = self._neighbor_lists()

    def begin_epoch(self, epoch: int):
        """Deliver messages that are due, then set each local best from them."""
        if self.swarm.topology == "dynamic":
            self.neighbors = self._neighbor_lists()
        inbox: list[list[tuple]] = [[] for _ in self.swarm.swarm]
        still_waiting = []
        for arrive, receiver, pos, cost in self._pending:
            if arrive <= epoch:
                self.delivered += 1
                if self.swarm.defense == "spot-check":
                    cost = self._evaluate(pos)
                    self.spot_checks += 1
                inbox[receiver].append((pos, cost))
            else:
                still_waiting.append((arrive, receiver, pos, cost))
        self._pending = still_waiting
        for particle, claims in zip(self.swarm.swarm, inbox):
            best_pos = particle.local_best_pos
            best_cost = particle.local_best_cost
            if particle.best_cost < best_cost:
                best_pos = particle.best_pos
                best_cost = particle.best_cost
            for pos, cost in claims:
                if cost < best_cost:
                    best_pos = pos
                    best_cost = cost
            particle.local_best_pos = np.array(best_pos, copy=True)
            particle.local_best_cost = best_cost

    def send(self, epoch: int):
        """Queue one claim from each particle to each neighbour it can reach."""
        if self.swarm.topology == "dynamic":
            self.neighbors = self._neighbor_lists()
        arrive = epoch + 1 + int(self.swarm.message_delay)
        bandwidth = self.swarm.message_bandwidth
        for sender, particle in enumerate(self.swarm.swarm):
            targets = self.neighbors[sender]
            if bandwidth is not None:
                targets = targets[: int(bandwidth)]
            pos, cost = self._claim(particle)
            for receiver in targets:
                self.sent += 1
                if self.swarm.rng.random() < self.swarm.message_loss:
                    self.lost += 1
                    continue
                self._pending.append((arrive, receiver, pos.copy(), float(cost)))

    def best_known(self, particle):
        """One-item neighbour list shaped like ``get_best_neighbor``."""
        return [(particle.local_best_pos, particle.local_best_cost, 0.0)]

    def _claim(self, particle):
        if particle.index not in self.liars:
            return particle.best_pos.copy(), float(particle.best_cost)
        if self._colluding_pos is not None and self.swarm.liar_mode == "colluding":
            return self._colluding_pos.copy(), LIE_COST
        pos = self.swarm.rng.uniform(
            self.swarm.val_min, self.swarm.val_max, self.swarm.dims
        )
        return pos, LIE_COST

    def _evaluate(self, pos):
        if self.swarm.use_respect_boundary:
            return float(self.swarm.objective_with_respect_boundary(pos))
        return float(self.swarm.obj_func(pos))

    def _neighbor_lists(self) -> list[list[int]]:
        topology = self.swarm.topology
        n = self.swarm.n_particles
        degree = min(int(self.swarm.k), max(n - 1, 0))
        if topology == "ring":
            return [_ring(i, n, degree) for i in range(n)]
        if topology == "random-k":
            return [_random_k(self.swarm.rng, i, n, degree) for i in range(n)]
        if topology in ("geometric", "dynamic"):
            return _geometric_or_nearest(self.swarm, degree, topology == "dynamic")
        raise ValueError(f"unknown topology {topology!r}")


def _ring(i: int, n: int, degree: int) -> list[int]:
    if degree <= 0 or n <= 1:
        return []
    neighbors = []
    step = 1
    while len(neighbors) < degree:
        neighbors.append((i - step) % n)
        if len(neighbors) == degree:
            break
        neighbors.append((i + step) % n)
        step += 1
    return neighbors


def _random_k(rng, i: int, n: int, degree: int) -> list[int]:
    if degree <= 0 or n <= 1:
        return []
    others = [j for j in range(n) if j != i]
    picked = rng.choice(others, size=degree, replace=False)
    return [int(j) for j in np.atleast_1d(picked)]


def _geometric_or_nearest(swarm, degree: int, nearest_only: bool) -> list[list[int]]:
    positions = np.stack([particle.pos for particle in swarm.swarm])
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    n = distance.shape[0]
    lists = []
    if nearest_only:
        radius = np.inf
    else:
        radius = swarm.topology_radius
        if radius is None:
            span = float(swarm.val_max - swarm.val_min)
            radius = 0.25 * span * np.sqrt(swarm.dims)
    for i in range(n):
        within = np.where(distance[i] <= radius)[0]
        if within.size == 0:
            lists.append([])
            continue
        order = within[np.argsort(distance[i, within])]
        lists.append([int(j) for j in order[:degree]])
    return lists
