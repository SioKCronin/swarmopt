"""
Summary statistics and simple-regret for benchmark trials.

Dewancker et al. (arXiv:1603.09441) emphasize reporting empirical behavior with
repeated trials; we record mean, spread, and (when the optimum f* is known)
simple regret: max(0, f_best - f*).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def simple_regret(best_cost: float, optimal: Optional[float]) -> Optional[float]:
    if optimal is None:
        return None
    return max(0.0, float(best_cost) - float(optimal))


def summarize_costs(values: Sequence[float]) -> Tuple[float, float, float]:
    """Mean, std (sample), median."""
    xs = [float(x) for x in values]
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(xs) / n
    if n == 1:
        return (mean, 0.0, xs[0])
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    sorted_x = sorted(xs)
    mid = n // 2
    if n % 2:
        med = sorted_x[mid]
    else:
        med = 0.5 * (sorted_x[mid - 1] + sorted_x[mid])
    return (mean, std, med)


def growth_exponent(sizes: Sequence[float], messages: Sequence[float]) -> float:
    """Power-law exponent of messages against swarm size.

    Fits ``log(messages) = a + b log(size)`` and returns ``b``. A value near
    1 is linear in N. A value near 2 is the all-pairs cost the oracle pays.
    """
    if len(sizes) != len(messages) or len(sizes) < 2:
        raise ValueError("need at least two (size, messages) pairs")
    log_n = [math.log(float(n)) for n in sizes]
    log_m = [math.log(float(m)) for m in messages]
    mean_n = sum(log_n) / len(log_n)
    mean_m = sum(log_m) / len(log_m)
    numer = sum((n - mean_n) * (m - mean_m) for n, m in zip(log_n, log_m))
    denom = sum((n - mean_n) ** 2 for n in log_n)
    if denom == 0.0:
        raise ValueError("sizes must not all be equal")
    return numer / denom


def messages_per_particle(sizes: Sequence[float], messages: Sequence[float]) -> List[float]:
    return [float(m) / float(n) for n, m in zip(sizes, messages)]


def mean_ranks_by_stratum(
    rows: List[dict],
    stratum_key: str = "stratum",
) -> List[Tuple[str, str, float]]:
    """
    For each stratum, rank algorithms per function (1 = best mean cost), then
    return average rank per (stratum, algo) across functions in that stratum.

    `rows` are dicts with keys: stratum, function, algo, avg_cost (one row per
    algo x function, or best-of-file duplicates resolved upstream).
    """
    from collections import defaultdict

    # stratum -> function -> algo -> min cost
    by_stratum: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        s = r.get(stratum_key) or "unknown"
        fn = r.get("function")
        al = r.get("algo")
        if not fn or not al:
            continue
        c = float(r["avg_cost"])
        d = by_stratum[s][fn]
        if al not in d or c < d[al]:
            d[al] = c

    out: List[Tuple[str, str, float]] = []
    for s, func_map in by_stratum.items():
        rank_sums: dict = defaultdict(float)
        n_funcs = 0
        for _fn, by_algo in func_map.items():
            if not by_algo:
                continue
            n_funcs += 1
            order = sorted(by_algo.items(), key=lambda t: t[1])
            for rank, (al, _c) in enumerate(order, start=1):
                rank_sums[al] += rank
        if n_funcs == 0:
            continue
        for al, total in rank_sums.items():
            out.append((s, al, total / n_funcs))
    out.sort(key=lambda t: (t[0], t[2], t[1]))
    return out
