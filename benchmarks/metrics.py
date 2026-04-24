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
