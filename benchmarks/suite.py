"""
Benchmark problem definitions for SwarmOpt.

Strata follow the spirit of Dewancker et al., "A Stratified Analysis of Bayesian
Optimization Methods" (arXiv:1603.09441): group test functions so that
performance can be summarized and ranked within comparable *genres* (here:
unimodal vs multimodal, plus a fine tag for oscillatory / ridged landscapes).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from swarmopt import functions

# Optional fine stratum for ranking within multimodal problems (paper-style "genres").
STRATUM_FINE: Dict[str, str] = {
    "sphere": "smooth",
    "rosenbrock": "valley",
    "ackley": "oscillatory",
    "griewank": "ridged_multimodal",
    "rastrigin": "oscillatory",
    "schwefel": "deceptive",
    "levy": "multimodal",
    "weierstrass": "oscillatory",
    "michalewicz": "oscillatory_boundary",
    "beale": "multimodal_2d",
    "booth": "multimodal_2d",
}


def _meta(name: str) -> Dict[str, Any]:
    m = functions.FUNCTION_METADATA.get(name)
    if m is None:
        raise KeyError(f"Unknown function in suite: {name}")
    return m


def _bounds(name: str) -> Tuple[float, float]:
    b = _meta(name)["bounds"]
    if not isinstance(b, (list, tuple)) or len(b) != 2:
        raise ValueError(f"Function {name} has non-scalar bounds; not in default suite.")
    return float(b[0]), float(b[1])


def problem(
    name: str,
    func: Callable[..., float],
    *,
    stratum_fine: Optional[str] = None,
    fixed_dims: Optional[int] = None,
) -> Dict[str, Any]:
    meta = _meta(name)
    st = meta.get("type", "unknown")
    sf = stratum_fine if stratum_fine is not None else STRATUM_FINE.get(name, st)
    return {
        "name": name,
        "func": func,
        "bounds": _bounds(name),
        "stratum": st,
        "stratum_fine": sf,
        "fixed_dims": fixed_dims,
    }


# ---------------------------------------------------------------------------
# Suites: (name, func) registered with metadata from functions.FUNCTION_METADATA
# ---------------------------------------------------------------------------

SUITE_DEFAULT: List[Dict[str, Any]] = [
    problem("sphere", functions.sphere),
    problem("rosenbrock", functions.rosenbrock),
    problem("ackley", functions.ackley),
    problem("griewank", functions.griewank),
    problem("rastrigin", functions.rastrigin),
    problem("schwefel", functions.schwefel),
    problem("levy", functions.levy),
    problem("weierstrass", functions.weierstrass),
    problem("michalewicz", functions.michalewicz, stratum_fine="oscillatory_boundary"),
]

# Additional 2D-only problems (fixed_dims=2 in runner). Branin is omitted here
# because it needs per-dimension bounds; Swarm uses a scalar box.
SUITE_2D: List[Dict[str, Any]] = [
    problem("beale", functions.beale, stratum_fine="multimodal_2d", fixed_dims=2),
    problem("booth", functions.booth, stratum_fine="multimodal_2d", fixed_dims=2),
]

SUITE_FULL: List[Dict[str, Any]] = SUITE_DEFAULT + [
    problem("sum_squares", functions.sum_squares, stratum_fine="unimodal_poly"),
    problem("rotated_hyper_ellipsoid", functions.rotated_hyper_ellipsoid, stratum_fine="ill_conditioned"),
    problem("zakharov", functions.zakharov, stratum_fine="plateau_ridge"),
] + SUITE_2D


def get_suite(name: str) -> List[Dict[str, Any]]:
    n = (name or "default").lower().strip()
    if n in ("default", "core"):
        return list(SUITE_DEFAULT)
    if n in ("2d", "2d_extra"):
        return list(SUITE_2D)
    if n in ("full", "all"):
        return list(SUITE_FULL)
    if n == "default+2d":
        return list(SUITE_DEFAULT) + list(SUITE_2D)
    raise ValueError("Unknown suite %r; try default, 2d, default+2d, or full" % (name,))


def optimal_value_for_regret(func_name: str) -> Optional[float]:
    """Return known global minimum when scalar; else None (skip regret)."""
    meta = functions.FUNCTION_METADATA.get(func_name)
    if not meta:
        return None
    v = meta.get("optimal_value")
    if isinstance(v, (int, float)):
        return float(v)
    return None
