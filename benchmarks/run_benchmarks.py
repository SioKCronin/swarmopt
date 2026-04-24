#!/usr/bin/env python3
"""
Stratified empirical benchmarks for SwarmOpt (inspired by Dewancker et al.,
arXiv:1603.09441: repeated trials, reporting spread, and ranking within function strata).

Run from the repository root:

    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --algos global hhoa --quick
"""

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

try:
    from swarmopt import Swarm  # noqa: E402
except ImportError as e:
    print(
        "Failed to import Swarm (benchmarks need the same dependencies as the library; "
        "try: pip install scipy). Original error: %s" % e,
        file=sys.stderr,
    )
    raise

import metrics  # noqa: E402
import suite  # noqa: E402


def _algorithm_registry():
    return {
        "global": ("global", {"algo": "global"}),
        "local": ("local", {"algo": "local"}),
        "hhoa": ("hhoa", {"algo": "hhoa"}),
        "cpso": ("cpso", {"algo": "cpso", "n_swarms": 3, "communication_strategy": "best"}),
        "unified": ("unified", {"algo": "unified", "u": 0.5}),
    }


def _apply_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _stable_trial_salt(algo: str, fname: str) -> int:
    """Deterministic 31-bit int (hash() is randomized per process in Python 3.3+)."""
    h = hashlib.md5(("{}\0{}".format(algo, fname)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31)


def _run_trial(
    problem: Dict[str, Any],
    algo_key: str,
    n_particles: int,
    dims: int,
    c1: float,
    c2: float,
    w: float,
    epochs: int,
    seed: int,
) -> Tuple[float, float]:
    _apply_seed(seed)
    display_name, extra = _algorithm_registry()[algo_key]
    d = int(problem["fixed_dims"] or dims)
    lo, hi = problem["bounds"]
    kwargs = {
        "n_particles": n_particles,
        "dims": d,
        "c1": c1,
        "c2": c2,
        "w": w,
        "epochs": epochs,
        "obj_func": problem["func"],
        "velocity_clamp": (lo, hi),
    }
    kwargs.update(extra)
    s = Swarm(**kwargs)
    s.optimize()
    return float(s.best_cost), float(s.runtime)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stratified SwarmOpt benchmark runner.")
    ap.add_argument(
        "--suite",
        default="default",
        help="Problem suite: default, 2d, default+2d, full (see benchmarks/suite.py).",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated function names to include (subset of the suite).",
    )
    ap.add_argument(
        "--algos",
        nargs="+",
        default=["global", "local", "hhoa"],
        help="Algorithm keys: %s" % (", ".join(sorted(_algorithm_registry().keys())),),
    )
    ap.add_argument("--dims", type=int, default=5)
    ap.add_argument("--n-particles", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--c1", type=float, default=2.0)
    ap.add_argument("--c2", type=float, default=2.0)
    ap.add_argument("--w", type=float, default=0.9)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="Base seed; trial k uses base + 1000 * k + algo hash.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for CSV (default: benchmarks/csvfiles).",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Shorter run: 2 trials, 3 epochs, first problem only, global+hhoa only.",
    )
    args = ap.parse_args(argv)

    if args.quick:
        args.n_trials = 2
        args.epochs = 3
        args.algos = ["global", "hhoa"]
        if args.only is None:
            args.only = "sphere"

    reg = _algorithm_registry()
    for a in args.algos:
        if a not in reg:
            print("Unknown algo %r; options: %s" % (a, ", ".join(reg)), file=sys.stderr)
            return 1

    try:
        problems = suite.get_suite(args.suite)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        problems = [p for p in problems if p["name"] in want]
        if not problems:
            print("No problems left after --only filter.", file=sys.stderr)
            return 1

    out_dir = (args.out_dir or (Path(__file__).resolve().parent / "csvfiles")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = strftime("%Y-%m-%d-%H-%M-%S")
    path = out_dir / ("benchmark_stratified_%s.csv" % stamp)

    base_seed = int(args.seed)
    n_trials = int(args.n_trials)
    rows_out: List[dict] = []

    fieldnames = [
        "algo",
        "function",
        "stratum",
        "stratum_fine",
        "n_trials",
        "seed_base",
        "dims",
        "n_particles",
        "epochs",
        "avg_cost",
        "std_cost",
        "median_cost",
        "avg_time",
        "std_time",
        "mean_regret",
    ]

    for problem in problems:
        fname = problem["name"]
        opt = suite.optimal_value_for_regret(fname)
        for algo in args.algos:
            display_name, _ = reg[algo]
            costs: List[float] = []
            times: List[float] = []
            regrets: List[float] = []
            for t in range(n_trials):
                seed_t = (base_seed + 1000 * t + _stable_trial_salt(algo, fname)) % (2**31)
                c, rtime = _run_trial(
                    problem,
                    algo,
                    args.n_particles,
                    args.dims,
                    args.c1,
                    args.c2,
                    args.w,
                    args.epochs,
                    seed_t,
                )
                costs.append(c)
                times.append(rtime)
                gr = metrics.simple_regret(c, opt)
                if gr is not None:
                    regrets.append(gr)

            mean_c, std_c, med_c = metrics.summarize_costs(costs)
            mean_t, std_t, _ = metrics.summarize_costs(times)
            mreg = None
            if opt is not None and regrets:
                mreg, _, _ = metrics.summarize_costs(regrets)
            row = {
                "algo": display_name,
                "function": fname,
                "stratum": problem["stratum"],
                "stratum_fine": problem["stratum_fine"],
                "n_trials": n_trials,
                "seed_base": base_seed,
                "dims": int(problem["fixed_dims"] or args.dims),
                "n_particles": int(args.n_particles),
                "epochs": int(args.epochs),
                "avg_cost": "%.10g" % mean_c,
                "std_cost": "%.10g" % std_c,
                "median_cost": "%.10g" % med_c,
                "avg_time": "%.10g" % mean_t,
                "std_time": "%.10g" % std_t,
                "mean_regret": ("" if mreg is None else "%.10g" % mreg),
            }
            rows_out.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Add legacy rows with avg_time column name for leaderboard (same as avg_time)
    legacy_path = out_dir / ("legacy_leaderboard_%s.csv" % stamp)
    with legacy_path.open("w", newline="", encoding="utf-8") as f:
        leg_fields = [
            "algo",
            "function",
            "stratum",
            "avg_cost",
            "avg_time",
            "std_cost",
        ]
        w = csv.DictWriter(f, fieldnames=leg_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(
                {
                    "algo": r["algo"],
                    "function": r["function"],
                    "stratum": r["stratum"],
                    "avg_cost": r["avg_cost"],
                    "avg_time": r["avg_time"],
                    "std_cost": r["std_cost"],
                }
            )

    print("Wrote %s" % path)
    print("Wrote %s (leaderboard-compatible: algo, function, stratum, avg_*, std_cost)" % legacy_path)

    # Stratified mean ranks (Dewancker-style layer on top of point estimates)
    mr = metrics.mean_ranks_by_stratum(
        [
            {
                "stratum": r["stratum"],
                "function": r["function"],
                "algo": r["algo"],
                "avg_cost": float(r["avg_cost"]),
            }
            for r in rows_out
        ]
    )
    if mr:
        print("\nMean rank per stratum (1 = best; averaged over functions in that stratum):")
        for s, al, avg_r in mr:
            print("  %r / %r : %.2f" % (s, al, avg_r))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
