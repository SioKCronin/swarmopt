#!/usr/bin/env python3
"""
SwarmOpt benchmark suite for researchers.

Runs multiple (algorithm × function × run) combinations with fixed seeds,
outputs CSV and optional JSON for analysis and papers.

Usage:
  python run_suite.py                    # default quick config
  python run_suite.py --config full      # config name from configs/
  python run_suite.py --config configs/quick.json
  python run_suite.py --runs 5 --output results/
  python run_suite.py --list-functions   # show available functions
  python run_suite.py --list-algorithms  # show available algorithm presets
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Project root
TESTS_DIR = Path(__file__).resolve().parent.parent
ROOT = TESTS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import (
    FUNCTION_METADATA,
    sphere,
    rosenbrock,
    ackley,
    rastrigin,
    griewank,
    weierstrass,
    sum_squares,
    zakharov,
    levy,
    get_function_metadata,
)

# Name -> callable for benchmark functions (extend as needed)
BENCHMARK_FUNCTIONS = {
    "sphere": sphere,
    "sum_squares": sum_squares,
    "zakharov": zakharov,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
    "griewank": griewank,
    "rastrigin": rastrigin,
    "levy": levy,
    "weierstrass": weierstrass,
}

# Algorithm presets: name -> dict of Swarm kwargs (algo, inertia_func, etc.)
ALGORITHM_PRESETS = {
    "global_constant": {"algo": "global", "inertia_func": "constant", "velocity_clamp_func": "basic"},
    "global_linear": {"algo": "global", "inertia_func": "linear", "velocity_clamp_func": "basic"},
    "global_adaptive": {"algo": "global", "inertia_func": "adaptive", "velocity_clamp_func": "adaptive"},
    "global_exponential": {"algo": "global", "inertia_func": "exponential", "velocity_clamp_func": "basic"},
    "local_linear": {"algo": "local", "inertia_func": "linear", "velocity_clamp_func": "basic"},
    "unified_linear": {"algo": "unified", "inertia_func": "linear", "velocity_clamp_func": "basic"},
    "cpso_best": {"algo": "cpso", "n_swarms": 3, "communication_strategy": "best"},
    "cpso_tournament": {"algo": "cpso", "n_swarms": 3, "communication_strategy": "tournament"},
    "ppso": {"algo": "global", "ppso_enabled": True, "proactive_ratio": 0.25, "inertia_func": "linear"},
}


def get_function_by_name(name):
    """Return objective function callable by name."""
    if name in BENCHMARK_FUNCTIONS:
        return BENCHMARK_FUNCTIONS[name]
    # Try to get from swarmopt.functions
    try:
        from swarmopt import functions as fn_mod
        return getattr(fn_mod, name, None)
    except Exception:
        return None


def get_bounds_for_function(name, dims=10):
    """Return (low, high) bounds for a function. Default dims for 'any' dimension functions."""
    meta = get_function_metadata(name)
    if not meta:
        return (-10.0, 10.0)
    b = meta.get("bounds")
    if b is None:
        return (-10.0, 10.0)
    if isinstance(b, list) and len(b) == 2:
        return (float(b[0]), float(b[1]))
    return (-10.0, 10.0)


def run_single(algorithm_id, preset, function_name, obj_func, dims, n_particles, epochs, velocity_clamp, seed):
    """Run one optimization and return best_cost, runtime, seed."""
    np.random.seed(seed)
    t0 = time.perf_counter()
    swarm = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0,
        c2=2.0,
        w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        velocity_clamp=velocity_clamp,
        w_start=0.9,
        w_end=0.4,
        **preset,
    )
    swarm.optimize()
    elapsed = time.perf_counter() - t0
    return {
        "algorithm": algorithm_id,
        "function": function_name,
        "best_cost": float(swarm.best_cost),
        "runtime_sec": round(elapsed, 4),
        "seed": seed,
        "dims": dims,
        "n_particles": n_particles,
        "epochs": epochs,
    }


def load_config(path_or_name):
    """Load benchmark config: JSON file or built-in name (quick, full, unimodal, multimodal)."""
    builtin = {
        "quick": {
            "algorithms": ["global_linear", "global_adaptive", "cpso_best"],
            "functions": ["sphere", "rosenbrock", "ackley"],
            "dims": 5,
            "n_particles": 20,
            "epochs": 30,
            "runs_per_cell": 2,
            "velocity_clamp": (-5, 5),
        },
        "medium": {
            "algorithms": ["global_linear", "global_adaptive", "local_linear", "cpso_best", "ppso"],
            "functions": ["sphere", "rosenbrock", "ackley", "rastrigin", "griewank"],
            "dims": 10,
            "n_particles": 30,
            "epochs": 50,
            "runs_per_cell": 3,
            "velocity_clamp": (-5, 5),
        },
        "full": {
            "algorithms": list(ALGORITHM_PRESETS.keys()),
            "functions": list(BENCHMARK_FUNCTIONS.keys()),
            "dims": 10,
            "n_particles": 30,
            "epochs": 80,
            "runs_per_cell": 5,
            "velocity_clamp": (-5, 5),
        },
        "unimodal": {
            "algorithms": ["global_linear", "global_adaptive", "cpso_best"],
            "functions": ["sphere", "sum_squares", "zakharov"],
            "dims": 10,
            "n_particles": 25,
            "epochs": 50,
            "runs_per_cell": 4,
            "velocity_clamp": (-5, 5),
        },
        "multimodal": {
            "algorithms": ["global_linear", "global_adaptive", "local_linear", "cpso_best", "ppso"],
            "functions": ["rosenbrock", "ackley", "rastrigin", "griewank", "weierstrass"],
            "dims": 10,
            "n_particles": 30,
            "epochs": 80,
            "runs_per_cell": 4,
            "velocity_clamp": (-5, 5),
        },
    }
    if path_or_name in builtin:
        return builtin[path_or_name].copy()
    p = Path(path_or_name)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / "configs" / (path_or_name + ".json")
    if not p.exists():
        p = Path(path_or_name)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    raise FileNotFoundError(f"Config not found: {path_or_name}")


def run_benchmark_suite(config, output_dir=None, verbose=True):
    """Run full suite from config. Returns list of result dicts."""
    algorithms = config.get("algorithms", ["global_linear"])
    functions = config.get("functions", ["sphere", "ackley"])
    dims = int(config.get("dims", 10))
    n_particles = int(config.get("n_particles", 30))
    epochs = int(config.get("epochs", 50))
    runs_per_cell = int(config.get("runs_per_cell", 3))
    velocity_clamp = tuple(config.get("velocity_clamp", [-5, 5]))
    base_seed = int(config.get("seed", 42))

    results = []
    total = len(algorithms) * len(functions) * runs_per_cell
    run_idx = 0
    for algo_id in algorithms:
        preset = ALGORITHM_PRESETS.get(algo_id, {"algo": "global", "inertia_func": "linear"})
        for func_name in functions:
            obj_func = get_function_by_name(func_name)
            if obj_func is None:
                if verbose:
                    print(f"  Skip unknown function: {func_name}")
                continue
            for r in range(runs_per_cell):
                seed = base_seed + r + run_idx * 1000
                run_idx += 1
                if verbose:
                    print(f"  [{run_idx}/{total}] {algo_id} × {func_name} run {r+1}/{runs_per_cell} (seed={seed})")
                row = run_single(
                    algo_id, preset, func_name, obj_func,
                    dims, n_particles, epochs, velocity_clamp, seed,
                )
                results.append(row)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "benchmark_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        if verbose:
            print(f"\n  Wrote {csv_path}")
        json_path = output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"  Wrote {json_path}")
    return results


def print_summary_table(results):
    """Print a compact mean ± std table per (algorithm, function)."""
    from collections import defaultdict
    keyed = defaultdict(list)
    for r in results:
        keyed[(r["algorithm"], r["function"])].append(r["best_cost"])
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (mean ± std of best_cost)")
    print("=" * 80)
    algorithms = sorted({k[0] for k in keyed})
    functions = sorted({k[1] for k in keyed})
    # Header
    print(f"{'Algorithm':<22}", end="")
    for f in functions:
        print(f" {f[:12]:>12}", end="")
    print()
    print("-" * (22 + 13 * len(functions)))
    for algo in algorithms:
        print(f"{algo:<22}", end="")
        for func in functions:
            vals = keyed.get((algo, func), [])
            if vals:
                m = np.mean(vals)
                s = np.std(vals)
                print(f" {m:8.4f}±{s:.2f}", end="")
            else:
                print(f" {'—':>12}", end="")
        print()
    print()


def main():
    parser = argparse.ArgumentParser(description="SwarmOpt benchmark suite for researchers")
    parser.add_argument("--config", "-c", default="quick", help="Config name or path (quick, medium, full, unimodal, multimodal, or path)")
    parser.add_argument("--output", "-o", default=None, help="Output directory for CSV/JSON")
    parser.add_argument("--runs", "-r", type=int, default=None, help="Override runs_per_cell")
    parser.add_argument("--list-functions", action="store_true", help="List available functions")
    parser.add_argument("--list-algorithms", action="store_true", help="List algorithm presets")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less output")
    args = parser.parse_args()

    if args.list_functions:
        print("Available benchmark functions:")
        for name in sorted(BENCHMARK_FUNCTIONS.keys()):
            meta = get_function_metadata(name)
            ftype = meta.get("type", "?") if meta else "?"
            print(f"  {name:<20} type={ftype}")
        return 0

    if args.list_algorithms:
        print("Algorithm presets:")
        for name, preset in ALGORITHM_PRESETS.items():
            print(f"  {name:<22} {preset}")
        return 0

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if args.runs is not None:
        config["runs_per_cell"] = args.runs
    if args.output is None and not args.quiet:
        config["_output_dir"] = Path.cwd() / "benchmark_results"
    else:
        config["_output_dir"] = Path(args.output) if args.output else None

    print("SwarmOpt benchmark suite")
    print(f"  Config: {args.config}  |  Output: {config.get('_output_dir', 'none')}")
    print()

    results = run_benchmark_suite(
        config,
        output_dir=config.get("_output_dir"),
        verbose=not args.quiet,
    )
    if not results:
        print("No results.", file=sys.stderr)
        return 1
    print_summary_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
