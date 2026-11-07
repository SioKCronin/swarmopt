"""Compare the Rust bounded PSO with Python SwarmOpt on seeded benchmarks.

The two implementations use different random generators (SplitMix64 vs.
NumPy PCG64), so individual runs are not expected to match. What should match
is the *distribution* of results over many seeds with the same settings:
30 particles, c1 = c2 = 2.0, linear inertia 0.9 -> 0.4, velocity bound of
20% of the search range, 100 epochs.

Known behavioural difference: the Rust core clamps positions to the search
bounds (absorbing wall); Python SwarmOpt does not clamp positions.

Usage (from the repository root, with swarmopt installed):

    cargo build --release --manifest-path prototypes/deep_space/Cargo.toml
    python prototypes/deep_space/compare_python.py --seeds 30
"""

import argparse
import pathlib
import statistics
import subprocess
import sys

from swarmopt import Swarm
from swarmopt.functions import rosenbrock, sphere

HERE = pathlib.Path(__file__).resolve().parent
BIN = HERE / "target" / "release" / "deep-space-sim"
PROBLEMS = {
    "sphere": (sphere, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
}


def rust_costs(func, dims, seeds, epochs):
    out = subprocess.run(
        [str(BIN), "bench", func, str(dims), "0", str(seeds), "30", str(epochs)],
        check=True, capture_output=True, text=True,
    ).stdout
    return [float(line) for line in out.split()]


def python_costs(func, dims, seeds, epochs):
    obj, bounds = PROBLEMS[func]
    costs = []
    for seed in range(seeds):
        s = Swarm(30, dims, 2.0, 2.0, 0.9, epochs, obj, algo="global",
                  inertia_func="linear", velocity_clamp=bounds, seed=seed)
        s.optimize()
        costs.append(float(s.best_cost))
    return costs


def summary(xs):
    xs = sorted(xs)
    return statistics.median(xs), xs[len(xs) // 4], xs[(3 * len(xs)) // 4]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--dims", type=int, nargs="+", default=[2, 10])
    args = p.parse_args(argv)
    if not BIN.exists():
        sys.exit(f"build the Rust binary first: {BIN} not found")

    print(f"{args.seeds} seeds, {args.epochs} epochs, 30 particles; median [IQR] best cost")
    print(f"{'function':<11}{'dims':>5}  {'Rust':>32}  {'Python SwarmOpt':>32}")
    for func in PROBLEMS:
        for dims in args.dims:
            r = summary(rust_costs(func, dims, args.seeds, args.epochs))
            py = summary(python_costs(func, dims, args.seeds, args.epochs))
            fmt = lambda t: f"{t[0]:.3e} [{t[1]:.2e}, {t[2]:.2e}]"
            print(f"{func:<11}{dims:>5}  {fmt(r):>32}  {fmt(py):>32}")


if __name__ == "__main__":
    main()
