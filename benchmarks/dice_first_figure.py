"""First DICE figure: liar regret, and recovery messages against swarm size.

The plan's four implementation steps end in two curves. This script runs
them and writes a CSV plus a PNG.

    python benchmarks/dice_first_figure.py
    python benchmarks/dice_first_figure.py --quick
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from swarmopt import Swarm
from swarmopt.functions import rastrigin, sphere

import metrics

OUT = ROOT / "dice_figure"


def _make(n, epochs, obj, seed, **kwargs):
    return Swarm(
        n_particles=n,
        dims=2,
        c1=1.4,
        c2=1.4,
        w=0.7,
        epochs=epochs,
        obj_func=obj,
        algo="local",
        seed=seed,
        velocity_clamp=(-5.12, 5.12),
        k=4,
        topology="ring",
        **kwargs,
    )


def _run(n, epochs, obj, seed, **kwargs):
    swarm = _make(n, epochs, obj, seed, **kwargs)
    swarm.optimize()
    return swarm


def liar_regret(quick: bool):
    fractions = (0.0, 0.5) if quick else (0.0, 0.1, 0.25, 0.5)
    functions = (("sphere", sphere),) if quick else (("sphere", sphere), ("rastrigin", rastrigin))
    n = 40 if quick else 500
    epochs = 8 if quick else 40
    rows = []
    for name, func in functions:
        for defense in ("trust", "spot-check"):
            for fraction in fractions:
                swarm = _run(
                    n, epochs, func, seed=0,
                    liar_fraction=fraction, defense=defense,
                )
                rows.append({
                    "function": name,
                    "defense": defense,
                    "liar_fraction": fraction,
                    "n": n,
                    "regret": swarm.best_cost,
                    "messages_sent": swarm.messages_sent,
                })
                print(
                    f"liar {name} {defense} f={fraction}: "
                    f"regret={swarm.best_cost:.4g}"
                )
    return rows


def recovery_messages(quick: bool):
    sizes = (30, 60) if quick else (500, 2000, 5000)
    epochs = 12 if quick else 60
    period = 6 if quick else 20
    tolerance = 1.0
    jump = np.array([1.5, 0.0])
    rows = []
    for n in sizes:
        state = {
            "center": np.zeros(2),
            "open_sent": None,
            "recovered": [],
        }

        def objective(x, state=state):
            return float(np.sum((np.asarray(x, dtype=float) - state["center"]) ** 2))

        def on_epoch(epoch, swarm, state=state, period=period, jump=jump, tolerance=tolerance):
            if epoch > 0 and epoch % period == 0:
                state["center"] = state["center"] + jump
                state["open_sent"] = swarm.messages_sent
            elif state["open_sent"] is not None and swarm.best_cost <= tolerance:
                state["recovered"].append(swarm.messages_sent - state["open_sent"])
                state["open_sent"] = None

        swarm = _make(n, epochs, objective, seed=1)
        swarm.epoch_callback = on_epoch
        swarm.optimize()
        used = state["recovered"]
        mean_messages = float(np.mean(used)) if used else float("nan")
        rows.append({
            "n": n,
            "recoveries": len(used),
            "mean_messages": mean_messages,
            "messages_sent": swarm.messages_sent,
        })
        print(f"recovery n={n}: {len(used)} recoveries, mean messages={mean_messages:.4g}")
    return rows


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(liar_rows, recovery_rows, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    ax = axes[0]
    for function in dict.fromkeys(row["function"] for row in liar_rows):
        for defense, style in (("trust", "-"), ("spot-check", "--")):
            series = [
                row for row in liar_rows
                if row["function"] == function and row["defense"] == defense
            ]
            ax.plot(
                [row["liar_fraction"] for row in series],
                [row["regret"] for row in series],
                style,
                marker="o",
                label=f"{function}, {defense}",
            )
    ax.set_xlabel("Liar fraction")
    ax.set_ylabel("Regret (best cost, optimum is 0)")
    ax.set_title("Spot-check against a lying neighbour")
    ax.legend(fontsize=8)

    ax = axes[1]
    ns = [row["n"] for row in recovery_rows]
    messages = [row["mean_messages"] for row in recovery_rows]
    ax.plot(ns, messages, marker="o", label="Measured recovery messages")
    if all(np.isfinite(messages)) and len(ns) >= 2:
        scale = messages[0] / ns[0]
        ax.plot(ns, [scale * n for n in ns], linestyle=":", label="Linear in N")
        exponent = metrics.growth_exponent(ns, messages)
        ax.set_title(f"Recovery messages vs N (exponent {exponent:.2f})")
    else:
        ax.set_title("Recovery messages vs N")
    ax.set_xlabel("Swarm size N")
    ax.set_ylabel("Mean messages to recover")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny run for a smoke check")
    args = parser.parse_args()
    liar_rows = liar_regret(args.quick)
    recovery_rows = recovery_messages(args.quick)
    _write_csv(OUT / "liar_regret.csv", liar_rows)
    _write_csv(OUT / "recovery_messages.csv", recovery_rows)
    finite = [row for row in recovery_rows if np.isfinite(row["mean_messages"])]
    if len(finite) >= 2:
        exponent = metrics.growth_exponent(
            [row["n"] for row in finite],
            [row["mean_messages"] for row in finite],
        )
        print(f"recovery growth exponent {exponent:.3f}")
    _plot(liar_rows, recovery_rows, OUT / "first_figure.png")


if __name__ == "__main__":
    main()
