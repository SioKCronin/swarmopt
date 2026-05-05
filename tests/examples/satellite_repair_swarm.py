#!/usr/bin/env python3
"""
Satellite repair swarm positioning demo with 3D visualization and HTML report.

Generates:
- examples_output/satellite_repair_swarm.png — preview image
- examples_output/satellite_repair_report.html — open in a browser (file://)

Use case:
- A damaged satellite is at a fixed target position.
- Helper drones must hold safe standoff positions around it.
- The swarm optimizes for delegate-position coverage while respecting a
  mandatory exclusion boundary around the satellite.

Leaderboard: compares standard PSO variants on the same objective and seed.
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

# Headless, faster saves (safe for CI and file:// reports).
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from swarmopt import Swarm
from swarmopt.utils.example_html_report import build_example_html_page, png_to_data_uri

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "examples_output"


def _coverage_cost(position: np.ndarray, delegate_positions: Iterable[np.ndarray]) -> float:
    """Distance to nearest delegate (lower is better)."""
    return min(float(np.linalg.norm(position - delegate)) for delegate in delegate_positions)


def _run_with_history(swarm: Swarm) -> list[np.ndarray]:
    """Run PSO loop manually so we can capture trajectory history."""
    history: list[np.ndarray] = []
    for epoch in range(swarm.epochs):
        history.append(np.array([p.pos.copy() for p in swarm.swarm]))
        for particle in swarm.swarm:
            particle.update(epoch)
        swarm.update_local_best_pos()
        swarm.update_global_best_pos()
        swarm.update_global_worst_pos()
    return history


def _make_repair_objective(delegate_positions: list[np.ndarray]) -> Callable[[np.ndarray], float]:
    def repair_objective(pos: np.ndarray) -> float:
        return _coverage_cost(pos, delegate_positions)

    return repair_objective


def _build_delegate_positions(
    target_position: np.ndarray,
    n_delegates: int,
) -> list[np.ndarray]:
    bootstrap_swarm = Swarm(
        n_particles=24,
        dims=3,
        c1=1.8,
        c2=2.2,
        w=0.8,
        epochs=1,
        obj_func=lambda x: float(np.linalg.norm(x)),
        algo="global",
        velocity_clamp=(-8.0, 8.0),
        inertia_func="exponential",
        velocity_clamp_func="hybrid",
        target_position=target_position,
        n_delegates=n_delegates,
        delegate_spread="uniform",
    )
    return [np.array(d) for d in bootstrap_swarm.delegate_positions]


def _swarm_factory(
    repair_objective: Callable[[np.ndarray], float],
    target_position: np.ndarray,
    n_delegates: int,
    algo: str,
    epochs: int,
    n_particles: int,
) -> Swarm:
    return Swarm(
        n_particles=n_particles,
        dims=3,
        c1=1.8,
        c2=2.2,
        w=0.8,
        epochs=epochs,
        obj_func=repair_objective,
        algo=algo,
        velocity_clamp=(-8.0, 8.0),
        inertia_func="exponential",
        velocity_clamp_func="hybrid",
        target_position=target_position,
        n_delegates=n_delegates,
        delegate_spread="uniform",
    )


def _benchmark_algorithms(
    algorithms: list[tuple[str, str]],
    repair_objective: Callable[[np.ndarray], float],
    target_position: np.ndarray,
    n_delegates: int,
    epochs: int,
    n_particles: int,
    seed: int = 42,
) -> list[dict]:
    """Return list of dicts: key, label, best_cost, runtime_s."""
    results: list[dict] = []
    for key, label in algorithms:
        np.random.seed(seed)
        try:
            swarm = _swarm_factory(
                repair_objective,
                target_position,
                n_delegates,
                key,
                epochs,
                n_particles,
            )
            t0 = time.perf_counter()
            _run_with_history(swarm)
            elapsed = time.perf_counter() - t0
            results.append(
                {
                    "key": key,
                    "label": label,
                    "best_cost": float(swarm.best_cost),
                    "runtime_s": float(elapsed),
                }
            )
        except Exception as exc:  # noqa: BLE001 — example script resilience
            results.append(
                {
                    "key": key,
                    "label": label,
                    "best_cost": float("inf"),
                    "runtime_s": float("nan"),
                    "error": str(exc),
                }
            )
    return results


def _rank_results(raw: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Return (enriched_sorted, table_rows).
    enriched entries include numeric cost_score, time_score, composite.
    """
    finite_costs = [r["best_cost"] for r in raw if np.isfinite(r["best_cost"])]
    finite_times = [r["runtime_s"] for r in raw if np.isfinite(r["runtime_s"])]
    min_cost = min(finite_costs) if finite_costs else 1.0
    min_time = min(finite_times) if finite_times else 1.0

    enriched: list[dict] = []
    for r in raw:
        cost = r["best_cost"]
        t = r["runtime_s"]
        if np.isfinite(cost) and cost > 0:
            cost_score = 100.0 * min_cost / cost
        else:
            cost_score = 0.0
        if np.isfinite(t) and t > 0:
            time_score = 100.0 * min_time / t
        else:
            time_score = 0.0
        composite = 0.6 * cost_score + 0.4 * time_score
        row = {**r, "cost_score": cost_score, "time_score": time_score, "composite": composite}
        enriched.append(row)

    enriched.sort(key=lambda x: x["composite"], reverse=True)
    table_rows: list[dict] = []
    for rank, item in enumerate(enriched, start=1):
        table_rows.append(
            {
                "rank": rank,
                "algorithm": item["label"],
                "best_cost": f"{item['best_cost']:.6f}" if np.isfinite(item["best_cost"]) else "—",
                "runtime_s": f"{item['runtime_s']:.4f}" if np.isfinite(item["runtime_s"]) else "—",
                "cost_score": f"{item['cost_score']:.1f}",
                "time_score": f"{item['time_score']:.1f}",
                "composite_score": f"{item['composite']:.1f}",
            }
        )
    return enriched, table_rows


def _render_figure(
    swarm: Swarm,
    history: list[np.ndarray],
    delegate_positions: list[np.ndarray],
    target_position: np.ndarray,
    png_path: Path,
) -> None:
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    trail_particles = min(10, swarm.n_particles)
    for p_idx in range(trail_particles):
        xs = [frame[p_idx, 0] for frame in history]
        ys = [frame[p_idx, 1] for frame in history]
        zs = [frame[p_idx, 2] for frame in history]
        ax.plot(xs, ys, zs, linewidth=1.0, alpha=0.35, color="steelblue")

    final_positions = np.array([p.pos for p in swarm.swarm])
    ax.scatter(
        final_positions[:, 0],
        final_positions[:, 1],
        final_positions[:, 2],
        s=45,
        c="royalblue",
        label="Helper swarm (final)",
    )

    delegates = np.array(delegate_positions)
    ax.scatter(
        delegates[:, 0],
        delegates[:, 1],
        delegates[:, 2],
        s=90,
        c="darkorange",
        marker="^",
        label="Repair delegate positions",
    )

    # Lighter mesh than full plot_surface for speed.
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 14)
    r = float(swarm.respect_boundary)
    xs = target_position[0] + r * np.outer(np.cos(u), np.sin(v))
    ys = target_position[1] + r * np.outer(np.sin(u), np.sin(v))
    zs = target_position[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="crimson", alpha=0.12, linewidth=0, antialiased=True)

    ax.scatter(
        [target_position[0]],
        [target_position[1]],
        [target_position[2]],
        s=180,
        c="crimson",
        marker="*",
        label="Damaged satellite",
    )

    ax.set_title("Satellite Repair Helper Swarm Positioning")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")
    ax.view_init(elev=24, azim=42)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_satellite_repair_demo(
    output_dir: Path | None = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    png_path = out / "satellite_repair_swarm.png"
    html_path = out / "satellite_repair_report.html"

    target_position = np.array([0.0, 0.0, 0.0])
    n_delegates = 8
    epochs = 120
    n_particles = 24
    seed = 42

    delegate_positions = _build_delegate_positions(target_position, n_delegates)
    repair_objective = _make_repair_objective(delegate_positions)

    algorithms: list[tuple[str, str]] = [
        ("global", "Global Best PSO"),
        ("local", "Local Best PSO"),
        ("unified", "Unified PSO"),
        ("sa", "Simulated Annealing PSO"),
    ]

    bench = _benchmark_algorithms(
        algorithms,
        repair_objective,
        target_position,
        n_delegates,
        epochs,
        n_particles,
        seed=seed,
    )
    enriched, table_rows = _rank_results(bench)
    winner_key = enriched[0]["key"] if enriched else "global"

    np.random.seed(seed)
    viz_swarm = _swarm_factory(
        repair_objective,
        target_position,
        n_delegates,
        winner_key,
        epochs,
        n_particles,
    )
    history = _run_with_history(viz_swarm)
    assignments = viz_swarm.get_delegate_assignments()

    _render_figure(viz_swarm, history, delegate_positions, target_position, png_path)

    intro = (
        "<p>Helper swarms position around a damaged satellite while enforcing a mandatory "
        "<strong>respect boundary</strong> (standoff). Delegate sites are precomputed on a sphere; "
        "particles optimize coverage (min distance to nearest delegate). Visualization uses the "
        f"<strong>highest composite-score</strong> algorithm: <code>{html.escape(winner_key)}</code>.</p>"
        "<p><strong>Delegate assignments</strong> (nearest unique particle):</p><ul>"
    )
    for delegate_idx, payload in sorted(assignments.items()):
        intro += (
            f"<li>Delegate {delegate_idx:02d} &larr; particle {payload['particle_index']:02d} "
            f"(distance={payload['distance']:.4f})</li>"
        )
    intro += "</ul>"

    data_uri = png_to_data_uri(png_path)
    page = build_example_html_page(
        title="SwarmOpt — Satellite repair helper swarm",
        intro_html=intro,
        image_data_uri=data_uri,
        leaderboard_rows=table_rows,
        image_alt="Satellite repair swarm 3D plot",
    )
    html_path.write_text(page, encoding="utf-8")

    print("=== Satellite Repair Swarm Demo ===")
    print(f"Output directory: {out.resolve()}")
    print(f"Particles: {n_particles}, Delegates: {n_delegates}, Epochs: {epochs}")
    print(f"Respect boundary radius: {viz_swarm.respect_boundary:.3f}")
    print(f"Viz algorithm (composite winner): {winner_key}")
    print(f"Best cost (viz run): {viz_swarm.best_cost:.6f}")
    print("")
    print("Leaderboard (composite = 0.6×cost score + 0.4×time score):")
    for row in table_rows:
        print(
            f"  #{row['rank']} {row['algorithm']}: cost={row['best_cost']} "
            f"time={row['runtime_s']}s composite={row['composite_score']}"
        )
    print("")
    print(f"PNG:  {png_path.resolve()}")
    print(f"HTML: {html_path.resolve()}")
    print("Open the HTML file in your browser for the full report.")
    return html_path


if __name__ == "__main__":
    run_satellite_repair_demo()
