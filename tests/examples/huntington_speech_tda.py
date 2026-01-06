#!/usr/bin/env python3
"""
Huntington Speech Topology Demo
================================

This example wires the multiobjective PSO in SwarmOpt to a Dionysus-powered
persistent homology pipeline tailored for synthetic speech features. The goal
is to show how the Pareto archive can surface distinct topological classes
from competing objectives (healthy-like vs Huntington-like speech plus jitter
suppression). After optimization we cluster the archive to expose representative
topology archetypes.
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import dionysus as dio
except ImportError:  # pragma: no cover - runtime guard for optional dependency
    dio = None

from swarmopt import Swarm


@dataclass
class TopologyMetadata:
    """Container for the rich metadata we stash per particle evaluation."""

    params: np.ndarray
    barcode: Dict[str, np.ndarray]
    feature_vector: np.ndarray
    jitter_metric: float
    energy_swing: float
    objectives: np.ndarray


class SpeechTopologyObjective:
    """
    Multiobjective callable that synthesizes speech features, computes persistent
    homology and emits objective scores while caching topology metadata.
    """

    def __init__(self, bounds: List[Tuple[float, float]], max_dim: int = 2, radius: float = 2.5):
        if dio is None:
            raise ImportError(
                "Dionysus is required for this example. Install with `pip install dionysus`."
            )

        self.bounds_min = np.array([b[0] for b in bounds], dtype=float)
        self.bounds_max = np.array([b[1] for b in bounds], dtype=float)
        self.max_dim = max_dim
        self.radius = radius
        self.metadata: Dict[Tuple[float, ...], TopologyMetadata] = {}

        # Reference profiles: synthetic healthy vs Huntington-like parameterizations.
        self.reference_profiles = {
            "healthy": np.array([175.0, 0.12, 0.18], dtype=float),
            "huntington": np.array([155.0, 0.48, 0.55], dtype=float),
        }

        self.reference_features = {
            name: self._simulate_and_describe(params, store=False)["feature_vector"]
            for name, params in self.reference_profiles.items()
        }

    def __call__(self, params: np.ndarray) -> np.ndarray:
        params = np.clip(params, self.bounds_min, self.bounds_max)
        description = self._simulate_and_describe(params, store=False)

        feat = description["feature_vector"]
        jitter = description["jitter_metric"]
        energy_swing = description["energy_swing"]

        # Objectives:
        #   1. Stay close to the healthy topology descriptor.
        #   2. Stay close to the Huntington topology descriptor (contradictory on purpose).
        #   3. Minimize vocal instability via jitter + energy swings.
        obj_healthy = np.linalg.norm(feat - self.reference_features["healthy"])
        obj_huntington = np.linalg.norm(feat - self.reference_features["huntington"])
        obj_instability = jitter + 0.5 * energy_swing

        objectives = np.array([obj_healthy, obj_huntington, obj_instability], dtype=float)

        key = tuple(np.round(params, decimals=6))
        self.metadata[key] = TopologyMetadata(
            params=params.copy(),
            barcode=description["barcode"],
            feature_vector=feat,
            jitter_metric=jitter,
            energy_swing=energy_swing,
            objectives=objectives,
        )

        return objectives

    def _simulate_and_describe(self, params: np.ndarray, store: bool = True) -> Dict[str, np.ndarray]:
        point_cloud = self._generate_speech_point_cloud(params)
        barcode = self._compute_persistence(point_cloud)
        feature_vector = self._summarize_barcode(barcode)
        jitter_metric = self._phonatory_instability(point_cloud[:, 0])
        energy_swing = float(np.ptp(point_cloud[:, 1]))

        description = {
            "point_cloud": point_cloud,
            "barcode": barcode,
            "feature_vector": feature_vector,
            "jitter_metric": jitter_metric,
            "energy_swing": energy_swing,
        }

        if store:
            key = tuple(np.round(params, decimals=6))
            self.metadata[key] = TopologyMetadata(
                params=params.copy(),
                barcode=barcode,
                feature_vector=feature_vector,
                jitter_metric=jitter_metric,
                energy_swing=energy_swing,
                objectives=np.zeros(3, dtype=float),  # Placeholder, overwritten by __call__
            )

        return description

    def _generate_speech_point_cloud(self, params: np.ndarray, n_frames: int = 160) -> np.ndarray:
        """
        Generate a synthetic three-dimensional speech feature trajectory:
        [fundamental frequency, energy envelope, articulation irregularity]
        """
        base_pitch, tremor, breathiness = params.astype(float)
        t = np.linspace(0.0, 1.0, n_frames, dtype=float)

        # Fundamental frequency contour (Hz) with tremor-induced modulation.
        pitch = base_pitch + 15.0 * np.sin(2 * np.pi * (1.2 + tremor * 2.5) * t)
        pitch += 5.0 * tremor * np.sin(2 * np.pi * (5 + breathiness * 3.0) * t)

        # Energy envelope (normalized) impacted by breathiness and tremor.
        energy = (1.0 - breathiness) + 0.35 * np.sin(2 * np.pi * 2.3 * t)
        energy -= 0.25 * breathiness * np.sin(2 * np.pi * (4.2 + tremor) * t + 0.4)

        # Articulation irregularity proxy capturing jitter/shimmer effects.
        irregularity = 0.15 * np.sin(2 * np.pi * (6.0 + tremor * 8.0) * t)
        irregularity += 0.05 * np.sin(2 * np.pi * 18.0 * t + breathiness)
        irregularity += tremor * 0.6 * np.sin(2 * np.pi * (10.0 + breathiness * 4.0) * t + 1.3)

        features = np.column_stack([pitch, energy, irregularity])
        # Z-score normalization for stability in persistent homology.
        features -= features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        features /= std
        return features

    def _compute_persistence(self, point_cloud: np.ndarray) -> Dict[str, np.ndarray]:
        filtration = dio.fill_rips(point_cloud, self.max_dim, self.radius)
        persistence = dio.homology_persistence(filtration)
        diagrams = dio.init_diagrams(persistence, filtration)

        barcode: Dict[str, np.ndarray] = {}
        for dim in range(self.max_dim + 1):
            if dim >= len(diagrams):
                barcode[f"H{dim}"] = np.empty((0, 2), dtype=float)
                continue

            intervals = []
            for pt in diagrams[dim]:
                if math.isinf(pt.death):
                    continue
                intervals.append([pt.birth, pt.death])
            barcode[f"H{dim}"] = np.array(intervals, dtype=float) if intervals else np.empty((0, 2), dtype=float)

        return barcode

    @staticmethod
    def _summarize_barcode(barcode: Dict[str, np.ndarray]) -> np.ndarray:
        summary: List[float] = []
        for dim_key in ("H0", "H1", "H2"):
            intervals = barcode.get(dim_key, np.empty((0, 2)))
            if len(intervals) == 0:
                summary.extend([0.0, 0.0, 0.0, 0.0])
                continue

            persistence = intervals[:, 1] - intervals[:, 0]
            summary.append(float(len(intervals)))
            summary.append(float(np.sum(persistence)))
            summary.append(float(np.max(persistence)))
            summary.append(float(np.mean(intervals[:, 0])))

        return np.array(summary, dtype=float)

    @staticmethod
    def _phonatory_instability(pitch_series: np.ndarray) -> float:
        """Simple instability score based on frame-to-frame pitch fluctuations."""
        derivatives = np.diff(pitch_series)
        if derivatives.size == 0:
            return 0.0
        return float(np.mean(np.abs(derivatives)) + np.std(derivatives))


def simple_kmeans(data: np.ndarray, k: int, seed: int = 42, max_iter: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(len(data), size=k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = []
        for idx in range(k):
            cluster_points = data[labels == idx]
            if len(cluster_points) == 0:
                new_centroids.append(centroids[idx])
                continue
            new_centroids.append(cluster_points.mean(axis=0))

        new_centroids = np.vstack(new_centroids)
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids

    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def cluster_pareto_archive(
    archive: List[Dict[str, np.ndarray]],
    metadata: Dict[Tuple[float, ...], TopologyMetadata],
    max_clusters: int = 3,
) -> List[Dict[str, object]]:
    if not archive:
        return []

    entries = []
    feature_matrix = []

    for sol in archive:
        key = tuple(np.round(sol["pos"], decimals=6))
        meta = metadata.get(key)
        if meta is None:
            continue
        entries.append({"solution": sol, "metadata": meta})
        feature_matrix.append(meta.feature_vector)

    if not feature_matrix:
        return []

    data = np.vstack(feature_matrix)
    k = min(max_clusters, len(data))
    labels = simple_kmeans(data, k)

    clusters: List[Dict[str, object]] = []
    for cluster_id in range(k):
        cluster_entries = [entries[i] for i, label in enumerate(labels) if label == cluster_id]
        if not cluster_entries:
            continue

        centroid = np.mean([entry["metadata"].feature_vector for entry in cluster_entries], axis=0)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "centroid": centroid,
                "members": cluster_entries,
            }
        )

    return clusters


def main() -> None:
    if dio is None:
        warnings.warn(
            "Dionysus is not installed. Install it to run this example: pip install dionysus",
            RuntimeWarning,
        )
        sys.exit(0)

    print("=" * 72)
    print("🎤 Huntington Speech Persistent Homology Demo")
    print("=" * 72)

    # Parameter definitions: [baseline pitch, tremor magnitude, breathiness]
    bounds = [(140.0, 200.0), (0.05, 0.60), (0.05, 0.70)]
    objective = SpeechTopologyObjective(bounds=bounds, max_dim=2, radius=3.0)

    lower_bounds = np.array([b[0] for b in bounds], dtype=float)
    upper_bounds = np.array([b[1] for b in bounds], dtype=float)

    swarm = Swarm(
        n_particles=24,
        dims=3,
        c1=1.6,
        c2=1.8,
        w=0.78,
        epochs=60,
        obj_func=objective,
        multiobjective=True,
        archive_size=60,
        velocity_clamp=(lower_bounds, upper_bounds),
    )

    print("🚀 Running multiobjective PSO...")
    swarm.optimize()
    archive = swarm.mo_optimizer.archive if swarm.mo_optimizer else []
    print(f"✅ Optimization complete. Archive size: {len(archive)}")

    clusters = cluster_pareto_archive(archive, objective.metadata, max_clusters=4)
    if not clusters:
        print("⚠️ No clusters identified from archive metadata.")
        return

    print("\n🌀 Discovered Topology Classes:")
    for cluster in clusters:
        members = cluster["members"]
        centroid = cluster["centroid"]
        cluster_id = cluster["cluster_id"]

        avg_objectives = np.mean([entry["metadata"].objectives for entry in members], axis=0)
        avg_jitter = np.mean([entry["metadata"].jitter_metric for entry in members])
        avg_energy_swing = np.mean([entry["metadata"].energy_swing for entry in members])

        print("-" * 72)
        print(f"Cluster {cluster_id} · Members: {len(members)}")
        print(f"  Feature centroid: {np.round(centroid, 3)}")
        print(f"  Avg objectives [healthy, huntington, instability]: {np.round(avg_objectives, 3)}")
        print(f"  Avg jitter metric: {avg_jitter:.3f} | Avg energy swing: {avg_energy_swing:.3f}")

        representative = members[0]
        params = representative["metadata"].params
        barcode = representative["metadata"].barcode
        print(f"  Representative params: {np.round(params, 3)}")
        print(f"  H0 count: {len(barcode['H0'])} | H1 count: {len(barcode['H1'])} | H2 count: {len(barcode['H2'])}")

    print("-" * 72)
    print("📌 Use the clusters above to inspect distinct speech topology archetypes.")
    print("    • Cluster near healthy reference → intact loops & low jitter")
    print("    • Cluster near Huntington reference → persistent H1 features with higher instability")
    print("    • Mixed clusters → potential transitional speech phenotypes")


if __name__ == "__main__":
    main()


