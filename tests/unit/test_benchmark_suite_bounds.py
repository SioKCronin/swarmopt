"""Tests for researcher benchmark search-bound selection."""

import sys
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))
import run_suite  # noqa: E402


class RecordingSwarm:
    calls = []

    def __init__(self, **kwargs):
        self.best_cost = 0.0
        RecordingSwarm.calls.append(kwargs)

    def optimize(self):
        return None


class TestBenchmarkSuiteBounds(unittest.TestCase):
    def setUp(self):
        self.original_swarm = run_suite.Swarm
        run_suite.Swarm = RecordingSwarm
        RecordingSwarm.calls = []

    def tearDown(self):
        run_suite.Swarm = self.original_swarm

    def test_uses_function_metadata_bounds_by_default(self):
        config = {
            "algorithms": ["global_linear"],
            "functions": ["ackley", "weierstrass"],
            "dims": 5,
            "n_particles": 2,
            "epochs": 1,
            "runs_per_cell": 1,
        }

        run_suite.run_benchmark_suite(config, verbose=False)

        self.assertEqual(RecordingSwarm.calls[0]["velocity_clamp"], (-32.768, 32.768))
        self.assertEqual(RecordingSwarm.calls[1]["velocity_clamp"], (-0.5, 0.5))

    def test_legacy_velocity_clamp_does_not_override_function_bounds(self):
        config = {
            "algorithms": ["global_linear"],
            "functions": ["ackley"],
            "dims": 5,
            "n_particles": 2,
            "epochs": 1,
            "runs_per_cell": 1,
            "velocity_clamp": [-5, 5],
        }

        run_suite.run_benchmark_suite(config, verbose=False)

        self.assertEqual(RecordingSwarm.calls[0]["velocity_clamp"], (-32.768, 32.768))

    def test_search_bounds_can_override_metadata_for_custom_runs(self):
        config = {
            "algorithms": ["global_linear"],
            "functions": ["ackley"],
            "dims": 5,
            "n_particles": 2,
            "epochs": 1,
            "runs_per_cell": 1,
            "search_bounds": [-1, 1],
        }

        run_suite.run_benchmark_suite(config, verbose=False)

        self.assertEqual(RecordingSwarm.calls[0]["velocity_clamp"], (-1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
