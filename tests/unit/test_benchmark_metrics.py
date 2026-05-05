"""Tests for benchmarks/metrics.py (no Swarm / scipy)."""

import sys
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))
import metrics  # noqa: E402


class TestBenchmarkMetrics(unittest.TestCase):
    def test_simple_regret(self):
        self.assertAlmostEqual(metrics.simple_regret(0.1, 0.0), 0.1)
        self.assertEqual(metrics.simple_regret(0.0, 0.0), 0.0)
        self.assertIsNone(metrics.simple_regret(1.0, None))

    def test_summarize_costs(self):
        m, s, med = metrics.summarize_costs([1.0, 3.0])
        self.assertEqual(m, 2.0)
        self.assertEqual(med, 2.0)
        self.assertGreater(s, 0.0)
        m, s, med = metrics.summarize_costs([5.0])
        self.assertEqual(s, 0.0)

    def test_mean_ranks_by_stratum(self):
        rows = [
            {"stratum": "a", "function": "f1", "algo": "x", "avg_cost": 1.0},
            {"stratum": "a", "function": "f1", "algo": "y", "avg_cost": 2.0},
            {"stratum": "a", "function": "f2", "algo": "x", "avg_cost": 2.0},
            {"stratum": "a", "function": "f2", "algo": "y", "avg_cost": 1.0},
        ]
        r = metrics.mean_ranks_by_stratum(rows)
        by_algo = {al: avg for s, al, avg in r if s == "a"}
        self.assertAlmostEqual(by_algo["x"], 1.5)
        self.assertAlmostEqual(by_algo["y"], 1.5)


if __name__ == "__main__":
    unittest.main()
