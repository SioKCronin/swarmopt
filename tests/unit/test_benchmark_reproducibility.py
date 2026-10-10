"""Benchmark runners should honor the seed they record in their outputs."""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BENCH_DIR = ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
run_benchmarks = load_module("bench_run_benchmarks", BENCH_DIR / "run_benchmarks.py")

research_suite = load_module(
    "research_benchmark_suite",
    ROOT / "tests" / "benchmarks" / "run_suite.py",
)


class TestBenchmarkSeedReproducibility(unittest.TestCase):
    def test_stratified_runner_passes_seed_to_swarm(self):
        problem = run_benchmarks.suite.get_suite("default")[0]
        args = (problem, "global", 8, 3, 2.0, 2.0, 0.9, 8, 123)

        first_cost, _ = run_benchmarks._run_trial(*args)
        second_cost, _ = run_benchmarks._run_trial(*args)

        self.assertEqual(first_cost, second_cost)

    def test_research_runner_passes_seed_to_swarm(self):
        preset = research_suite.ALGORITHM_PRESETS["global_linear"]
        args = (
            "global_linear",
            preset,
            "sphere",
            research_suite.sphere,
            3,
            8,
            8,
            (-5.12, 5.12),
            123,
        )

        first = research_suite.run_single(*args)
        second = research_suite.run_single(*args)

        self.assertEqual(first["best_cost"], second["best_cost"])


if __name__ == "__main__":
    unittest.main()
