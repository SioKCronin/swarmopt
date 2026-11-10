"""Benchmark helpers should make recorded trial seeds reproducible."""

import importlib.util
import unittest
from pathlib import Path

from benchmarks import run_benchmarks, suite
from swarmopt.functions import sphere


ROOT = Path(__file__).resolve().parents[2]


def _load_benchmark_run_suite():
    path = ROOT / "tests" / "benchmarks" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("benchmark_run_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBenchmarkSeedReproducibility(unittest.TestCase):
    def test_stratified_runner_reuses_recorded_seed(self):
        problem = next(p for p in suite.get_suite("default") if p["name"] == "sphere")
        kwargs = dict(
            problem=problem,
            algo_key="global",
            n_particles=8,
            dims=3,
            c1=2.0,
            c2=2.0,
            w=0.9,
            epochs=8,
            seed=123,
        )

        first_cost, _ = run_benchmarks._run_trial(**kwargs)
        second_cost, _ = run_benchmarks._run_trial(**kwargs)

        self.assertEqual(first_cost, second_cost)

    def test_research_runner_reuses_recorded_seed(self):
        run_suite = _load_benchmark_run_suite()
        kwargs = dict(
            algorithm_id="global_linear",
            preset=run_suite.ALGORITHM_PRESETS["global_linear"],
            function_name="sphere",
            obj_func=sphere,
            dims=3,
            n_particles=8,
            epochs=8,
            velocity_clamp=(-5, 5),
            seed=123,
        )

        first = run_suite.run_single(**kwargs)
        second = run_suite.run_single(**kwargs)

        self.assertEqual(first["best_cost"], second["best_cost"])


if __name__ == "__main__":
    unittest.main()
