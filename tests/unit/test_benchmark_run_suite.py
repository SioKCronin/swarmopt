import importlib.util
import unittest
from pathlib import Path

import context  # noqa: F401 - ensure project root is importable


def _load_run_suite():
    root = Path(__file__).resolve().parents[2]
    path = root / "tests" / "benchmarks" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("benchmark_run_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBenchmarkRunSuite(unittest.TestCase):
    def test_run_benchmark_suite_uses_function_metadata_bounds(self):
        run_suite = _load_run_suite()
        captured_bounds = []
        original_swarm = run_suite.Swarm

        class RecordingSwarm:
            def __init__(self, *args, **kwargs):
                captured_bounds.append(tuple(kwargs["velocity_clamp"]))
                self.best_cost = 0.0

            def optimize(self):
                return None

        try:
            run_suite.Swarm = RecordingSwarm
            run_suite.run_benchmark_suite(
                {
                    "algorithms": ["global_linear"],
                    "functions": ["ackley"],
                    "dims": 5,
                    "n_particles": 2,
                    "epochs": 1,
                    "runs_per_cell": 1,
                    "velocity_clamp": [-5, 5],
                    "seed": 42,
                },
                output_dir=None,
                verbose=False,
            )
        finally:
            run_suite.Swarm = original_swarm

        self.assertEqual(captured_bounds, [(-32.768, 32.768)])


if __name__ == "__main__":
    unittest.main()
