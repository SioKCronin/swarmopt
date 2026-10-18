import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestBenchmarkEntrypoints(unittest.TestCase):
    def test_generate_scores_forwards_to_run_benchmarks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/generate_scores.py",
                    "--quick",
                    "--out-dir",
                    tmpdir,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Wrote", result.stdout)
            outputs = {path.name for path in Path(tmpdir).glob("*.csv")}
            self.assertTrue(any(name.startswith("benchmark_stratified_") for name in outputs))
            self.assertTrue(any(name.startswith("legacy_leaderboard_") for name in outputs))


if __name__ == "__main__":
    unittest.main()
