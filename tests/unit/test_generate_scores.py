"""Regression tests for benchmark compatibility entry points."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent


class TestGenerateScoresEntry(unittest.TestCase):
    def test_legacy_entry_forwards_to_run_benchmarks(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/generate_scores.py",
                    "--quick",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(
                result.returncode,
                0,
                "stdout:\n%s\nstderr:\n%s" % (result.stdout, result.stderr),
            )
            self.assertTrue(list(out_dir.glob("benchmark_stratified_*.csv")))
            self.assertTrue(list(out_dir.glob("legacy_leaderboard_*.csv")))


if __name__ == "__main__":
    unittest.main()
