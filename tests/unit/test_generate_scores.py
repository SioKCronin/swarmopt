"""Smoke tests for legacy benchmark entry points."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGenerateScores(unittest.TestCase):
    def test_legacy_entry_point_forwards_to_benchmark_runner(self):
        repo_root = Path(__file__).resolve().parent.parent.parent
        script = repo_root / "benchmarks" / "generate_scores.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--quick",
                    "--out-dir",
                    tmpdir,
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            outputs = {p.name for p in Path(tmpdir).glob("*.csv")}
            self.assertTrue(any(name.startswith("benchmark_stratified_") for name in outputs))
            self.assertTrue(any(name.startswith("legacy_leaderboard_") for name in outputs))


if __name__ == "__main__":
    unittest.main()
