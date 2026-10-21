"""Smoke tests for legacy benchmark score tooling."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATE_SCORES = ROOT / "benchmarks" / "generate_scores.py"


class TestGenerateScoresEntrypoint(unittest.TestCase):
    def test_leaderboard_without_csvs_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(GENERATE_SCORES), "--leaderboard"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No CSV files found in csvfiles", result.stdout)


if __name__ == "__main__":
    unittest.main()
