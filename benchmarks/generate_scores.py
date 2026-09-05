"""
Legacy entry point. Prefer:

    python benchmarks/run_benchmarks.py
"""

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
sys.path[:0] = [str(ROOT), str(BENCH)]

if __name__ == "__main__":
    from run_benchmarks import main

    raise SystemExit(main())
