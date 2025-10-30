#!/usr/bin/env python3
"""
SwarmOpt Test Runner

Quick entry point to run the full SwarmOpt test suite.
"""

import sys
from pathlib import Path

# Add tests directory to path
tests_dir = Path(__file__).parent / 'tests'
sys.path.insert(0, str(tests_dir))

# Import and run the test index
from index import run_all_tests, show_index

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("SwarmOpt Test Runner")
        print()
        print("Usage:")
        print("  python run_tests.py              # Run full test suite")
        print("  python run_tests.py --show       # Show test index")
        print("  python run_tests.py --unit       # Run only unit tests")
        print("  python run_tests.py --help       # Show this help")
        print()
        print("Or use tests/index.py for interactive mode:")
        print("  python tests/index.py")
    elif len(sys.argv) > 1 and sys.argv[1] == '--show':
        show_index()
    elif len(sys.argv) > 1 and sys.argv[1] == '--unit':
        from index import run_unit_tests
        run_unit_tests()
    else:
        # Run full test suite
        run_all_tests()
