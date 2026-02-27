#!/usr/bin/env python3
"""
SwarmOpt Test Runner

Entry point to run the SwarmOpt test suite from the project root.
Delegates to tests/index.py for all behavior; exits with 0 on success, 1 on failure.
"""

import sys
from pathlib import Path

# Add tests directory to path so index can be imported
tests_dir = Path(__file__).parent / "tests"
sys.path.insert(0, str(tests_dir))

# Import test index (will run as main via subprocess so cwd and exit codes are correct)
if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        # Default: run full test suite and exit with 0/1
        from index import run_all_tests
        ok = run_all_tests(quick=False, include_examples=False)
        sys.exit(0 if ok else 1)

    # Parse flags
    show_help = "--help" in argv or "-h" in argv
    show_index = "--show" in argv or "-s" in argv
    quick = "--quick" in argv or "-q" in argv
    include_examples = "--with-examples" in argv  # include examples when running full suite
    argv = [a for a in argv if a not in ("--help", "-h", "--show", "-s", "--quick", "-q", "--with-examples")]

    if show_help:
        print("SwarmOpt Test Runner")
        print()
        print("Usage:")
        print("  python run_tests.py                    # Run full test suite (unit + integration)")
        print("  python run_tests.py --quick            # Skip long-running integration tests")
        print("  python run_tests.py --examples        # Also run example scripts")
        print("  python run_tests.py --unit            # Run only unit tests (pytest)")
        print("  python run_tests.py --integration     # Run all integration tests")
        print("  python run_tests.py --integration NAME  # Run one integration test")
        print("  python run_tests.py --with-examples   # Full suite + examples")
        print("  python run_tests.py --examples       # Run all examples only")
        print("  python run_tests.py --examples NAME  # Run one example")
        print("  python run_tests.py --benchmark [key] # Run researcher benchmark suite")
        print("  python run_tests.py --show            # Show test index and exit")
        print("  python run_tests.py --help            # Show this help")
        print()
        print("Interactive mode (pick by number):")
        print("  python tests/index.py")
        sys.exit(0)

    if show_index:
        from index import show_index
        show_index(quick=quick)
        sys.exit(0)

    from index import (
        run_all_tests,
        run_unit_tests,
        run_integration_test,
        run_example,
        get_test_categories,
    )

    if "--unit" in sys.argv:
        ok, _ = run_unit_tests()
        sys.exit(0 if ok else 1)

    if "--integration" in sys.argv:
        idx = sys.argv.index("--integration")
        name = sys.argv[idx + 1] if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-") else None
        if name:
            ok, _ = run_integration_test(name, quick=quick)
            sys.exit(0 if ok else 1)
        cats = get_test_categories(quick=quick)
        all_ok = True
        for script in cats["integration"]["scripts"].keys():
            ok, _ = run_integration_test(script, quick=quick)
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    if "--examples" in sys.argv:
        idx = sys.argv.index("--examples")
        name = sys.argv[idx + 1] if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-") else None
        if name:
            ok, _ = run_example(name)
            sys.exit(0 if ok else 1)
        cats = get_test_categories()
        all_ok = True
        for script in cats["examples"]["scripts"].keys():
            ok, _ = run_example(script)
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    if "--benchmark" in sys.argv:
        from index import run_benchmark, get_test_categories
        idx = sys.argv.index("--benchmark")
        key = sys.argv[idx + 1] if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-") else "run_suite.py"
        ok, _ = run_benchmark(key)
        sys.exit(0 if ok else 1)

    # No recognized command: run full suite (--quick / --with-examples)
    ok = run_all_tests(quick=quick, include_examples=include_examples)
    sys.exit(0 if ok else 1)
