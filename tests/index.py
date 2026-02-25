#!/usr/bin/env python3
"""
SwarmOpt Test Suite Index

Unified test runner for all SwarmOpt tests:
- Unit tests (pytest-based)
- Integration tests (comprehensive feature tests)
- Examples (demonstrations and tutorials)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Get the tests directory
TESTS_DIR = Path(__file__).parent

# Optional short descriptions for discovered scripts (script_name -> description)
INTEGRATION_DESCRIPTIONS = {
    'test_installation.py': 'Verify installation and basic functionality (~30s)',
    'test_inertia_variations.py': 'Test inertia weight strategies (~2min)',
    'test_velocity_clamping.py': 'Test velocity clamping variations (~3min)',
    'test_cpso.py': 'Test Cooperative PSO (~3min)',
    'test_ppso.py': 'Test Proactive PSO (~3min)',
    'test_diversity_system.py': 'Test diversity monitoring (~4min)',
    'test_variation_operators.py': 'Test variation operators (~3min)',
    'test_multiobjective.py': 'Test multiobjective PSO (~5min)',
    'test_multiobjective_challenges.py': 'Stress test on hard benchmarks (~15min)',
    'test_respect_boundary.py': 'Test respect boundary feature (~2min)',
    'test_respect_simple.py': 'Simple respect boundary test (~30s)',
}
EXAMPLE_DESCRIPTIONS = {
    'example.py': 'Comprehensive SwarmOpt demo (~1min)',
    'multiobjective_example.py': 'Multiobjective optimization demo (~2min)',
    'cancer_tda_example.py': 'Cancer TDA guided PSO demo (~3min)',
}

# Scripts to skip in "quick" mode (long-running)
QUICK_SKIP_INTEGRATION = {'test_multiobjective_challenges.py'}


def discover_integration_scripts():
    """Discover integration test scripts from tests/integration/."""
    integration_dir = TESTS_DIR / 'integration'
    if not integration_dir.is_dir():
        return {}
    scripts = {}
    for p in sorted(integration_dir.glob('test_*.py')):
        name = p.name
        scripts[name] = INTEGRATION_DESCRIPTIONS.get(
            name, name.replace('_', ' ').replace('.py', '')
        )
    return scripts


def discover_example_scripts():
    """Discover example scripts from tests/examples/ (exclude index.py)."""
    examples_dir = TESTS_DIR / 'examples'
    if not examples_dir.is_dir():
        return {}
    scripts = {}
    for p in sorted(examples_dir.glob('*.py')):
        if p.name == 'index.py':
            continue
        name = p.name
        scripts[name] = EXAMPLE_DESCRIPTIONS.get(
            name, name.replace('_', ' ').replace('.py', '')
        )
    return scripts


def get_test_categories(quick=False):
    """Build test categories (with optional quick-mode filter)."""
    integration = discover_integration_scripts()
    if quick:
        integration = {k: v for k, v in integration.items() if k not in QUICK_SKIP_INTEGRATION}
    return {
        'unit': {
            'description': 'Unit tests (pytest-based, fast)',
            'command': 'pytest unit/ -v',
            'time': '~10s',
        },
        'integration': {
            'description': 'Integration tests (comprehensive feature tests)',
            'scripts': integration,
        },
        'examples': {
            'description': 'Examples and demonstrations',
            'scripts': discover_example_scripts(),
        },
    }

def show_index(quick=False):
    """Display all available tests. If quick=True, exclude long-running integration tests."""
    cats = get_test_categories(quick=quick)
    print("=" * 80)
    print("🧪 SwarmOpt Test Suite" + (" (quick mode)" if quick else ""))
    print("=" * 80)
    print()

    # Unit tests
    print("📋 UNIT TESTS")
    print("-" * 80)
    print(f"  {cats['unit']['description']}")
    print(f"  Command: {cats['unit']['command']}")
    print(f"  Time: {cats['unit']['time']}")
    print()

    # Integration tests
    print("🔬 INTEGRATION TESTS")
    print("-" * 80)
    print(f"  {cats['integration']['description']}")
    for i, (script, desc) in enumerate(cats['integration']['scripts'].items(), 1):
        print(f"  {i:2d}. {script:40s} - {desc}")
    print()

    # Examples
    print("📚 EXAMPLES")
    print("-" * 80)
    print(f"  {cats['examples']['description']}")
    for i, (script, desc) in enumerate(cats['examples']['scripts'].items(), 1):
        print(f"  {i:2d}. {script:40s} - {desc}")
    print()

def run_unit_tests(verbose=True, with_timing=True):
    """Run all unit tests. Returns (success: bool, elapsed_sec: float)."""
    if verbose:
        print("\n🧪 Running Unit Tests...")
        print("=" * 80)
    os.chdir(TESTS_DIR)
    t0 = time.perf_counter()
    result = subprocess.run(['pytest', 'unit/', '-v'], capture_output=False)
    elapsed = time.perf_counter() - t0
    success = result.returncode == 0
    if verbose and with_timing:
        print(f"\n  ⏱  Unit tests: {elapsed:.1f}s")
    return success, elapsed

def run_integration_test(script_name, verbose=True, with_timing=True, quick=False):
    """Run a specific integration test. Returns (success: bool, elapsed_sec: float)."""
    if quick and script_name in QUICK_SKIP_INTEGRATION:
        if verbose:
            print(f"\n⏭  Skipping (quick mode): {script_name}")
        return True, 0.0
    script_path = TESTS_DIR / 'integration' / script_name
    if not script_path.exists():
        if verbose:
            print(f"❌ Script not found: {script_name}")
        return False, 0.0

    if verbose:
        print(f"\n🔬 Running {script_name}...")
        print("=" * 80)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    elapsed = time.perf_counter() - t0
    success = result.returncode == 0
    if verbose and with_timing:
        print(f"\n  ⏱  {script_name}: {elapsed:.1f}s")
    return success, elapsed

def run_example(script_name, verbose=True, with_timing=True):
    """Run a specific example. Returns (success: bool, elapsed_sec: float)."""
    script_path = TESTS_DIR / 'examples' / script_name
    if not script_path.exists():
        if verbose:
            print(f"❌ Example not found: {script_name}")
        return False, 0.0

    if verbose:
        print(f"\n📚 Running {script_name}...")
        print("=" * 80)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    elapsed = time.perf_counter() - t0
    success = result.returncode == 0
    if verbose and with_timing:
        print(f"\n  ⏱  {script_name}: {elapsed:.1f}s")
    return success, elapsed

def run_all_tests(quick=False, include_examples=False):
    """
    Run full test suite: unit + integration (optionally + examples).
    Returns True if all passed, False otherwise.
    quick: skip long-running integration tests (e.g. test_multiobjective_challenges).
    """
    cats = get_test_categories(quick=quick)
    print("\n🚀 Running Full Test Suite" + (" (quick)" if quick else "") + "...")
    print("=" * 80)

    # Unit tests
    print("\n1️⃣  Unit Tests")
    unit_ok, unit_time = run_unit_tests(verbose=True, with_timing=True)

    # Integration tests
    print("\n2️⃣  Integration Tests")
    integration_results = {}
    integration_times = {}
    for script in cats['integration']['scripts'].keys():
        ok, elapsed = run_integration_test(
            script, verbose=True, with_timing=True, quick=quick
        )
        integration_results[script] = ok
        integration_times[script] = elapsed

    # Optional: examples
    example_results = {}
    example_times = {}
    if include_examples and cats['examples']['scripts']:
        print("\n3️⃣  Examples")
        for script in cats['examples']['scripts'].keys():
            ok, elapsed = run_example(script, verbose=True, with_timing=True)
            example_results[script] = ok
            example_times[script] = elapsed

    # Summary
    total_elapsed = unit_time + sum(integration_times.values()) + sum(example_times.values())
    print("\n" + "=" * 80)
    print("📊 TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Unit Tests: {'✅ PASSED' if unit_ok else '❌ FAILED'} ({unit_time:.1f}s)")
    print("\nIntegration Tests:")
    for script, success in integration_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {script} ({integration_times.get(script, 0):.1f}s)")
    if example_results:
        print("\nExamples:")
        for script, success in example_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {script} ({example_times.get(script, 0):.1f}s)")
    total_passed = (
        (1 if unit_ok else 0)
        + sum(integration_results.values())
        + sum(example_results.values())
    )
    total_tests = 1 + len(integration_results) + len(example_results)
    print(f"\n🎯 Total: {total_passed}/{total_tests} passed in {total_elapsed:.1f}s")
    return total_passed == total_tests

def main():
    """Interactive test runner"""
    show_index()

    cats = get_test_categories()
    print("=" * 80)
    print("🎯 OPTIONS")
    print("=" * 80)
    print("  1. Run all unit tests (pytest)")
    print("  2. Run all integration tests")
    print("  3. Run specific integration test")
    print("  4. Run specific example")
    print("  5. Run FULL test suite (unit + integration)")
    print("  6. Run full suite + examples")
    print("  7. Run quick suite (skip long integration tests)")
    print("  8. Show index again")
    print("  0. Exit")
    print()

    try:
        choice = input("Select option: ").strip()

        if choice == "1":
            run_unit_tests()
        elif choice == "2":
            for script in cats["integration"]["scripts"].keys():
                run_integration_test(script)
        elif choice == "3":
            show_index()
            script_name = input("\nEnter integration test name: ").strip()
            run_integration_test(script_name)
        elif choice == "4":
            show_index()
            script_name = input("\nEnter example name: ").strip()
            run_example(script_name)
        elif choice == "5":
            ok = run_all_tests(quick=False, include_examples=False)
            sys.exit(0 if ok else 1)
        elif choice == "6":
            ok = run_all_tests(quick=False, include_examples=True)
            sys.exit(0 if ok else 1)
        elif choice == "7":
            ok = run_all_tests(quick=True, include_examples=False)
            sys.exit(0 if ok else 1)
        elif choice == "8":
            show_index()
            main()
        elif choice == "0":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid option")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(130)
    except EOFError:
        print("\n\n📋 Index displayed (non-interactive mode)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        arg = sys.argv[1].lower()
        quick = "--quick" in sys.argv or "-q" in sys.argv
        if arg == "all":
            ok = run_all_tests(quick=quick, include_examples="--examples" in sys.argv)
            sys.exit(0 if ok else 1)
        elif arg == "unit":
            ok, _ = run_unit_tests()
            sys.exit(0 if ok else 1)
        elif arg == "integration":
            if len(sys.argv) > 2 and not sys.argv[2].startswith("-"):
                ok, _ = run_integration_test(sys.argv[2])
                sys.exit(0 if ok else 1)
            cats = get_test_categories(quick=quick)
            all_ok = True
            for script in cats["integration"]["scripts"].keys():
                ok, _ = run_integration_test(script, quick=quick)
                all_ok = all_ok and ok
            sys.exit(0 if all_ok else 1)
        elif arg == "example" or arg == "examples":
            name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else None
            if name:
                ok, _ = run_example(name)
                sys.exit(0 if ok else 1)
            cats = get_test_categories()
            all_ok = True
            for script in cats["examples"]["scripts"].keys():
                ok, _ = run_example(script)
                all_ok = all_ok and ok
            sys.exit(0 if all_ok else 1)
        elif arg in ("--help", "-h"):
            print("Usage: python index.py [all|unit|integration|examples] [name] [--quick] [--examples]")
            print("  all              Run unit + integration (exit 0/1)")
            print("  unit             Run pytest unit tests")
            print("  integration [n]  Run integration test(s)")
            print("  examples [n]     Run example(s)")
            print("  --quick, -q      Skip long-running tests (with all/integration)")
            print("  --examples       Include examples in 'all'")
        else:
            print(f"Usage: {sys.argv[0]} [all|unit|integration|examples] [name] [--quick] [--examples]")
    else:
        # Interactive mode
        main()

