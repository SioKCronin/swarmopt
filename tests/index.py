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
from pathlib import Path

# Get the tests directory
TESTS_DIR = Path(__file__).parent

# Test organization
TEST_CATEGORIES = {
    'unit': {
        'description': 'Unit tests (pytest-based, fast)',
        'command': 'pytest unit/ -v',
        'time': '~10s'
    },
    'integration': {
        'description': 'Integration tests (comprehensive feature tests)',
        'scripts': {
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
            'test_delegate_positioning.py': 'Test delegate positioning (~2min)',
        }
    },
    'examples': {
        'description': 'Examples and demonstrations',
        'scripts': {
            'example.py': 'Comprehensive SwarmOpt demo (~1min)',
            'multiobjective_example.py': 'Multiobjective optimization demo (~2min)',
            'cancer_tda_example.py': 'Cancer TDA guided PSO demo (~3min)',
            'test_installation.py': 'Installation verification (~30s)',
            'test_respect_simple.py': 'Simple respect boundary test (~30s)',
        }
    }
}

def show_index():
    """Display all available tests"""
    print("=" * 80)
    print("🧪 SwarmOpt Test Suite")
    print("=" * 80)
    print()
    
    # Unit tests
    print("📋 UNIT TESTS")
    print("-" * 80)
    print(f"  {TEST_CATEGORIES['unit']['description']}")
    print(f"  Command: {TEST_CATEGORIES['unit']['command']}")
    print(f"  Time: {TEST_CATEGORIES['unit']['time']}")
    print()
    
    # Integration tests
    print("🔬 INTEGRATION TESTS")
    print("-" * 80)
    print(f"  {TEST_CATEGORIES['integration']['description']}")
    for i, (script, desc) in enumerate(TEST_CATEGORIES['integration']['scripts'].items(), 1):
        print(f"  {i:2d}. {script:40s} - {desc}")
    print()
    
    # Examples
    print("📚 EXAMPLES")
    print("-" * 80)
    print(f"  {TEST_CATEGORIES['examples']['description']}")
    for i, (script, desc) in enumerate(TEST_CATEGORIES['examples']['scripts'].items(), 1):
        print(f"  {i:2d}. {script:40s} - {desc}")
    print()

def run_unit_tests():
    """Run all unit tests"""
    print("\n🧪 Running Unit Tests...")
    print("=" * 80)
    os.chdir(TESTS_DIR)
    result = subprocess.run(['pytest', 'unit/', '-v'], capture_output=False)
    return result.returncode == 0

def run_integration_test(script_name):
    """Run a specific integration test"""
    script_path = TESTS_DIR / 'integration' / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_name}")
        return False
    
    print(f"\n🔬 Running {script_name}...")
    print("=" * 80)
    result = subprocess.run(['python', str(script_path)], capture_output=False)
    return result.returncode == 0

def run_example(script_name):
    """Run a specific example"""
    script_path = TESTS_DIR / 'examples' / script_name
    if not script_path.exists():
        print(f"❌ Example not found: {script_name}")
        return False
    
    print(f"\n📚 Running {script_name}...")
    print("=" * 80)
    result = subprocess.run(['python', str(script_path)], capture_output=False)
    return result.returncode == 0

def run_all_tests():
    """Run all tests (unit + integration)"""
    print("\n🚀 Running Full Test Suite...")
    print("=" * 80)
    
    # Run unit tests
    print("\n1️⃣  Unit Tests")
    unit_success = run_unit_tests()
    
    # Run integration tests
    print("\n2️⃣  Integration Tests")
    integration_results = {}
    for script in TEST_CATEGORIES['integration']['scripts'].keys():
        success = run_integration_test(script)
        integration_results[script] = success
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
    print(f"\nIntegration Tests:")
    for script, success in integration_results.items():
        status = '✅' if success else '❌'
        print(f"  {status} {script}")
    
    total_passed = sum(integration_results.values()) + (1 if unit_success else 0)
    total_tests = len(integration_results) + 1
    print(f"\n🎯 Total: {total_passed}/{total_tests} passed")

def main():
    """Interactive test runner"""
    show_index()
    
    print("=" * 80)
    print("🎯 OPTIONS")
    print("=" * 80)
    print("  1. Run all unit tests (pytest)")
    print("  2. Run all integration tests")
    print("  3. Run specific integration test")
    print("  4. Run specific example")
    print("  5. Run FULL test suite (unit + integration)")
    print("  6. Show index again")
    print("  0. Exit")
    print()
    
    try:
        choice = input("Select option: ").strip()
        
        if choice == '1':
            run_unit_tests()
        elif choice == '2':
            for script in TEST_CATEGORIES['integration']['scripts'].keys():
                run_integration_test(script)
        elif choice == '3':
            show_index()
            script_name = input("\nEnter integration test name: ").strip()
            run_integration_test(script_name)
        elif choice == '4':
            show_index()
            script_name = input("\nEnter example name: ").strip()
            run_example(script_name)
        elif choice == '5':
            run_all_tests()
        elif choice == '6':
            show_index()
            main()
        elif choice == '0':
            print("👋 Goodbye!")
        else:
            print("❌ Invalid option")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except EOFError:
        print("\n\n📋 Index displayed (non-interactive mode)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == 'all':
            run_all_tests()
        elif sys.argv[1] == 'unit':
            run_unit_tests()
        elif sys.argv[1] == 'integration':
            if len(sys.argv) > 2:
                run_integration_test(sys.argv[2])
            else:
                for script in TEST_CATEGORIES['integration']['scripts'].keys():
                    run_integration_test(script)
        else:
            print(f"Usage: {sys.argv[0]} [all|unit|integration [test_name]]")
    else:
        # Interactive mode
        main()

