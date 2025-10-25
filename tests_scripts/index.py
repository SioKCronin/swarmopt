#!/usr/bin/env python3
"""
Test Scripts Index

This script provides a quick overview of all available test scripts
and can run them individually or in sequence.
"""

import os
import sys
import subprocess
from pathlib import Path

# Test scripts with descriptions
TEST_SCRIPTS = {
    'test_installation.py': {
        'description': 'Verify SwarmOpt installation and basic functionality',
        'category': 'Installation',
        'time': '~30s'
    },
    'test_inertia_variations.py': {
        'description': 'Test and compare different inertia weight strategies',
        'category': 'Inertia Weights',
        'time': '~2min'
    },
    'test_velocity_clamping.py': {
        'description': 'Test all velocity clamping variations',
        'category': 'Velocity Clamping',
        'time': '~3min'
    },
    'test_cpso.py': {
        'description': 'Comprehensive Cooperative PSO testing',
        'category': 'Cooperative PSO',
        'time': '~2min'
    },
    'test_mutation_operators.py': {
        'description': 'Test mutation operators for local optima escape',
        'category': 'Mutation Operators',
        'time': '~1min'
    },
    'test_diversity_system.py': {
        'description': 'Test diversity measurement and intervention system',
        'category': 'Diversity System',
        'time': '~2min'
    },
    'test_ppso.py': {
        'description': 'Test Proactive Particle Swarm Optimization with knowledge gain',
        'category': 'Proactive PSO',
        'time': '~3min'
    },
    'example.py': {
        'description': 'Comprehensive example showcasing all SwarmOpt features',
        'category': 'Examples',
        'time': '~1min'
    }
}

def show_index():
    """Display all available test scripts"""
    print("🧪 SwarmOpt Test Scripts Index")
    print("=" * 50)
    print()
    
    categories = {}
    for script, info in TEST_SCRIPTS.items():
        category = info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((script, info))
    
    for category, scripts in categories.items():
        print(f"📁 {category}")
        print("-" * 30)
        for script, info in scripts:
            status = "✅" if os.path.exists(script) else "❌"
            print(f"  {status} {script:<35} ({info['time']})")
            print(f"     {info['description']}")
        print()

def run_script(script_name):
    """Run a specific test script"""
    if script_name not in TEST_SCRIPTS:
        print(f"❌ Unknown script: {script_name}")
        return False
    
    if not os.path.exists(script_name):
        print(f"❌ Script not found: {script_name}")
        return False
    
    print(f"🚀 Running {script_name}...")
    print(f"📝 {TEST_SCRIPTS[script_name]['description']}")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def run_all_tests():
    """Run all test scripts in sequence"""
    print("🚀 Running All SwarmOpt Tests")
    print("=" * 50)
    print()
    
    results = {}
    total_time = 0
    
    for script_name, info in TEST_SCRIPTS.items():
        if os.path.exists(script_name):
            print(f"🧪 Running {script_name}...")
            start_time = time.time()
            
            success = run_script(script_name)
            end_time = time.time()
            duration = end_time - start_time
            total_time += duration
            
            results[script_name] = {
                'success': success,
                'duration': duration
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status} ({duration:.1f}s)")
            print()
        else:
            print(f"❌ {script_name} not found")
            results[script_name] = {'success': False, 'duration': 0}
    
    # Summary
    print("📊 Test Summary")
    print("=" * 50)
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    for script_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        duration = f"{result['duration']:.1f}s"
        print(f"  {status} {script_name:<35} {duration}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    return passed == total

def main():
    """Main function"""
    import time
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'list':
            show_index()
        elif command == 'all':
            run_all_tests()
        elif command in TEST_SCRIPTS:
            run_script(command)
        else:
            print(f"❌ Unknown command: {command}")
            print("Usage: python index.py [list|all|script_name]")
    else:
        show_index()
        print("💡 Usage:")
        print("  python index.py list          - Show all available tests")
        print("  python index.py all           - Run all tests")
        print("  python index.py script_name  - Run specific test")

if __name__ == "__main__":
    main()
