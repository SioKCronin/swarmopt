#!/usr/bin/env python3
"""
SwarmOpt Test Runner

This script provides an easy way to run all SwarmOpt tests
from the main directory.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run all tests in the tests_scripts directory"""
    print("🚀 SwarmOpt Test Runner")
    print("=" * 30)
    
    # Change to tests_scripts directory
    tests_dir = Path(__file__).parent / "tests_scripts"
    
    if not tests_dir.exists():
        print("❌ tests_scripts directory not found!")
        return False
    
    # Run the index script
    index_script = tests_dir / "index.py"
    
    if index_script.exists():
        print("📁 Running tests from tests_scripts directory...")
        print()
        
        # Change to tests_scripts directory and run index
        os.chdir(tests_dir)
        result = subprocess.run([sys.executable, "index.py", "all"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    else:
        print("❌ index.py not found in tests_scripts directory!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
