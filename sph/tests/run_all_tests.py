#!/usr/bin/env python3
"""
Run all SPH tests and benchmarks.
"""

import sys
import subprocess
import time


def run_test(test_name, module):
    """Run a single test module."""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, module],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"✓ {test_name} passed ({elapsed:.1f}s)")
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
    else:
        print(f"✗ {test_name} failed ({elapsed:.1f}s)")
        if result.stderr:
            print("\nError:")
            print(result.stderr)
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
    
    return result.returncode == 0


def main():
    """Run all tests."""
    print("SPH Test Suite")
    print("="*60)
    
    tests = [
        ("Backend Tests", "test_backends.py"),
        ("Physics Tests", "test_physics.py"),
        ("Performance Benchmarks", "test_performance.py"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module in tests:
        if run_test(name, module):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()