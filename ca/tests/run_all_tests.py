#!/usr/bin/env python3
"""
Comprehensive test runner for organized test suite.

This script can run different categories of tests:
- Unit tests: Pure functionality tests
- Integration tests: Multi-system tests  
- Scenarios: Visual test scenarios
- All tests: Complete test suite

Usage:
    python tests/run_all_tests.py unit
    python tests/run_all_tests.py integration  
    python tests/run_all_tests.py scenarios --list
    python tests/run_all_tests.py scenarios buoyancy ice_floating
    python tests/run_all_tests.py all
"""

import sys
import argparse
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Callable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_unit_tests() -> bool:
    """Run all unit tests"""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    unit_test_modules = [
        'tests.unit.test_materials',
        'tests.unit.test_pressure_solver', 
        'tests.unit.test_buoyancy_physics',
    ]
    
    passed = 0
    failed = 0
    
    for module_name in unit_test_modules:
        print(f"\n🧪 Running {module_name}...")
        try:
            module = importlib.import_module(module_name)
            # Run the module's main function
            if hasattr(module, '__main__') or '__main__' in module.__dict__:
                # Execute as script
                exec(compile(open(module.__file__).read(), module.__file__, 'exec'))
            else:
                # Call main() if it exists
                if hasattr(module, 'main'):
                    module.main()
                else:
                    print(f"  ✓ Module imported successfully (no main function)")
            
            passed += 1
            print(f"  ✓ {module_name} PASSED")
            
        except Exception as e:
            failed += 1
            print(f"  ✗ {module_name} FAILED: {e}")
            traceback.print_exc()
    
    print(f"\nUnit Test Summary: {passed} passed, {failed} failed")
    return failed == 0


def run_integration_tests() -> bool:
    """Run all integration tests"""
    print("=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    integration_test_modules = [
        'tests.integration.test_simulation_lifecycle',
    ]
    
    passed = 0
    failed = 0
    
    for module_name in integration_test_modules:
        print(f"\n🔗 Running {module_name}...")
        try:
            module = importlib.import_module(module_name)
            # Execute the module
            exec(compile(open(module.__file__).read(), module.__file__, 'exec'))
            passed += 1
            print(f"  ✓ {module_name} PASSED")
            
        except Exception as e:
            failed += 1
            print(f"  ✗ {module_name} FAILED: {e}")
            traceback.print_exc()
    
    print(f"\nIntegration Test Summary: {passed} passed, {failed} failed")
    return failed == 0


def list_scenarios() -> Dict[str, Dict[str, Callable]]:
    """List all available test scenarios"""
    scenario_modules = [
        ('buoyancy', 'tests.scenarios.test_buoyancy'),
        ('fluids', 'tests.scenarios.test_fluids'),
        ('rigid_body', 'tests.scenarios.test_rigid_body'),
        ('materials', 'tests.scenarios.test_materials'),
    ]
    
    all_scenarios = {}
    
    print("=" * 60)
    print("AVAILABLE TEST SCENARIOS")
    print("=" * 60)
    
    for category, module_name in scenario_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, 'SCENARIOS'):
                scenarios = module.SCENARIOS
                all_scenarios[category] = scenarios
                
                print(f"\n{category.title()} Scenarios:")
                for name, factory in scenarios.items():
                    scenario = factory()
                    print(f"  {name:<20} - {scenario.description}")
            else:
                print(f"\n{category.title()}: No SCENARIOS found")
                
        except Exception as e:
            print(f"\n{category.title()}: Failed to load - {e}")
    
    return all_scenarios


def run_scenarios(categories: List[str] = None, scenario_names: List[str] = None) -> bool:
    """Run specific scenarios or categories"""
    all_scenarios = list_scenarios()
    
    if not categories and not scenario_names:
        print(f"\nUse --list to see available scenarios")
        return True
    
    print("\n" + "=" * 60)
    print("RUNNING SCENARIO TESTS")
    print("=" * 60)
    
    # Import scenario runner
    try:
        from tests.framework.test_framework import ScenarioRunner
    except ImportError:
        print("Error: Could not import ScenarioRunner from tests.framework.test_framework")
        return False
    
    scenarios_to_run = []
    
    # Collect scenarios to run
    if categories:
        for category in categories:
            if category in all_scenarios:
                for name, factory in all_scenarios[category].items():
                    scenarios_to_run.append((f"{category}.{name}", factory))
            else:
                print(f"Warning: Unknown category '{category}'")
    
    if scenario_names:
        for scenario_name in scenario_names:
            found = False
            for category, scenarios in all_scenarios.items():
                if scenario_name in scenarios:
                    scenarios_to_run.append((f"{category}.{scenario_name}", scenarios[scenario_name]))
                    found = True
                    break
            if not found:
                print(f"Warning: Unknown scenario '{scenario_name}'")
    
    if not scenarios_to_run:
        print("No scenarios to run")
        return True
    
    # Run scenarios
    passed = 0
    failed = 0
    
    for scenario_name, factory in scenarios_to_run:
        print(f"\n🎬 Running scenario: {scenario_name}")
        try:
            scenario = factory()
            runner = ScenarioRunner(scenario, sim_width=30, sim_height=40)
            
            # Run scenario for limited steps
            max_steps = 50
            for step in range(max_steps):
                success = runner.step()
                if not success:
                    break
            
            # Get final result
            result = runner.get_result()
            if result.get('success', True):
                passed += 1
                print(f"  ✓ {scenario_name} PASSED - {result.get('message', 'Complete')}")
            else:
                failed += 1
                print(f"  ✗ {scenario_name} FAILED - {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            failed += 1
            print(f"  ✗ {scenario_name} FAILED - Exception: {e}")
            traceback.print_exc()
    
    print(f"\nScenario Test Summary: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run organized test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'test_type',
        choices=['unit', 'integration', 'scenarios', 'all'],
        help='Type of tests to run'
    )
    
    parser.add_argument(
        'targets',
        nargs='*',
        help='Specific categories or scenario names to run (for scenarios mode)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available scenarios (scenarios mode only)'
    )
    
    args = parser.parse_args()
    
    if args.test_type == 'scenarios' and args.list:
        list_scenarios()
        return 0
    
    success = True
    
    if args.test_type == 'unit':
        success = run_unit_tests()
    elif args.test_type == 'integration':
        success = run_integration_tests()
    elif args.test_type == 'scenarios':
        success = run_scenarios(scenario_names=args.targets)
    elif args.test_type == 'all':
        print("Running complete test suite...\n")
        unit_success = run_unit_tests()
        integration_success = run_integration_tests()
        scenario_success = run_scenarios(['buoyancy', 'fluids', 'rigid_body', 'materials'])
        success = unit_success and integration_success and scenario_success
        
        print("\n" + "=" * 60)
        print("OVERALL TEST SUMMARY")
        print("=" * 60)
        print(f"Unit tests: {'PASSED' if unit_success else 'FAILED'}")
        print(f"Integration tests: {'PASSED' if integration_success else 'FAILED'}")
        print(f"Scenario tests: {'PASSED' if scenario_success else 'FAILED'}")
        print(f"Overall: {'PASSED' if success else 'FAILED'}")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())