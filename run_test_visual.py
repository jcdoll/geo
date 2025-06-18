#!/usr/bin/env python3
"""
Command-line tool for running test scenarios with visualization.

This tool discovers available test scenarios and allows running them
either headless or with interactive visualization.

Usage:
    python run_test_visual.py --list                      # List all available tests
    python run_test_visual.py magma_containment           # Run specific test headless
    python run_test_visual.py magma_containment --visual  # Run with visualization
    python run_test_visual.py magma_containment --steps 200 --width 100 --height 100
"""

import sys
import os
import argparse
import importlib
import inspect
from typing import Dict, Type, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_framework import TestScenario, ScenarioRunner
from tests.test_visualizer import TestScenarioVisualizer


def discover_test_scenarios() -> Dict[str, Type[TestScenario]]:
    """Discover all available test scenarios from the tests directory.
    
    Returns:
        Dictionary mapping scenario names to their classes
    """
    scenarios = {}
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Look for Python files in tests directory
    for filename in os.listdir(tests_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            
            try:
                # Import the module
                module = importlib.import_module(f'tests.{module_name}')
                
                # Look for TestScenario subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, TestScenario) and 
                        obj != TestScenario and
                        not inspect.isabstract(obj)):
                        
                        # Try to instantiate to get the name
                        try:
                            instance = obj()
                            scenario_name = instance.get_name()
                            scenarios[scenario_name] = obj
                        except Exception as e:
                            # Some scenarios might require parameters
                            # Try with common parameter patterns
                            for params in [{'scenario': 'small'}, {'size': 'small'}, {}]:
                                try:
                                    instance = obj(**params)
                                    scenario_name = instance.get_name()
                                    scenarios[scenario_name] = obj
                                    break
                                except:
                                    continue
                                    
            except ImportError as e:
                # Skip modules that can't be imported
                continue
                
    return scenarios


def list_scenarios(scenarios: Dict[str, Type[TestScenario]]):
    """Print a formatted list of available scenarios."""
    print("\nAvailable test scenarios:")
    print("-" * 60)
    
    for name in sorted(scenarios.keys()):
        scenario_class = scenarios[name]
        
        # Try to get description
        try:
            # Try different initialization patterns
            instance = None
            for params in [{}, {'scenario': 'small'}, {'size': 'small'}]:
                try:
                    instance = scenario_class(**params)
                    break
                except:
                    continue
                    
            if instance:
                description = instance.get_description()
                print(f"\n{name}")
                print(f"  {description}")
                
                # Show available parameters if any
                if hasattr(scenario_class, '__init__'):
                    sig = inspect.signature(scenario_class.__init__)
                    params = [p for p in sig.parameters.keys() if p not in ['self', 'kwargs']]
                    if params:
                        print(f"  Parameters: {', '.join(params)}")
        except:
            print(f"\n{name}")
            print(f"  (No description available)")
            
    print("\n" + "-" * 60)
    print("\nUsage: python run_test_visual.py <scenario_name> [options]")


def parse_scenario_params(args: List[str]) -> Dict[str, str]:
    """Parse scenario-specific parameters from command line.
    
    Args:
        args: List of parameter strings in format key=value
        
    Returns:
        Dictionary of parameters
    """
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to convert to appropriate type
            try:
                # Try int first
                params[key] = int(value)
            except ValueError:
                try:
                    # Try float
                    params[key] = float(value)
                except ValueError:
                    # Keep as string
                    params[key] = value
                    
    return params


def main():
    """Main entry point for the test runner."""
    # Discover available scenarios
    scenarios = discover_test_scenarios()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run test scenarios with optional visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_test_visual.py --list
  python run_test_visual.py magma_containment
  python run_test_visual.py magma_containment --visual
  python run_test_visual.py magma_containment --visual scenario=large
  python run_test_visual.py water_blob --steps 200 --width 100
        """
    )
    
    parser.add_argument('scenario', nargs='?', help='Name of the test scenario to run')
    parser.add_argument('params', nargs='*', help='Scenario parameters (key=value)')
    parser.add_argument('--list', action='store_true', help='List available scenarios')
    parser.add_argument('--visual', action='store_true', help='Run with visualization')
    parser.add_argument('--steps', type=int, default=100, help='Maximum simulation steps')
    parser.add_argument('--width', type=int, default=80, help='Simulation width')
    parser.add_argument('--height', type=int, default=80, help='Simulation height')
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_scenarios(scenarios)
        return 0
        
    # Check if scenario was provided
    if not args.scenario:
        print("Error: No scenario specified")
        list_scenarios(scenarios)
        return 1
        
    # Check if scenario exists
    if args.scenario not in scenarios:
        print(f"Error: Unknown scenario '{args.scenario}'")
        print(f"\nDid you mean one of these?")
        
        # Find similar names
        from difflib import get_close_matches
        similar = get_close_matches(args.scenario, scenarios.keys(), n=3, cutoff=0.6)
        for name in similar:
            print(f"  - {name}")
            
        print("\nUse --list to see all available scenarios")
        return 1
        
    # Parse scenario parameters
    scenario_params = parse_scenario_params(args.params)
    
    try:
        # Create scenario instance
        scenario_class = scenarios[args.scenario]
        scenario = scenario_class(**scenario_params)
        
        # Create runner
        runner = ScenarioRunner(scenario, sim_width=args.width, sim_height=args.height)
        
        if args.visual:
            # Run with visualization
            try:
                visualizer = TestScenarioVisualizer(runner, max_steps=args.steps)
                visualizer.run()
            except ImportError as e:
                print(f"Error: Cannot run visual mode - {e}")
                print("Make sure pygame is installed: pip install pygame")
                return 1
        else:
            # Run headless
            final_eval = runner.run_headless(max_steps=args.steps)
            
            # Return appropriate exit code
            return 0 if final_eval.get('success', False) else 1
            
    except Exception as e:
        print(f"Error running scenario: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 