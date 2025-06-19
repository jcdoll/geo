#!/usr/bin/env python3
"""
Run test scenarios with visualization.

This script allows you to run any test scenario with real-time visualization
to see what's happening during the test.

Usage:
    python tests/run_visual_tests.py [scenario_name] [options]

Examples:
    python tests/run_visual_tests.py magma_small
    python tests/run_visual_tests.py water_conservation --steps 500
    python tests/run_visual_tests.py --list
"""

import sys
import argparse
from typing import Dict, Type, Callable

# Add parent directory to path
sys.path.insert(0, '.')

from tests.test_framework import TestScenario, ScenarioRunner
from tests.test_visualizer import TestScenarioVisualizer

# Import scenario registries from consolidated test files
from tests.test_magma import SCENARIOS as MAGMA_SCENARIOS
from tests.test_water import SCENARIOS as WATER_SCENARIOS
from tests.test_gravity_buoyancy import SCENARIOS as GRAVITY_BUOYANCY_SCENARIOS

# Combine all scenarios
SCENARIOS: Dict[str, Callable[[], TestScenario]] = {}
SCENARIOS.update(MAGMA_SCENARIOS)
SCENARIOS.update(WATER_SCENARIOS)
SCENARIOS.update(GRAVITY_BUOYANCY_SCENARIOS)


def list_scenarios():
    """Print all available scenarios."""
    print("\nAvailable test scenarios:")
    print("=" * 60)
    
    # Group by category
    categories = {
        'Magma': [k for k in SCENARIOS if k.startswith('magma') or k.startswith('granite')],
        'Water': [k for k in SCENARIOS if k.startswith('water')],
        'Gravity & Buoyancy': [k for k in SCENARIOS if k.startswith('gravity') or k.startswith('buoyancy') or k.startswith('rock_on')],
    }
    
    for category, names in categories.items():
        if names:
            print(f"\n{category}:")
            for name in sorted(names):
                # Create instance to get description
                instance = SCENARIOS[name]()
                print(f"  {name:<20} - {instance.get_description()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run test scenarios with visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'scenario',
        nargs='?',
        choices=list(SCENARIOS.keys()),
        help='Name of the scenario to run'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available scenarios'
    )
    
    parser.add_argument(
        '--steps', '-s',
        type=int,
        default=200,
        help='Maximum number of steps to run (default: 200)'
    )
    
    parser.add_argument(
        '--size', '-z',
        type=int,
        default=60,
        help='Simulation grid size (default: 60)'
    )
    
    parser.add_argument(
        '--scale', '-c',
        type=int,
        default=10,
        help='Display scale factor (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_scenarios()
        return 0
        
    # Require scenario if not listing
    if not args.scenario:
        parser.error("Please specify a scenario name or use --list")
        
    # Create scenario instance
    scenario = SCENARIOS[args.scenario]()
    
    # Create runner
    runner = ScenarioRunner(scenario, sim_width=args.size, sim_height=args.size)
    
    # Create and run visualizer
    # Calculate window dimensions based on grid size and scale
    window_width = args.size * args.scale + 300  # Add space for test panel
    window_height = args.size * args.scale + 30   # Add space for status bar
    
    visualizer = TestScenarioVisualizer(
        runner,
        max_steps=args.steps,
        window_width=window_width,
        window_height=window_height
    )
    
    print(f"\n{'='*60}")
    print(f"TEST: {scenario.get_name()}")
    print(f"{'='*60}")
    print(f"Description: {scenario.get_description()}")
    print(f"Grid: {args.size}x{args.size}, Max steps: {args.steps}")
    print(f"{'='*60}\n")
    
    # Run the visualizer
    visualizer.run()
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"Test complete: {visualizer.scenario_runner.step_count} steps")
    final_eval = visualizer.scenario_runner.scenario.evaluate(visualizer.simulation)
    if final_eval.get('success'):
        print("✓ TEST PASSED")
    else:
        print("✗ TEST FAILED")
    print(f"Final: {final_eval.get('message', 'N/A')}")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 