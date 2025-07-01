#!/usr/bin/env python3
"""Visual test runner for scenarios.

This script allows running test scenarios with live visualization,
useful for debugging physics issues and understanding test behavior.

Usage:
    python tests/run_visual_tests.py --list
    python tests/run_visual_tests.py water_conservation
    python tests/run_visual_tests.py gravity_fall --size 80 --steps 200
"""

import argparse
import sys
import os
from typing import Dict, Any
import pygame

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_game import GeoGame
from visualizer import GeologyVisualizer
from tests.scenarios import SCENARIO_GROUPS, ALL_SCENARIOS


class VisualScenarioRunner:
    """Runs scenarios with visualization using standard visualizer."""
    
    def __init__(self, scenario_name: str, **override_params):
        """Initialize visual runner with a scenario."""
        # Find scenario
        if scenario_name not in ALL_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}\n" + 
                           f"Available: {', '.join(ALL_SCENARIOS.keys())}")
        
        scenario_class, default_params = ALL_SCENARIOS[scenario_name]
        
        # Extract grid_size before merging params (it's for the runner, not the scenario)
        grid_size = override_params.pop('grid_size', 60)
        
        # Now merge the remaining params
        params = {**default_params, **override_params}
        
        try:
            self.scenario = scenario_class(**params)
        except Exception as e:
            print(f"Error creating scenario: {e}")
            print(f"Scenario class: {scenario_class}")
            print(f"Parameters: {params}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create simulation
        self.sim = GeoGame(
            grid_size, grid_size,
            cell_size=1.0,
            log_level="INFO",
            setup_planet=False  # Scenarios set up their own environment
        )
        
        # Setup scenario
        print(f"\nRunning scenario: {self.scenario.get_name()}")
        print(f"Description: {self.scenario.get_description()}")
        print(f"Parameters: {params}")
        print(f"Grid size: {grid_size}x{grid_size}")
        print("-" * 60)
        
        try:
            self.scenario.setup(self.sim)
            self.scenario.store_initial_state(self.sim)
        except Exception as e:
            print(f"Error during setup: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create visualizer with our simulation - just use the standard one!
        self.viz = GeologyVisualizer(simulation=self.sim)
        
        # Apply visualization hints
        viz_hints = self.scenario.get_visualization_hints()
        if 'preferred_display_mode' in viz_hints:
            mode = viz_hints['preferred_display_mode']
            if mode in ['materials', 'temperature', 'pressure']:
                self.viz.display_mode = mode
                
        self.step_count = 0
        
        # Store original reset function and replace with scenario reset
        self.original_reset = self.viz.simulation.reset
        self.viz.simulation.reset = self._reset_scenario
        
        # Store original step_forward and wrap it
        self.original_step = self.viz.simulation.step_forward
        self.viz.simulation.step_forward = self._wrapped_step_forward
        
    def _wrapped_step_forward(self):
        """Wrap step_forward to track steps and print evaluation."""
        # Call original step
        self.original_step()
        self.step_count += 1
        
        # Print evaluation every 10 steps
        if self.step_count % 10 == 0:
            result = self.scenario.evaluate(self.sim)
            print(f"Step {self.step_count}: {result['message']}")
            if result.get('success'):
                print("  SUCCESS!")
        
    def _reset_scenario(self):
        """Reset to scenario initial state."""
        print("\nResetting scenario...")
        self.scenario.setup(self.sim)
        self.scenario.store_initial_state(self.sim)
        self.step_count = 0
    
    def run(self):
        """Run the visualization with evaluation monitoring."""
        print("\nScenario Controls:")
        print("  R: Reset scenario")
        print("  Evaluation printed every 10 steps")
        print("\nUse standard visualizer controls (H for help)")
        print()
        
        # Print initial evaluation
        result = self.scenario.evaluate(self.sim)
        print(f"Initial: {result['message']}")
        
        # Run the standard visualizer
        try:
            self.viz.print_controls()
            self.viz.run()
        except Exception as e:
            print(f"Visualizer error: {e}")
            import traceback
            traceback.print_exc()


def list_scenarios():
    """List all available scenarios."""
    print("\nAvailable Test Scenarios:")
    print("=" * 60)
    
    for group_name, group in SCENARIO_GROUPS.items():
        print(f"\n{group.name} ({group.description}):")
        print("-" * 40)
        
        for scenario_key in group.list_scenarios():
            scenario_class, params = group.scenarios[scenario_key]
            scenario = scenario_class(**params)
            print(f"  {scenario_key:20} - {scenario.get_description()}")
            
    print("\nUsage: python tests/run_visual_tests.py <scenario_name> [options]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run test scenarios with visualization"
    )
    
    parser.add_argument(
        'scenario',
        nargs='?',
        help='Name of scenario to run'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available scenarios'
    )
    
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=60,
        help='Grid size (default: 60)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        help='Maximum steps to run (default: unlimited)'
    )
    
    # Parse any additional parameters as key=value
    parser.add_argument(
        'params',
        nargs='*',
        help='Additional parameters as key=value'
    )
    
    args = parser.parse_args()
    
    if args.list or not args.scenario:
        list_scenarios()
        return
        
    # Parse additional parameters
    override_params = {'grid_size': args.size}
    for param in args.params:
        if '=' in param:
            key, value = param.split('=', 1)
            # Try to parse as number
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            override_params[key] = value
            
    # Run scenario
    try:
        runner = VisualScenarioRunner(args.scenario, **override_params)
        runner.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())