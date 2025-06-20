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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_game import GeoGame
from visualizer import GeologyVisualizer
from tests.scenarios import SCENARIO_GROUPS, ALL_SCENARIOS


class VisualScenarioRunner:
    """Runs scenarios with visualization."""
    
    def __init__(self, scenario_name: str, **override_params):
        """Initialize visual runner with a scenario."""
        # Find scenario
        if scenario_name not in ALL_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}\n" + 
                           f"Available: {', '.join(ALL_SCENARIOS.keys())}")
        
        scenario_class, default_params = ALL_SCENARIOS[scenario_name]
        params = {**default_params, **override_params}
        self.scenario = scenario_class(**params)
        
        # Create simulation
        grid_size = params.get('grid_size', 60)
        self.sim = GeoGame(
            grid_size, grid_size,
            cell_size=50.0,
            quality=1,
            log_level="INFO",
            setup_planet=False  # Scenarios set up their own environment
        )
        
        # Setup scenario
        print(f"\nRunning scenario: {self.scenario.get_name()}")
        print(f"Description: {self.scenario.get_description()}")
        print(f"Parameters: {params}")
        print("-" * 60)
        
        self.scenario.setup(self.sim)
        self.scenario.store_initial_state(self.sim)
        
        # Get visualization hints
        self.viz_hints = self.scenario.get_visualization_hints()
        
        # Create visualizer
        self.viz = GeologyVisualizer(self.sim)
        
        # Apply visualization hints
        if 'preferred_display_mode' in self.viz_hints:
            mode = self.viz_hints['preferred_display_mode']
            if mode == 'temperature':
                self.viz.display_mode = 1
            elif mode == 'pressure':
                self.viz.display_mode = 2
            # Add more modes as needed
                
        self.step_count = 0
        self.paused = True
        
    def run(self):
        """Run the visual simulation."""
        print("\nControls:")
        print("  SPACE: Play/Pause")
        print("  RIGHT: Step forward")
        print("  R: Reset scenario")
        print("  M: Cycle display modes")
        print("  ESC: Exit")
        print()
        
        running = True
        while running:
            # Handle events
            events = self.viz.handle_events()
            
            for event in events:
                if event == 'quit':
                    running = False
                elif event == 'pause':
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Playing")
                elif event == 'reset':
                    self.reset()
                elif event == 'step':
                    self.step()
                    
            # Auto-step if not paused
            if not self.paused:
                self.step()
                
            # Update display
            self.viz.render()
            
            # Show evaluation metrics
            if self.step_count % 10 == 0:
                result = self.scenario.evaluate(self.sim)
                print(f"Step {self.step_count}: {result['message']}")
                if result.get('success'):
                    print("  SUCCESS!")
                    
    def step(self):
        """Advance simulation by one step."""
        self.sim.step_forward()
        self.step_count += 1
        
    def reset(self):
        """Reset the scenario."""
        print("\nResetting scenario...")
        self.scenario.setup(self.sim)
        self.scenario.store_initial_state(self.sim)
        self.step_count = 0
        self.paused = True


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
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())