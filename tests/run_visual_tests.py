#!/usr/bin/env python3
"""Visual test runner for flux-based scenarios.

This script allows running test scenarios with live visualization,
useful for debugging physics issues and understanding test behavior.

Usage:
    python tests/run_visual_tests.py --list
    python tests/run_visual_tests.py hydrostatic_equilibrium
    python tests/run_visual_tests.py water_drop_fall --size 80 --dt 0.01
"""

import argparse
import sys
import os
from typing import Dict, Any
import pygame

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from visualizer import FluxVisualizer
from tests.scenarios import SCENARIO_GROUPS, ALL_SCENARIOS


class VisualScenarioRunner:
    """Runs scenarios with visualization using flux visualizer."""
    
    def __init__(self, scenario_name: str, **override_params):
        """Initialize visual runner with a scenario."""
        # Find scenario
        if scenario_name not in ALL_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}\n" + 
                           f"Available: {', '.join(ALL_SCENARIOS.keys())}")
        
        scenario_class, default_params = ALL_SCENARIOS[scenario_name]
        
        # Extract simulation parameters
        grid_size = override_params.pop('grid_size', 100)
        cell_size = override_params.pop('cell_size', 50.0)
        dt = override_params.pop('dt', None)
        
        # Merge remaining params for scenario
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
        self.sim = FluxSimulation(
            nx=grid_size, 
            ny=grid_size,
            dx=cell_size
        )
        
        if dt is not None:
            self.sim.dt = dt
        
        # Setup scenario
        print(f"\nRunning scenario: {self.scenario.get_name()}")
        print(f"Description: {self.scenario.get_description()}")
        print(f"Parameters: {params}")
        print(f"Grid: {grid_size}x{grid_size}, cell size: {cell_size}m")
        print(f"Time step: {self.sim.dt:.4f}s")
        print("-" * 60)
        
        try:
            self.scenario.setup(self.sim)
            self.scenario.store_initial_state(self.sim)
        except Exception as e:
            print(f"Error during setup: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create visualizer with our simulation
        self.viz = FluxVisualizer(simulation=self.sim)
        
        # Apply visualization hints
        viz_hints = self.scenario.get_visualization_hints()
        if 'preferred_display_mode' in viz_hints:
            mode = viz_hints['preferred_display_mode']
            if hasattr(self.viz, 'display_mode'):
                self.viz.display_mode = mode
                
        self.step_count = 0
        self.paused = False
        
        # Store original functions
        self.original_reset = self.sim.reset if hasattr(self.sim, 'reset') else None
        self.original_step = self.sim.timestep
        
        # Replace with wrapped versions
        if self.original_reset:
            self.sim.reset = self._reset_scenario
        self.sim.timestep = self._wrapped_timestep
        
    def _wrapped_timestep(self, dt: float):
        """Wrap timestep to track steps and print evaluation."""
        # Call original step
        self.original_step(dt)
        self.step_count += 1
        
        # Print evaluation periodically
        print_interval = 50 if self.sim.dt < 0.01 else 10
        if self.step_count % print_interval == 0:
            result = self.scenario.evaluate(self.sim)
            print(f"Step {self.step_count} (t={self.sim.state.time:.2f}s): {result['message']}")
            if result.get('success'):
                print("  ✓ SUCCESS!")
                
            # Check timeout
            if self.scenario.check_timeout():
                print("  ⏱ TIMEOUT!")
        
    def _reset_scenario(self):
        """Reset to scenario initial state."""
        print("\nResetting scenario...")
        self.scenario.setup(self.sim)
        self.scenario.store_initial_state(self.sim)
        self.step_count = 0
        self.sim.state.time = 0.0
    
    def run(self):
        """Run the visualization with evaluation monitoring."""
        print("\nScenario Controls:")
        print("  R: Reset scenario")
        print("  SPACE: Pause/Resume")
        print(f"  Evaluation printed every {50 if self.sim.dt < 0.01 else 10} steps")
        print("\nUse standard visualizer controls (H for help)")
        print()
        
        # Print initial evaluation
        result = self.scenario.evaluate(self.sim)
        print(f"Initial: {result['message']}")
        
        # Custom event handling
        self.viz.custom_event_handler = self._handle_event
        
        # Run the visualizer
        try:
            self.viz.run()
        except Exception as e:
            print(f"Visualizer error: {e}")
            import traceback
            traceback.print_exc()
            
    def _handle_event(self, event):
        """Handle custom keyboard events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self._reset_scenario()
                return True
        return False  # Let visualizer handle other events


class ScenarioGroup:
    """Container for related test scenarios."""
    
    def __init__(self, name: str, description: str):
        """Initialize scenario group."""
        self.name = name
        self.description = description
        self.scenarios = {}
        
    def add_scenario(self, key: str, scenario_class: type, **default_params):
        """Add a scenario to this group."""
        self.scenarios[key] = (scenario_class, default_params)
        
    def list_scenarios(self) -> list:
        """List all scenario keys in this group."""
        return list(self.scenarios.keys())


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
            print(f"  {scenario_key:25} - {scenario.get_description()}")
            
    print("\nUsage: python tests/run_visual_tests.py <scenario_name> [options]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run flux-based test scenarios with visualization"
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
        default=100,
        help='Grid size (default: 100)'
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        help='Time step override (default: auto)'
    )
    
    parser.add_argument(
        '--cell-size',
        type=float,
        default=50.0,
        help='Cell size in meters (default: 50.0)'
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
    override_params = {
        'grid_size': args.size,
        'cell_size': args.cell_size,
    }
    
    if args.dt is not None:
        override_params['dt'] = args.dt
        
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
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    # else keep as string
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