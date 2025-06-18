#!/usr/bin/env python3
"""
Run test scenarios with visualization.

This script allows you to run any test scenario with real-time visualization
to see what's happening during the test.

Usage:
    python tests/run_visual_tests.py [scenario_name] [options]

Examples:
    python tests/run_visual_tests.py magma_containment
    python tests/run_visual_tests.py water_conservation --steps 500
    python tests/run_visual_tests.py surface_tension --list
"""

import sys
import argparse
from typing import Dict, Type

# Add parent directory to path
sys.path.insert(0, '.')

from tests.test_framework import TestScenario, ScenarioRunner
from tests.test_visualizer import TestScenarioVisualizer

# Import all scenario modules
from tests.test_magma_containment_scenarios import (
    MagmaContainmentScenario,
    MagmaContainmentNoPhysicsScenario,
    MagmaContainmentHeatOnlyScenario,
    MagmaContainmentFluidOnlyScenario,
    MagmaContainmentMaterialOnlyScenario,
    MagmaContainmentGravityOnlyScenario,
    MagmaBindingForceScenario
)
from tests.test_water_blob_scenario import WaterBlobCondensationScenario
from tests.test_water_conservation_scenarios import (
    WaterConservationScenario,
    WaterConservationStressTestScenario,
    WaterConservationByPhaseScenario
)
from tests.test_chunk_settle_scenarios import (
    ChunkSettleScenario,
    MultiChunkSettleScenario
)
from tests.test_surface_tension_scenarios import (
    WaterLineCollapseScenario,
    WaterDropletFormationScenario
)


# Registry of available scenarios
SCENARIOS: Dict[str, Type[TestScenario]] = {
    # Magma containment
    'magma_containment': MagmaContainmentScenario,
    'magma_no_physics': MagmaContainmentNoPhysicsScenario,
    'magma_heat_only': MagmaContainmentHeatOnlyScenario,
    'magma_fluid_only': MagmaContainmentFluidOnlyScenario,
    'magma_material_only': MagmaContainmentMaterialOnlyScenario,
    'magma_gravity_only': MagmaContainmentGravityOnlyScenario,
    'magma_binding': MagmaBindingForceScenario,
    
    # Water scenarios
    'water_blob': WaterBlobCondensationScenario,
    'water_conservation': WaterConservationScenario,
    'water_stress': WaterConservationStressTestScenario,
    'water_phase': WaterConservationByPhaseScenario,
    
    # Chunk settling
    'chunk_settle': ChunkSettleScenario,
    'multi_chunk': MultiChunkSettleScenario,
    
    # Surface tension
    'water_line': WaterLineCollapseScenario,
    'water_droplet': WaterDropletFormationScenario,
}


def list_scenarios():
    """Print all available scenarios."""
    print("\nAvailable test scenarios:")
    print("=" * 60)
    
    # Group by category
    categories = {
        'Magma': [k for k in SCENARIOS if k.startswith('magma')],
        'Water': [k for k in SCENARIOS if k.startswith('water')],
        'Mechanics': [k for k in SCENARIOS if k.startswith('chunk') or k.startswith('multi')],
        'Surface Effects': [k for k in SCENARIOS if 'line' in k or 'droplet' in k],
    }
    
    for category, names in categories.items():
        if names:
            print(f"\n{category}:")
            for name in sorted(names):
                scenario_class = SCENARIOS[name]
                # Create instance to get description
                instance = scenario_class()
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
    
    # Scenario-specific options
    parser.add_argument(
        '--variant',
        choices=['small', 'large'],
        default='small',
        help='Scenario variant for scenarios that support it'
    )
    
    parser.add_argument(
        '--disabled-phase',
        choices=['fluid_dynamics', 'heat_transfer', 'material_processes', 'self_gravity'],
        help='Disable a specific physics phase (for phase testing scenarios)'
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
    scenario_class = SCENARIOS[args.scenario]
    
    # Build kwargs based on scenario type
    kwargs = {}
    
    # Handle scenario variants
    if args.variant and hasattr(scenario_class, '__init__'):
        # Check if scenario accepts variant parameter
        import inspect
        sig = inspect.signature(scenario_class.__init__)
        if 'scenario' in sig.parameters:
            kwargs['scenario'] = args.variant
            
    # Handle disabled phase for phase testing
    if args.disabled_phase and 'phase' in args.scenario:
        kwargs['disabled_phase'] = args.disabled_phase
        
    # Create scenario
    scenario = scenario_class(**kwargs)
    
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