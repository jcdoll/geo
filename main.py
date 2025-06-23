#!/usr/bin/env python3
"""
Flux-based geological simulation - Main entry point.

This provides an interactive simulation with various preset scenarios
and real-time parameter adjustment.
"""

import argparse
import sys
from typing import Optional

from simulation import FluxSimulation
from visualizer import FluxVisualizer
from scenarios import get_scenario_names


class FluxSimulationApp:
    """Main application for flux-based geological simulation."""
    
    def __init__(self, nx: int = 128, ny: int = 96, dx: float = 50.0, scenario: Optional[str] = None):
        """Initialize the simulation app."""
        self.sim = FluxSimulation(nx=nx, ny=ny, dx=dx, scenario=scenario)
        self.viz = FluxVisualizer(simulation=self.sim)
        
    def run(self):
        """Run the simulation."""
        print(f"\nGrid: {self.sim.state.nx}x{self.sim.state.ny} cells")
        print(f"Domain: {self.sim.state.nx * self.sim.state.dx / 1000:.1f} x {self.sim.state.ny * self.sim.state.dx / 1000:.1f} km")
        print("(Timestep will be computed dynamically)")
        
        # Run visualizer
        self.viz.run()


def main():
    """Main entry point."""
    # Get available scenarios
    scenarios = get_scenario_names()
    
    parser = argparse.ArgumentParser(
        description="Flux-based geological simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  planet    - Circular planet with atmosphere and water pools
  layered   - Earth-like planet with flat layers (atmosphere, ocean, crust)
  volcanic  - Volcanic island with magma chamber
  ice       - Ice world with subsurface ocean
  empty     - Empty space for sandbox mode

Examples:
  python main.py                    # Default planet
  python main.py --scenario volcanic --size 150
  python main.py --scenario empty --size 200 --scale 100
        """
    )
    
    parser.add_argument(
        '--scenario', '-s',
        choices=scenarios,
        default='planet',
        help='Initial scenario to load (default: planet)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=128,
        help='Grid size (square, default: 128)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=50.0,
        help='Cell size in meters (default: 50.0)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        help='Grid width (overrides --size)'
    )
    
    parser.add_argument(
        '--height', 
        type=int,
        help='Grid height (overrides --size)'
    )
    
    args = parser.parse_args()
    
    # Determine grid dimensions
    if args.width and args.height:
        nx, ny = args.width, args.height
    else:
        nx = ny = args.size
        
    # Create and run app
    print(f"Loading scenario: {args.scenario}")
    app = FluxSimulationApp(nx=nx, ny=ny, dx=args.scale, scenario=args.scenario)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nSimulation terminated by user")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())