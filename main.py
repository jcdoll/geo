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
from materials import MaterialType


class FluxSimulationApp:
    """Main application for flux-based geological simulation."""
    
    def __init__(self, nx: int = 128, ny: int = 96, dx: float = 50.0):
        """Initialize the simulation app."""
        self.sim = FluxSimulation(nx=nx, ny=ny, dx=dx)
        self.viz = FluxVisualizer(simulation=self.sim)
        
    def setup_empty_world(self):
        """Create an empty world with just space."""
        self.sim.state.vol_frac.fill(0.0)
        self.sim.state.vol_frac[MaterialType.SPACE] = 1.0
        self.sim.state.temperature.fill(273.0)  # 0°C
        self.sim.state.normalize_volume_fractions()
        self.sim.state.update_mixture_properties(self.sim.material_db)
        
    def setup_planet(self):
        """Create a simple planet with atmosphere, ocean, and crust."""
        nx, ny = self.sim.state.nx, self.sim.state.ny
        
        # Clear everything
        self.sim.state.vol_frac.fill(0.0)
        
        # Define layers from bottom up
        crust_height = int(ny * 0.4)
        ocean_height = int(ny * 0.2)
        atmos_height = ny - crust_height - ocean_height
        
        # Rock crust with some uranium deposits
        self.sim.state.vol_frac[MaterialType.ROCK, -crust_height:, :] = 1.0
        
        # Add uranium veins
        for i in range(3):
            x = int((i + 1) * nx / 4)
            y = ny - int(crust_height * 0.5)
            size = 3
            self.sim.state.vol_frac[MaterialType.ROCK, y-size:y+size, x-size:x+size] = 0.0
            self.sim.state.vol_frac[MaterialType.URANIUM, y-size:y+size, x-size:x+size] = 1.0
        
        # Ocean layer
        ocean_top = ny - crust_height
        ocean_bottom = ocean_top - ocean_height
        self.sim.state.vol_frac[MaterialType.WATER, ocean_bottom:ocean_top, :] = 1.0
        
        # Atmosphere
        self.sim.state.vol_frac[MaterialType.AIR, :ocean_bottom, :] = 1.0
        
        # Temperature gradient - hot core, cool surface
        for j in range(ny):
            depth_fraction = j / ny
            # Temperature from 250K at top to 350K at bottom
            self.sim.state.temperature[j, :] = 250 + 100 * depth_fraction
            
        # Enable solar heating
        self.sim.physics.solar_constant = 1361.0  # Earth-like
        
        # Update properties
        self.sim.state.normalize_volume_fractions()
        self.sim.state.update_mixture_properties(self.sim.material_db)
        
    def setup_volcanic_island(self):
        """Create a volcanic island scenario."""
        nx, ny = self.sim.state.nx, self.sim.state.ny
        
        # Clear and fill with air
        self.sim.state.vol_frac.fill(0.0)
        self.sim.state.vol_frac[MaterialType.AIR] = 1.0
        
        # Ocean
        ocean_level = int(ny * 0.6)
        self.sim.state.vol_frac[MaterialType.AIR, ocean_level:, :] = 0.0
        self.sim.state.vol_frac[MaterialType.WATER, ocean_level:, :] = 1.0
        
        # Island - triangular shape
        island_center = nx // 2
        island_base_width = nx // 3
        island_height = int(ny * 0.5)
        
        for j in range(ny):
            if j > ny - island_height:
                # Calculate island width at this height
                height_from_base = ny - j
                width_fraction = height_from_base / island_height
                width = int(island_base_width * width_fraction)
                
                if width > 0:
                    x_start = max(0, island_center - width)
                    x_end = min(nx, island_center + width)
                    
                    # Clear water and add rock
                    self.sim.state.vol_frac[MaterialType.WATER, j, x_start:x_end] = 0.0
                    self.sim.state.vol_frac[MaterialType.ROCK, j, x_start:x_end] = 1.0
        
        # Magma chamber
        chamber_y = ny - island_height // 3
        chamber_size = 5
        self.sim.state.vol_frac[MaterialType.ROCK, 
                               chamber_y-chamber_size:chamber_y+chamber_size,
                               island_center-chamber_size:island_center+chamber_size] = 0.0
        self.sim.state.vol_frac[MaterialType.MAGMA, 
                               chamber_y-chamber_size:chamber_y+chamber_size,
                               island_center-chamber_size:island_center+chamber_size] = 1.0
        
        # Temperature - hot magma, cool elsewhere
        self.sim.state.temperature.fill(290.0)  # 17°C
        magma_mask = self.sim.state.vol_frac[MaterialType.MAGMA] > 0.5
        self.sim.state.temperature[magma_mask] = 1500.0  # Hot magma
        
        # Update properties
        self.sim.state.normalize_volume_fractions()
        self.sim.state.update_mixture_properties(self.sim.material_db)
        
    def setup_ice_world(self):
        """Create an ice world with subsurface ocean."""
        nx, ny = self.sim.state.nx, self.sim.state.ny
        
        # Clear
        self.sim.state.vol_frac.fill(0.0)
        
        # Rock core
        core_height = int(ny * 0.3)
        self.sim.state.vol_frac[MaterialType.ROCK, -core_height:, :] = 1.0
        
        # Subsurface ocean
        ocean_height = int(ny * 0.3)
        ocean_bottom = ny - core_height
        ocean_top = ocean_bottom - ocean_height
        self.sim.state.vol_frac[MaterialType.WATER, ocean_top:ocean_bottom, :] = 1.0
        
        # Ice shell
        self.sim.state.vol_frac[MaterialType.ICE, :ocean_top, :] = 1.0
        
        # Cold surface, warm interior
        for j in range(ny):
            if j < ocean_top:
                # Ice layer - very cold at surface
                self.sim.state.temperature[j, :] = 100 + 150 * (j / ocean_top)  # 100K to 250K
            elif j < ocean_bottom:
                # Ocean layer - just above freezing
                self.sim.state.temperature[j, :] = 275.0  # 2°C
            else:
                # Rock core - warmer
                self.sim.state.temperature[j, :] = 280 + 20 * ((j - ocean_bottom) / core_height)
                
        # Update properties
        self.sim.state.normalize_volume_fractions()
        self.sim.state.update_mixture_properties(self.sim.material_db)
        
    def run(self, scenario: Optional[str] = None):
        """Run the simulation with optional initial scenario."""
        # Setup initial scenario
        if scenario == 'planet':
            self.setup_planet()
            print("Created planet with atmosphere, ocean, and crust")
        elif scenario == 'volcanic':
            self.setup_volcanic_island()
            print("Created volcanic island")
        elif scenario == 'ice':
            self.setup_ice_world()
            print("Created ice world with subsurface ocean")
        elif scenario == 'empty':
            self.setup_empty_world()
            print("Created empty world")
        else:
            # Default planet
            self.setup_planet()
            print("Created default planet scenario")
            
        print(f"\nGrid: {self.sim.state.nx}x{self.sim.state.ny} cells")
        print(f"Domain: {self.sim.state.nx * self.sim.state.dx / 1000:.1f} x {self.sim.state.ny * self.sim.state.dx / 1000:.1f} km")
        print(f"Time step: {self.sim.dt:.3f} seconds")
        
        # Run visualizer
        self.viz.run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Flux-based geological simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  planet    - Earth-like planet with atmosphere, ocean, and crust
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
        choices=['planet', 'volcanic', 'ice', 'empty'],
        help='Initial scenario to load'
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
    app = FluxSimulationApp(nx=nx, ny=ny, dx=args.scale)
    
    try:
        app.run(scenario=args.scenario)
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