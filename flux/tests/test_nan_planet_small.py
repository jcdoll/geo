#!/usr/bin/env python3
"""Debug NaN issue in planet scenario with smaller grid."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from materials import MaterialType

def test_planet_sizes():
    """Test planet scenario at different sizes."""
    sizes = [16, 32, 48, 64]
    
    for size in sizes:
        print(f"\n=== Testing {size}x{size} grid ===")
        
        # Create planet scenario
        sim = FluxSimulation(nx=size, ny=size, dx=50.0, scenario="planet")
        
        # Check initial state
        has_nan_density = np.any(np.isnan(sim.state.density))
        has_nan_vx = np.any(np.isnan(sim.state.velocity_x))
        has_nan_vy = np.any(np.isnan(sim.state.velocity_y))
        
        print(f"After initialization:")
        print(f"  NaN in density: {has_nan_density}")
        print(f"  NaN in velocity_x: {has_nan_vx}")
        print(f"  NaN in velocity_y: {has_nan_vy}")
        
        if has_nan_vx or has_nan_vy:
            print(f"  ERROR: {size}x{size} has NaN after initialization!")
            
            # Find location of first NaN
            if has_nan_vx:
                j, i = np.where(np.isnan(sim.state.velocity_x))
                print(f"  First NaN in vx at ({i[0]}, {j[0]})")
                print(f"  Density: {sim.state.density[j[0], i[0]]}")
                
        else:
            # Try one step
            sim.paused = False
            try:
                sim.step_forward()
                has_nan_after = np.any(np.isnan(sim.state.velocity_x)) or np.any(np.isnan(sim.state.velocity_y))
                print(f"  After 1 step: NaN = {has_nan_after}")
                
                if not has_nan_after:
                    print(f"  Max velocity: {np.max(np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2)):.1f} m/s")
                    
            except Exception as e:
                print(f"  ERROR during step: {e}")


if __name__ == "__main__":
    test_planet_sizes()