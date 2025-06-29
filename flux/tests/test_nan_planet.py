#!/usr/bin/env python3
"""Debug NaN issue in planet scenario."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from materials import MaterialType

def test_planet_nan_debug():
    """Debug NaN issue in planet scenario."""
    print("=== Planet NaN Debug Test ===")
    
    # Create planet scenario
    sim = FluxSimulation(nx=64, ny=64, dx=50.0, scenario="planet")
    
    print(f"\nInitial state:")
    print(f"  Density range: {sim.state.density.min():.3f} - {sim.state.density.max():.1f} kg/m³")
    print(f"  Temperature range: {sim.state.temperature.min():.1f} - {sim.state.temperature.max():.1f} K")
    print(f"  Any NaN in density: {np.any(np.isnan(sim.state.density))}")
    print(f"  Any NaN in temperature: {np.any(np.isnan(sim.state.temperature))}")
    print(f"  Any NaN in velocity_x: {np.any(np.isnan(sim.state.velocity_x))}")
    print(f"  Any NaN in velocity_y: {np.any(np.isnan(sim.state.velocity_y))}")
    
    # Check if initialization produces extreme values
    print(f"\nAfter initialization:")
    print(f"  Max |velocity_x|: {np.max(np.abs(sim.state.velocity_x)):.3f} m/s")
    print(f"  Max |velocity_y|: {np.max(np.abs(sim.state.velocity_y)):.3f} m/s")
    print(f"  Max pressure: {np.max(np.abs(sim.state.pressure)):.1e} Pa")
    
    # Run one step
    print("\nRunning one step...")
    sim.paused = False
    
    try:
        sim.step_forward()
        print("Step completed successfully")
        print(f"  Max |velocity_x|: {np.max(np.abs(sim.state.velocity_x)):.3f} m/s")
        print(f"  Max |velocity_y|: {np.max(np.abs(sim.state.velocity_y)):.3f} m/s")
        
        if np.any(np.isnan(sim.state.velocity_x)) or np.any(np.isnan(sim.state.velocity_y)):
            print("ERROR: NaN detected after first step!")
            
            # Find where NaN first appears
            nan_x = np.isnan(sim.state.velocity_x)
            nan_y = np.isnan(sim.state.velocity_y)
            
            if np.any(nan_x):
                j, i = np.where(nan_x)
                print(f"  First NaN in velocity_x at ({i[0]}, {j[0]})")
                print(f"  Density there: {sim.state.density[j[0], i[0]]}")
                print(f"  Temperature there: {sim.state.temperature[j[0], i[0]]}")
                
    except Exception as e:
        print(f"ERROR during step: {e}")
        import traceback
        traceback.print_exc()
        
    return True


if __name__ == "__main__":
    test_planet_nan_debug()