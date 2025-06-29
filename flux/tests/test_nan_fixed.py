#!/usr/bin/env python3
"""Test if NaN issue is fixed."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation

def test_planet_startup():
    """Test planet scenario startup."""
    print("=== Testing Planet Scenario Startup ===")
    
    # Create simulation
    sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario="planet")
    
    print(f"After initialization:")
    print(f"  Density range: {sim.state.density.min():.3f} - {sim.state.density.max():.1f} kg/m³")
    print(f"  Any zero density: {np.any(sim.state.density == 0)}")
    print(f"  Any NaN density: {np.any(np.isnan(sim.state.density))}")
    print(f"  Any NaN velocity_x: {np.any(np.isnan(sim.state.velocity_x))}")
    print(f"  Any NaN velocity_y: {np.any(np.isnan(sim.state.velocity_y))}")
    
    # Try a few steps
    print("\nRunning 5 steps...")
    sim.paused = False
    for i in range(5):
        sim.step_forward()
        if np.any(np.isnan(sim.state.velocity_x)) or np.any(np.isnan(sim.state.velocity_y)):
            print(f"  Step {i}: NaN detected!")
            return False
        else:
            print(f"  Step {i}: OK (dt={sim.state.dt:.3f}s)")
    
    print("\nSuccess! No NaN values detected.")
    return True

if __name__ == "__main__":
    test_planet_startup()