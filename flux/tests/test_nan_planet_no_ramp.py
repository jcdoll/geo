#!/usr/bin/env python3
"""Test planet scenario without gravity ramping."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation

def test_planet_no_ramp():
    """Test planet scenario with gravity ramping disabled."""
    
    print("=== Testing 64x64 planet without gravity ramp ===")
    
    # Create planet scenario WITHOUT gravity ramping
    sim = FluxSimulation(nx=64, ny=64, dx=50.0, scenario="planet", init_gravity_ramp=False)
    
    # Check initial state
    has_nan_density = np.any(np.isnan(sim.state.density))
    has_nan_vx = np.any(np.isnan(sim.state.velocity_x))
    has_nan_vy = np.any(np.isnan(sim.state.velocity_y))
    
    print(f"After initialization (no ramp):")
    print(f"  NaN in density: {has_nan_density}")
    print(f"  NaN in velocity_x: {has_nan_vx}")
    print(f"  NaN in velocity_y: {has_nan_vy}")
    
    if not (has_nan_vx or has_nan_vy):
        # Try one step
        sim.paused = False
        try:
            sim.step_forward()
            has_nan_after = np.any(np.isnan(sim.state.velocity_x)) or np.any(np.isnan(sim.state.velocity_y))
            print(f"  After 1 step: NaN = {has_nan_after}")
            
            if not has_nan_after:
                max_vel = np.max(np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2))
                print(f"  Max velocity: {max_vel:.1f} m/s")
                
                # Run more steps
                for i in range(9):
                    sim.step_forward()
                    
                has_nan_10 = np.any(np.isnan(sim.state.velocity_x)) or np.any(np.isnan(sim.state.velocity_y))
                max_vel_10 = np.max(np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2))
                print(f"  After 10 steps: NaN = {has_nan_10}, max velocity = {max_vel_10:.1f} m/s")
                
        except Exception as e:
            print(f"  ERROR during step: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_planet_no_ramp()