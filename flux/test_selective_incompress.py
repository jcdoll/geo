#!/usr/bin/env python3
"""Test selective incompressibility implementation."""

import numpy as np
from simulation import FluxSimulation

def test_simulation(size, scenario="planet"):
    """Test simulation with given size."""
    print(f"\n=== Testing {scenario} scenario at {size}x{size} ===")
    
    try:
        # Create simulation
        sim = FluxSimulation(nx=size, ny=size, dx=50.0, scenario=scenario)
        
        # Check initial state
        print(f"Density range: {np.min(sim.state.density):.2e} to {np.max(sim.state.density):.2e}")
        density_ratio = np.max(sim.state.density) / np.min(sim.state.density[sim.state.density > 0])
        print(f"Density ratio: {density_ratio:.2e}")
        
        # Check beta values
        print(f"Beta_x range: {np.min(sim.state.beta_x):.2e} to {np.max(sim.state.beta_x):.2e}")
        print(f"Beta_y range: {np.min(sim.state.beta_y):.2e} to {np.max(sim.state.beta_y):.2e}")
        
        # Check for NaN
        has_nan = (
            np.any(np.isnan(sim.state.velocity_x)) or 
            np.any(np.isnan(sim.state.velocity_y)) or
            np.any(np.isnan(sim.state.pressure))
        )
        print(f"Initial NaN: {has_nan}")
        
        if not has_nan:
            # Run a few steps
            sim.paused = False
            for i in range(5):
                # Check max values before step
                max_vel_before = np.max(np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2))
                max_p_before = np.max(np.abs(sim.state.pressure))
                
                sim.step_forward()
                
                # Check max values after step
                max_vel_after = np.max(np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2))
                max_p_after = np.max(np.abs(sim.state.pressure))
                
                has_nan = (
                    np.any(np.isnan(sim.state.velocity_x)) or 
                    np.any(np.isnan(sim.state.velocity_y))
                )
                
                print(f"  Step {i+1}: vel {max_vel_before:.1e} → {max_vel_after:.1e}, "
                      f"p {max_p_before:.1e} → {max_p_after:.1e}")
                
                if has_nan:
                    print(f"  NaN appeared!")
                    # Check where NaN appears
                    if np.any(np.isnan(sim.state.velocity_x)):
                        print(f"    NaN in velocity_x: {np.sum(np.isnan(sim.state.velocity_x))} cells")
                    if np.any(np.isnan(sim.state.velocity_y)):
                        print(f"    NaN in velocity_y: {np.sum(np.isnan(sim.state.velocity_y))} cells")
                    break
                    
            if not has_nan:
                print(f"Success! No NaN after 5 steps")
                print(f"  Max velocity: {np.max(np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2)):.2e} m/s")
                print(f"  Max pressure: {np.max(np.abs(sim.state.pressure)):.2e} Pa")
                print(f"  FPS estimate: {sim.fps:.1f}")
                
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

# Test different grid sizes
test_simulation(32)
test_simulation(64)
test_simulation(128)

# Also test empty scenario (should work fine)
test_simulation(64, "empty")

print("\nTests completed!")