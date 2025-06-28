#!/usr/bin/env python3
"""Simple test of all-speed pressure solver."""

import numpy as np
from simulation import FluxSimulation

# Test progressively larger grids
for size in [16, 24, 32]:
    print(f"\nTesting {size}x{size} grid with all-speed method:")
    sim = FluxSimulation(nx=size, ny=size, dx=50.0, scenario="planet")
    sim.physics.use_allspeed = True

    print(f"Initial density range: {np.min(sim.state.density):.2e} to {np.max(sim.state.density):.2e}")
    print(f"Initial beta range: {np.min(sim.state.beta_x):.2e} to {np.max(sim.state.beta_x):.2e}")

    # Take one step
    sim.paused = False
    try:
        sim.step_forward()
        print("Step 1 completed successfully")
        print(f"Max velocity: {np.max(np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2)):.2e} m/s")
        print(f"Max pressure: {np.max(np.abs(sim.state.pressure)):.2e} Pa")
        
        # Check for NaN
        if np.any(np.isnan(sim.state.velocity_x)) or np.any(np.isnan(sim.state.velocity_y)):
            print("WARNING: NaN detected in velocities!")
            nan_x = np.sum(np.isnan(sim.state.velocity_x))
            nan_y = np.sum(np.isnan(sim.state.velocity_y))
            print(f"  NaN cells: vx={nan_x}, vy={nan_y}")
        else:
            print("No NaN detected")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# Now test with standard method for comparison
print("\n\nTesting 16x16 grid with standard method:")
sim2 = FluxSimulation(nx=16, ny=16, dx=50.0, scenario="planet")
sim2.physics.use_allspeed = False

try:
    sim2.step_forward()
    print("Step 1 completed successfully")
    print(f"Max velocity: {np.max(np.sqrt(sim2.state.velocity_x**2 + sim2.state.velocity_y**2)):.2e} m/s")
    print(f"Max pressure: {np.max(np.abs(sim2.state.pressure)):.2e} Pa")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()