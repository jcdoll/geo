#!/usr/bin/env python3
"""Simple test of Low-Mach solver."""

import numpy as np
from simulation import FluxSimulation

# Test small grid
print("Testing 16x16 grid with Low-Mach solver:")
sim = FluxSimulation(nx=16, ny=16, dx=50.0, scenario="planet")
sim.physics.solver_type = "lowmach"

# Enable monitoring
sim.physics.pressure_solver = None
sim.physics.update_momentum(np.zeros_like(sim.state.density), 
                           np.zeros_like(sim.state.density), 0.001)
sim.physics.pressure_solver.enable_monitoring = True

print(f"Initial density range: {np.min(sim.state.density):.2e} to {np.max(sim.state.density):.2e}")

# Take one step
sim.paused = False
try:
    print("\nDEBUG: Before step")
    print(f"Max velocity: {np.max(np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2)):.2e} m/s")
    
    sim.step_forward()
    
    print("\nStep 1 completed")
    print(f"Max velocity: {np.max(np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2)):.2e} m/s")
    print(f"Max pressure: {np.max(np.abs(sim.state.pressure)):.2e} Pa")
    
    # Check for NaN
    if np.any(np.isnan(sim.state.velocity_x)) or np.any(np.isnan(sim.state.velocity_y)):
        print("WARNING: NaN detected in velocities!")
        print(f"  NaN in vx: {np.sum(np.isnan(sim.state.velocity_x))}")
        print(f"  NaN in vy: {np.sum(np.isnan(sim.state.velocity_y))}")
    else:
        print("No NaN detected")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()