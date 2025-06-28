#!/usr/bin/env python3
"""Test Low-Mach preconditioned pressure solver."""

import numpy as np
from simulation import FluxSimulation

def test_simulation(size, scenario="planet", solver_type="lowmach"):
    """Test simulation with given size and solver."""
    print(f"\n=== Testing {scenario} scenario at {size}x{size} (solver={solver_type}) ===")
    
    try:
        # Create simulation
        sim = FluxSimulation(nx=size, ny=size, dx=50.0, scenario=scenario)
        
        # Configure physics to use specified solver
        sim.physics.solver_type = solver_type
        sim.physics.pressure_solver = None  # Force recreation
        
        # Create pressure solver and enable monitoring
        _ = sim.physics.update_momentum(np.zeros_like(sim.state.density), 
                                      np.zeros_like(sim.state.density), 0.001)
        if hasattr(sim.physics.pressure_solver, 'enable_monitoring'):
            sim.physics.pressure_solver.enable_monitoring = True
        
        # Check initial state
        print(f"Density range: {np.min(sim.state.density):.2e} to {np.max(sim.state.density):.2e}")
        density_ratio = np.max(sim.state.density) / np.min(sim.state.density[sim.state.density > 0])
        print(f"Density ratio: {density_ratio:.2e}")
        
        # Check beta values
        print(f"Beta_x range: {np.min(sim.state.beta_x):.2e} to {np.max(sim.state.beta_x):.2e}")
        
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
            max_steps = 10
            stable_steps = 0
            
            for i in range(max_steps):
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
                    break
                    
                # Check if velocity is growing too fast
                if max_vel_after > 1e4:  # 10 km/s
                    print(f"  Velocity exceeded 10 km/s - stopping")
                    break
                    
                stable_steps = i + 1
                    
            print(f"Completed {stable_steps} stable steps")
            if stable_steps == max_steps:
                print(f"Success! Stable after {max_steps} steps")
                print(f"  Final max velocity: {max_vel_after:.2e} m/s")
                print(f"  Final max pressure: {max_p_after:.2e} Pa")
                print(f"  FPS estimate: {sim.fps:.1f}")
                
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

# Test Low-Mach solver on different grid sizes
print("Testing LOW-MACH PRECONDITIONED solver:")
for size in [16, 24, 32, 48, 64]:
    test_simulation(size, solver_type="lowmach")

# Compare with all-speed method
print("\n\nComparing with ALL-SPEED method:")
test_simulation(32, solver_type="allspeed")

# Compare with standard method
print("\n\nComparing with STANDARD method:")
test_simulation(32, solver_type="standard")

print("\nTests completed!")