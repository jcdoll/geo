#!/usr/bin/env python3
"""Profile simulation performance to find bottlenecks."""

import time
from simulation import FluxSimulation

def main():
    print("Creating simulation...")
    sim = FluxSimulation(nx=128, ny=128, dx=50.0, scenario='planet')
    
    print("\nInitial state check:")
    print(f"Temperature range: {sim.state.temperature.min():.1f} - {sim.state.temperature.max():.1f} K")
    print(f"Density range: {sim.state.density.min():.1f} - {sim.state.density.max():.1f} kg/m3")
    
    # Run one step with detailed timing
    print("\nRunning one timestep with profiling...")
    
    dt = sim.physics.apply_cfl_limit()
    print(f"Timestep dt = {dt:.3f}s")
    
    # Manually time each component
    start = time.perf_counter()
    
    t0 = time.perf_counter()
    gx, gy = sim.gravity_solver.solve_gravity()
    gravity_time = time.perf_counter() - t0
    print(f"Gravity solve: {gravity_time:.3f}s")
    
    t0 = time.perf_counter()
    sim.physics.update_momentum(gx, gy, dt)
    momentum_time = time.perf_counter() - t0
    print(f"Momentum + projection: {momentum_time:.3f}s")
    
    t0 = time.perf_counter()
    sim.transport.advect_materials_vectorized(dt)
    advection_time = time.perf_counter() - t0
    print(f"Advection: {advection_time:.3f}s")
    
    t0 = time.perf_counter()
    sim.heat_transfer.solve_heat_equation(dt)
    heat_time = time.perf_counter() - t0
    print(f"Heat transfer: {heat_time:.3f}s")
    
    total_time = time.perf_counter() - start
    print(f"\nTotal timestep: {total_time:.3f}s")
    print(f"Expected FPS: {1.0/total_time:.2f}")
    
    # Check for NaN
    print(f"\nAfter step - Temperature has NaN: {np.any(np.isnan(sim.state.temperature))}")

if __name__ == "__main__":
    import numpy as np
    main()