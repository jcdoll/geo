#!/usr/bin/env python3
"""
Headless version of main.py for testing without display.
Runs simulation for a few steps and reports performance.
"""

import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sph
from sph import scenarios
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.integrator_vectorized import integrate_leapfrog_vectorized
from sph.physics import MaterialDatabase, handle_phase_transitions
from sph.physics.thermal_vectorized import compute_heat_conduction_vectorized
from sph.physics.gravity_vectorized import compute_gravity_direct_batched


def main():
    parser = argparse.ArgumentParser(description="SPH Geological Simulation (Headless)")
    parser.add_argument("--scenario", default="planet", 
                       choices=["planet", "water", "thermal", "volcanic", "layered"])
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--backend", choices=["cpu", "numba", "gpu", "auto"], default="auto")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    
    args = parser.parse_args()
    
    # Set backend
    if args.backend == "auto":
        backend = None
    else:
        if not sph.set_backend(args.backend):
            print(f"Warning: Backend '{args.backend}' not available")
            backend = None
        else:
            backend = args.backend
    
    # Create scenario (same as main.py)
    print(f"Loading scenario: {args.scenario}")
    
    scenario_funcs = {
        "planet": lambda: scenarios.create_planet_simple(
            radius=args.size * 0.4,
            particle_spacing=args.size * 0.02,
            center=(args.size * 0.5, args.size * 0.5)
        ),
        "water": lambda: scenarios.create_planet_simple(
            radius=args.size * 0.2,
            particle_spacing=args.size * 0.02,
            center=(args.size * 0.5, args.size * 0.7)
        ),
        "thermal": lambda: scenarios.create_planet_simple(
            radius=args.size * 0.3,
            particle_spacing=args.size * 0.02,
            center=(args.size * 0.5, args.size * 0.5)
        ),
        "volcanic": lambda: scenarios.create_planet_earth_like(
            radius=args.size * 0.4,
            particle_spacing=args.size * 0.02,
            center=(args.size * 0.5, args.size * 0.5)
        ),
        "layered": lambda: scenarios.create_planet_earth_like(
            radius=args.size * 0.4,
            particle_spacing=args.size * 0.02,
            center=(args.size * 0.5, args.size * 0.5)
        ),
    }
    
    particles, n_active = scenario_funcs[args.scenario]()
    
    if args.particles and args.particles < n_active:
        n_active = args.particles
    
    # Auto-select backend
    if backend is None:
        backend = sph.auto_select_backend(n_active)
        print(f"Auto-selected {backend.upper()} backend for {n_active} particles")
    
    # Print info
    sph.print_backend_info()
    print(f"\nSimulation info:")
    print(f"  Particles: {n_active}")
    print(f"  Domain: {args.size}x{args.size} m")
    print(f"  Steps: {args.steps}")
    
    # Initialize physics
    kernel = CubicSplineKernel(dim=2)
    material_db = MaterialDatabase()
    spatial_hash = sph.create_spatial_hash((args.size, args.size), 4.0)
    dt = 0.001  # Can use larger timestep with stable pressure
    
    # Run simulation
    print("\nRunning simulation...")
    step_times = []
    
    for step in range(args.steps):
        t0 = time.perf_counter()
        
        # Physics step (same as visualizer)
        spatial_hash.build_vectorized(particles, n_active)
        spatial_hash.query_neighbors_vectorized(particles, n_active, 4.0)
        
        sph.compute_density(particles, kernel, n_active)
        
        # Pressure (use stable calculation)
        from sph.physics.pressure_stable import compute_pressure_stable, get_stable_bulk_modulus
        
        bulk_modulus = np.zeros(n_active, dtype=np.float32)
        for i in range(n_active):
            bulk_modulus[i] = get_stable_bulk_modulus(particles.material_id[i])
            
        density_ref = material_db.get_density_ref_array(particles.material_id[:n_active])
        particles.pressure[:n_active] = compute_pressure_stable(
            particles.density[:n_active],
            density_ref,
            bulk_modulus
        )
        
        # Forces
        gravity = np.array([0, -9.81])
        sph.compute_forces(particles, kernel, n_active, gravity, alpha_visc=0.1)
        
        # Heat transfer (optional)
        if step % 10 == 0:  # Every 10 steps
            dT_dt = compute_heat_conduction_vectorized(particles, kernel, material_db, n_active)
            particles.temperature[:n_active] += dT_dt * dt * 10
            handle_phase_transitions(particles, material_db, n_active, dt * 10)
        
        # Integration
        integrate_leapfrog_vectorized(particles, n_active, dt, (0, args.size, 0, args.size))
        
        step_time = time.perf_counter() - t0
        step_times.append(step_time)
        
        # Progress
        if (step + 1) % 20 == 0:
            avg_time = np.mean(step_times[-20:])
            fps = 1.0 / avg_time
            print(f"  Step {step+1}/{args.steps}: {avg_time*1000:.1f} ms/step ({fps:.1f} FPS)")
    
    # Summary
    print("\nSimulation complete!")
    avg_time = np.mean(step_times)
    fps = 1.0 / avg_time
    print(f"Average: {avg_time*1000:.1f} ms/step ({fps:.1f} FPS)")
    print(f"Total time: {sum(step_times):.1f} seconds")


if __name__ == "__main__":
    main()