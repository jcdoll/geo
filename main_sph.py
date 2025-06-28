#!/usr/bin/env python3
"""
Main SPH simulation runner with stable planet configuration.

This uses all the improvements:
- Proper neighbor search with domain_min support
- Cohesive forces to hold materials together
- Repulsion forces to prevent interpenetration
- Mixed material pressure handling
- Interface forces at material boundaries
"""

import numpy as np
import pygame
import time
import sys

# Core SPH
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.integrator_vectorized import integrate_leapfrog_vectorized
from sph.physics import (
    compute_density_vectorized,
    compute_forces_vectorized,
    compute_gravity_direct_batched,
    compute_cohesive_forces,
    compute_repulsion_forces,
    compute_boundary_force,
    compute_pressure_mixed,
    MaterialDatabase, MaterialType
)
from sph.scenarios.planet import create_planet_simple
from sph.visualizer import SPHVisualizer
import sph


def run_sph_simulation():
    """Run SPH planet simulation with visualization."""
    
    print("=== SPH Planet Simulation ===")
    
    # Create planet
    radius = 2000.0  # 2 km radius
    spacing = 50.0   # 50 m particle spacing
    particles, n_active = create_planet_simple(
        radius=radius,
        particle_spacing=spacing,
        center=(0.0, 0.0)  # Centered at origin
    )
    
    print(f"\nCreated planet with {n_active} particles")
    
    # Count materials
    material_db = MaterialDatabase()
    for mat in MaterialType:
        if mat == MaterialType.SPACE:
            continue
        count = np.sum(particles.material_id[:n_active] == mat.value)
        if count > 0:
            print(f"  {mat.name}: {count} particles")
    
    # Assign proper masses based on density
    print("\nAssigning particle masses...")
    particle_volume = spacing * spacing  # 2D area
    for i in range(n_active):
        mat_type = MaterialType(particles.material_id[i])
        density_ref = material_db.get_properties(mat_type).density_ref
        particles.mass[i] = density_ref * particle_volume / 1000  # Scale down
    
    # Setup physics
    kernel = CubicSplineKernel(dim=2)
    
    # Domain and spatial hash
    domain_size = (10000.0, 10000.0)  # 10 km x 10 km
    domain_min = (-5000.0, -5000.0)
    cell_size = spacing * 2.0  # Cell size = smoothing length
    spatial_hash = sph.create_spatial_hash(domain_size, cell_size, domain_min=domain_min)
    
    # Simulation parameters
    dt = 0.001  # 1 ms timestep
    G = 6.67430e-11 * 1e6  # Scaled gravity for smaller system
    background_pressure = 101325.0  # 1 atm
    
    # Physics flags
    enable_cohesion = True
    enable_repulsion = True
    enable_self_gravity = True
    enable_boundary_forces = True
    
    # Initialize visualization
    visualizer = SPHVisualizer(
        particles, n_active,
        domain_size=domain_size,
        window_size=(800, 800),
        target_fps=30
    )
    
    # Statistics
    sim_time = 0.0
    step = 0
    t0 = time.time()
    
    print("\nStarting simulation...")
    print("Controls:")
    print("  Space: Pause/Resume")
    print("  G: Toggle gravity visualization")
    print("  M/T/P/V/D/F: Change display mode")
    print("  H: Toggle help")
    print("  ESC: Exit")
    
    # Initial computation
    search_radius = 2.0 * cell_size
    spatial_hash.build_vectorized(particles, n_active)
    spatial_hash.query_neighbors_vectorized(particles, n_active, search_radius)
    compute_density_vectorized(particles, kernel, n_active)
    
    # Main loop
    running = True
    while running:
        # Physics update
        if not visualizer.paused:
            # 1. Update spatial hash and find neighbors
            spatial_hash.build_vectorized(particles, n_active)
            spatial_hash.query_neighbors_vectorized(particles, n_active, search_radius)
            
            # 2. Compute density
            compute_density_vectorized(particles, kernel, n_active)
            
            # 3. Compute pressure
            compute_pressure_mixed(particles, material_db, n_active, background_pressure)
            
            # 4. Reset forces
            particles.force_x[:n_active] = 0.0
            particles.force_y[:n_active] = 0.0
            
            # 5. Pressure and viscosity forces
            compute_forces_vectorized(
                particles, kernel, n_active,
                gravity=None,  # Using self-gravity instead
                alpha_visc=1.0,
                beta_visc=2.0
            )
            
            # 6. Repulsion forces (prevent interpenetration)
            if enable_repulsion:
                compute_repulsion_forces(
                    particles, n_active, material_db,
                    repulsion_distance=0.6,
                    repulsion_strength=1e6
                )
            
            # 7. Boundary forces (material interfaces)
            if enable_boundary_forces:
                compute_boundary_force(
                    particles, n_active, material_db,
                    interface_strength=5e4
                )
            
            # 8. Cohesive forces
            if enable_cohesion:
                compute_cohesive_forces(
                    particles, kernel, n_active, material_db,
                    cutoff_factor=1.3,
                    temperature_softening=False
                )
            
            # 9. Self-gravity
            if enable_self_gravity:
                compute_gravity_direct_batched(
                    particles, n_active, G=G,
                    softening=cell_size * 2.0
                )
            
            # 10. Time integration
            domain_bounds = (domain_min[0], -domain_min[0], domain_min[1], -domain_min[1])
            integrate_leapfrog_vectorized(
                particles, n_active, dt,
                domain_bounds,
                damping=0.999  # Very slight damping
            )
            
            sim_time += dt
            step += 1
            
            # Statistics every second
            if step % 1000 == 0:
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed
                
                # Material statistics
                print(f"\nStep {step}, t={sim_time:.2f}s, {steps_per_sec:.1f} steps/s")
                
                for mat in [MaterialType.AIR, MaterialType.WATER, MaterialType.ROCK]:
                    mask = particles.material_id[:n_active] == mat.value
                    if np.any(mask):
                        densities = particles.density[:n_active][mask]
                        pressures = particles.pressure[:n_active][mask]
                        velocities = np.sqrt(
                            particles.velocity_x[:n_active][mask]**2 + 
                            particles.velocity_y[:n_active][mask]**2
                        )
                        print(f"  {mat.name}: ρ={np.mean(densities):.1f} kg/m³, "
                              f"P={np.mean(pressures)/1000:.0f} kPa, "
                              f"v_max={np.max(velocities):.1f} m/s")
        
        # Update visualization
        if not visualizer.update(particles, n_active, sim_time, dt):
            running = False
    
    # Cleanup
    visualizer.quit()
    
    # Final statistics
    elapsed = time.time() - t0
    print(f"\n=== Simulation Complete ===")
    print(f"Total time: {elapsed:.1f}s for {step} steps")
    print(f"Average: {step/elapsed:.1f} steps/s")
    print(f"Simulated time: {sim_time:.2f}s")


if __name__ == "__main__":
    run_sph_simulation()