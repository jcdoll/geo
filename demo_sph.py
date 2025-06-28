#!/usr/bin/env python3
"""
Simple non-interactive demo of SPH simulation to verify it works.
"""

import numpy as np
import time
import sph
from sph import scenarios

print("SPH Demo - Non-interactive test")
print("=" * 40)

# Show backend info
sph.print_backend_info()

# Force CPU backend for compatibility
sph.set_backend('cpu')
print(f"\nUsing backend: {sph.get_backend()}")

# Create small planet
print("\nCreating planet scenario...")
particles, n_active = scenarios.create_planet_simple(
    radius=20,
    particle_spacing=2, 
    center=(50, 50)
)
print(f"Created {n_active} particles")

# Initialize physics
kernel = sph.CubicSplineKernel(dim=2)
spatial_hash = sph.create_spatial_hash((100, 100), 4.0)
material_db = sph.physics.MaterialDatabase()

# Run a few physics steps
print("\nRunning physics simulation...")
dt = 0.0001

for step in range(10):
    # Spatial hash
    spatial_hash.build_vectorized(particles, n_active)
    spatial_hash.query_neighbors_vectorized(particles, n_active, 4.0)
    
    # Density
    sph.compute_density(particles, kernel, n_active)
    
    # Pressure (Tait equation)
    bulk_modulus = material_db.get_bulk_modulus_array(particles.material_id[:n_active])
    density_ref = material_db.get_density_ref_array(particles.material_id[:n_active])
    particles.pressure[:n_active] = bulk_modulus * (
        (particles.density[:n_active] / density_ref)**7 - 1
    )
    
    # Forces
    gravity = np.array([0, -9.81])
    sph.compute_forces(particles, kernel, n_active, gravity, alpha_visc=0.1)
    
    # Integration
    from sph.core.integrator_vectorized import integrate_leapfrog_vectorized
    integrate_leapfrog_vectorized(particles, n_active, dt, (0, 100, 0, 100))
    
    if step % 3 == 0:
        avg_density = particles.density[:n_active].mean()
        avg_pressure = particles.pressure[:n_active].mean()
        max_velocity = np.sqrt(particles.velocity_x[:n_active]**2 + 
                              particles.velocity_y[:n_active]**2).max()
        print(f"  Step {step}: ρ={avg_density:.0f} kg/m³, "
              f"P={avg_pressure:.0f} Pa, v_max={max_velocity:.3f} m/s")

print("\n✓ Simulation completed successfully!")
print("\nTo run the interactive visualizer:")
print("  python main.py")
print("\nNote: The visualizer requires a display (X11 forwarding for WSL)")