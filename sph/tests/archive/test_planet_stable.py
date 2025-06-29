#!/usr/bin/env python3
"""
Test if the planet is now stable with fixed pressure.
"""

import numpy as np
import time
import sph
from sph import scenarios

print("Testing planet stability with fixed pressure...")

# Set backend
sph.set_backend('cpu')

# Create planet
particles, n_active = scenarios.create_planet_simple(
    radius=20, particle_spacing=2, center=(50, 50)
)
print(f"Created {n_active} particles")

# Physics setup
kernel = sph.CubicSplineKernel(dim=2)
material_db = sph.physics.MaterialDatabase()
spatial_hash = sph.create_spatial_hash((100, 100), 4.0)

# Import stable pressure
from sph.physics.pressure_stable import compute_pressure_stable, get_stable_bulk_modulus

# Run for several steps
dt = 0.001
max_velocities = []

for step in range(20):
    # Standard physics
    spatial_hash.build_vectorized(particles, n_active)
    spatial_hash.query_neighbors_vectorized(particles, n_active, 4.0)
    
    sph.compute_density(particles, kernel, n_active)
    
    # Stable pressure
    bulk_modulus = np.zeros(n_active, dtype=np.float32)
    for i in range(n_active):
        bulk_modulus[i] = get_stable_bulk_modulus(particles.material_id[i])
    
    density_ref = material_db.get_density_ref_array(particles.material_id[:n_active])
    
    particles.pressure[:n_active] = compute_pressure_stable(
        particles.density[:n_active],
        density_ref,
        bulk_modulus,
        gamma=7.0,
        max_compression=1.5,
        max_expansion=0.8
    )
    
    # Forces
    gravity = np.array([0, -9.81])
    sph.compute_forces(particles, kernel, n_active, gravity, alpha_visc=0.1)
    
    # Integration
    from sph.core.integrator_vectorized import integrate_leapfrog_vectorized
    integrate_leapfrog_vectorized(particles, n_active, dt, (0, 100, 0, 100))
    
    # Track max velocity
    vel_mag = np.sqrt(particles.velocity_x[:n_active]**2 + 
                     particles.velocity_y[:n_active]**2)
    max_vel = np.max(vel_mag)
    max_velocities.append(max_vel)
    
    if step % 5 == 0:
        avg_pressure = particles.pressure[:n_active].mean()
        print(f"Step {step}: max_vel={max_vel:.1f} m/s, avg_pressure={avg_pressure:.2e} Pa")

# Check if velocities are reasonable
final_max_vel = max_velocities[-1]
if final_max_vel < 100:
    print(f"\n✓ Planet is stable! Final max velocity: {final_max_vel:.1f} m/s")
else:
    print(f"\n⚠ Planet still unstable. Final max velocity: {final_max_vel:.1f} m/s")
    
# Check if velocities are decreasing
if max_velocities[-1] < max_velocities[0]:
    print("✓ Velocities are decreasing (system settling)")
else:
    print("⚠ Velocities are increasing!")