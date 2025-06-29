#!/usr/bin/env python3
"""
Simple test to debug explosive forces in rock blob.
"""

import numpy as np
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.physics import MaterialType, MaterialDatabase
from sph.physics.density_vectorized import compute_density_vectorized
from sph.physics.forces_vectorized import compute_forces_vectorized
from sph.physics.cohesion_vectorized import compute_cohesive_forces_vectorized


def main():
    # Create minimal particle setup
    n_particles = 7  # Just a small hexagon
    particles = ParticleArrays.allocate(n_particles)
    
    # Create hexagonal arrangement
    positions = [
        (0, 0),  # Center
        (1.3, 0), (-1.3, 0),  # Left/right
        (0.65, 1.13), (-0.65, 1.13),  # Top
        (0.65, -1.13), (-0.65, -1.13),  # Bottom
    ]
    
    for i, (x, y) in enumerate(positions[:n_particles]):
        particles.position_x[i] = x
        particles.position_y[i] = y
        particles.velocity_x[i] = 0.0
        particles.velocity_y[i] = 0.0
        particles.material_id[i] = MaterialType.ROCK.value
        particles.smoothing_h[i] = 1.3
        particles.mass[i] = 2700.0 * np.pi * 1.3**2  # density * area
        particles.density[i] = 2700.0  # Initial guess
        particles.pressure[i] = 0.0
        particles.temperature[i] = 288.0
        particles.force_x[i] = 0.0
        particles.force_y[i] = 0.0
        particles.neighbor_count[i] = 0
    
    # Create spatial hash
    domain_size = (10.0, 10.0)
    domain_min = (-5.0, -5.0)
    cell_size = 2.6
    spatial_hash = sph.create_spatial_hash(domain_size, cell_size, domain_min=domain_min)
    
    # Initialize
    kernel = CubicSplineKernel(dim=2)
    material_db = MaterialDatabase()
    
    print("Initial setup:")
    print(f"  Particle positions: {list(zip(particles.position_x[:n_particles], particles.position_y[:n_particles]))}")
    print(f"  Masses: {particles.mass[:n_particles]}")
    print(f"  Initial densities: {particles.density[:n_particles]}")
    
    # Step 1: Build spatial hash
    spatial_hash.build_vectorized(particles, n_particles)
    spatial_hash.query_neighbors_vectorized(particles, n_particles, 2.6)
    
    print(f"\nNeighbor counts: {particles.neighbor_count[:n_particles]}")
    
    # Step 2: Compute density
    compute_density_vectorized(particles, kernel, n_particles)
    print(f"\nComputed densities: {particles.density[:n_particles]}")
    print(f"  Min: {particles.density[:n_particles].min()}")
    print(f"  Max: {particles.density[:n_particles].max()}")
    print(f"  Mean: {particles.density[:n_particles].mean()}")
    
    # Step 3: Compute pressure
    bulk_modulus = 50e9  # Rock
    density_ref = 2700.0
    gamma = 7.0
    
    # Tait equation
    particles.pressure[:n_particles] = bulk_modulus / gamma * (
        (particles.density[:n_particles] / density_ref)**gamma - 1.0
    )
    
    print(f"\nPressures: {particles.pressure[:n_particles]}")
    print(f"  Min: {particles.pressure[:n_particles].min():.2e}")
    print(f"  Max: {particles.pressure[:n_particles].max():.2e}")
    print(f"  Mean: {particles.pressure[:n_particles].mean():.2e}")
    
    # Step 4: Compute forces (no gravity)
    particles.force_x[:n_particles] = 0.0
    particles.force_y[:n_particles] = 0.0
    compute_forces_vectorized(particles, kernel, n_particles, gravity=None, alpha_visc=0.1)
    
    force_mag = np.sqrt(particles.force_x[:n_particles]**2 + particles.force_y[:n_particles]**2)
    print(f"\nForce magnitudes after pressure: {force_mag}")
    print(f"  Min: {force_mag.min():.2e}")
    print(f"  Max: {force_mag.max():.2e}")
    print(f"  Mean: {force_mag.mean():.2e}")
    
    # Step 5: Add cohesion
    compute_cohesive_forces_vectorized(particles, kernel, n_particles, 
                                     material_db, cutoff_factor=1.5,
                                     temperature_softening=False)
    
    force_mag_with_cohesion = np.sqrt(particles.force_x[:n_particles]**2 + 
                                     particles.force_y[:n_particles]**2)
    print(f"\nForce magnitudes with cohesion: {force_mag_with_cohesion}")
    print(f"  Min: {force_mag_with_cohesion.min():.2e}")
    print(f"  Max: {force_mag_with_cohesion.max():.2e}")
    print(f"  Mean: {force_mag_with_cohesion.mean():.2e}")
    
    # Check accelerations
    accel_mag = force_mag_with_cohesion / particles.mass[:n_particles]
    print(f"\nAcceleration magnitudes: {accel_mag}")
    print(f"  Min: {accel_mag.min():.2e}")
    print(f"  Max: {accel_mag.max():.2e}")
    print(f"  Mean: {accel_mag.mean():.2e}")


if __name__ == "__main__":
    main()