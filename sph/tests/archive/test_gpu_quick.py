#!/usr/bin/env python3
"""Quick test of GPU backend."""

import numpy as np
import time
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.physics import MaterialType

# Check backends
sph.print_backend_info()

# Create test particles
n_particles = 1000
particles = ParticleArrays.allocate(n_particles + 100)

# Initialize in a grid
n_side = int(np.sqrt(n_particles))
spacing = 2.0

idx = 0
for i in range(n_side):
    for j in range(n_side):
        if idx >= n_particles:
            break
        particles.position_x[idx] = (i - n_side/2) * spacing
        particles.position_y[idx] = (j - n_side/2) * spacing  
        particles.velocity_x[idx] = 0.0
        particles.velocity_y[idx] = 0.0
        particles.mass[idx] = 1.0
        particles.smoothing_h[idx] = spacing * 1.3
        particles.material_id[idx] = MaterialType.WATER.value
        particles.density[idx] = 1000.0
        particles.temperature[idx] = 300.0
        idx += 1

n_active = idx
kernel = CubicSplineKernel(dim=2)

# Test each backend
for backend in ['cpu', 'numba', 'gpu']:
    try:
        sph.set_backend(backend)
        print(f"\nTesting {backend} backend...")
        
        # Create spatial hash
        spatial_hash = sph.create_spatial_hash(
            domain_size=(100.0, 100.0),
            cell_size=spacing * 2,
            domain_min=(-50.0, -50.0)
        )
        
        # Warmup
        for _ in range(3):
            spatial_hash.update(particles, n_active)
            spatial_hash.find_neighbors(particles, n_active, search_radius=spacing * 2)
            sph.compute_density(particles, kernel, n_active)
            sph.compute_forces(particles, kernel, n_active, gravity=np.array([0, -9.81]))
        
        # Time 10 steps
        start = time.time()
        for _ in range(10):
            spatial_hash.update(particles, n_active)
            spatial_hash.find_neighbors(particles, n_active, search_radius=spacing * 2)
            sph.compute_density(particles, kernel, n_active)
            sph.compute_forces(particles, kernel, n_active, gravity=np.array([0, -9.81]))
        elapsed = time.time() - start
        
        print(f"  10 steps took: {elapsed:.3f}s")
        print(f"  Average density: {particles.density[:n_active].mean():.1f}")
        print(f"  Force magnitude: {np.sqrt(particles.force_x[:n_active]**2 + particles.force_y[:n_active]**2).mean():.1f}")
        
    except Exception as e:
        print(f"  Failed: {e}")

print("\nGPU backend is working!" if sph.get_backend() == 'gpu' else "\nGPU backend not available")