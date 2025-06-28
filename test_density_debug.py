#!/usr/bin/env python3
"""Debug SPH density calculation."""

import numpy as np
import sys
sys.path.insert(0, '.')

from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.physics.density_vectorized import compute_density_vectorized
import sph

# Create simple test case with known density
particles = ParticleArrays.allocate(100)
n_active = 9

# 3x3 grid of particles
spacing = 50.0
for i in range(3):
    for j in range(3):
        idx = i * 3 + j
        particles.position_x[idx] = i * spacing - spacing
        particles.position_y[idx] = j * spacing - spacing
        particles.mass[idx] = 1.0  # 1 kg each
        particles.smoothing_h[idx] = spacing * 2.0  # h = 100

print("Particle setup:")
print(f"  {n_active} particles in 3x3 grid")
print(f"  Spacing: {spacing} m")
print(f"  Mass: 1.0 kg each")
print(f"  Smoothing length: {spacing * 2.0} m")

# Set up
kernel = CubicSplineKernel(dim=2)
domain_size = (300.0, 300.0)
domain_min = (-150.0, -150.0)
spatial_hash = sph.create_spatial_hash(domain_size, spacing * 2, domain_min=domain_min)

# Compute density
search_radius = 2.0 * spacing * 2.0
spatial_hash.build_vectorized(particles, n_active)
spatial_hash.query_neighbors_vectorized(particles, n_active, search_radius)

print(f"\nNeighbor counts: {particles.neighbor_count[:n_active]}")

# Check kernel normalization
h = spacing * 2.0
print(f"\nKernel check:")
print(f"  W(0, h={h}) = {kernel.W(0, h):.6f}")
print(f"  W(h/2, h={h}) = {kernel.W(h/2, h):.6f}")
print(f"  W(h, h={h}) = {kernel.W(h, h):.6f}")
print(f"  W(2h, h={h}) = {kernel.W(2*h, h):.6f}")

# Compute density manually for center particle
center_idx = 4  # Middle particle
print(f"\nManual density calculation for particle {center_idx}:")
density_sum = 0.0
n_neighbors = particles.neighbor_count[center_idx]
for j in range(n_neighbors):
    neighbor_id = particles.neighbor_ids[center_idx, j]
    dist = particles.neighbor_distances[center_idx, j]
    W = kernel.W(dist, particles.smoothing_h[center_idx])
    contrib = particles.mass[neighbor_id] * W
    print(f"  Neighbor {neighbor_id}: dist={dist:.1f}, W={W:.6f}, contrib={contrib:.6f}")
    density_sum += contrib

print(f"  Manual density sum: {density_sum:.6f}")

# Now compute with function
compute_density_vectorized(particles, kernel, n_active)

print(f"\nComputed densities: {particles.density[:n_active]}")
print(f"Center particle density: {particles.density[center_idx]:.6f}")

# Expected density calculation
# In 2D, particles form an area, so density should be mass/area
# With spacing of 50m and particles at each grid point,
# each particle "owns" an area of roughly 50m x 50m = 2500 m²
# So density = 1 kg / 2500 m² = 0.0004 kg/m²
# But SPH gives a volume density by assuming a thickness
print(f"\nExpected area density: {1.0 / (spacing * spacing):.6f} kg/m²")