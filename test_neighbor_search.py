#!/usr/bin/env python3
"""Debug neighbor search."""

import numpy as np
import sys
sys.path.insert(0, '.')

from sph.core.particles import ParticleArrays
import sph

# Create simple test case
particles = ParticleArrays.allocate(10)
n_active = 3

# Three particles in a line
particles.position_x[0] = 0.0
particles.position_y[0] = 0.0
particles.position_x[1] = 50.0  # 50m apart
particles.position_y[1] = 0.0
particles.position_x[2] = 100.0
particles.position_y[2] = 0.0

particles.smoothing_h[:n_active] = 100.0

print("Particle positions:")
for i in range(n_active):
    print(f"  {i}: ({particles.position_x[i]:.1f}, {particles.position_y[i]:.1f})")

# Create spatial hash
domain_size = (1000.0, 1000.0)
domain_min = (-500.0, -500.0)
cell_size = 100.0
spatial_hash = sph.create_spatial_hash(domain_size, cell_size, domain_min=domain_min)

# Build spatial hash first
print("\nBuilding spatial hash...")
spatial_hash.build_vectorized(particles, n_active)

# Search for neighbors
search_radius = 200.0  # 2 * h
print(f"\nSearching with radius {search_radius}...")
spatial_hash.query_neighbors_vectorized(particles, n_active, search_radius)

# Print results
print("\nNeighbor results:")
for i in range(n_active):
    n_neighbors = particles.neighbor_count[i]
    print(f"  Particle {i}: {n_neighbors} neighbors")
    if n_neighbors > 0:
        for j in range(n_neighbors):
            neighbor_id = particles.neighbor_ids[i, j]
            distance = particles.neighbor_distances[i, j]
            print(f"    -> neighbor {neighbor_id} at distance {distance:.1f}")

# Manual distance check
print("\nManual distance check:")
for i in range(n_active):
    for j in range(n_active):
        if i != j:
            dx = particles.position_x[i] - particles.position_x[j]
            dy = particles.position_y[i] - particles.position_y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            print(f"  Distance {i}->{j}: {dist:.1f} (should find: {dist < search_radius})")