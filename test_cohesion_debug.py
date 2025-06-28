#!/usr/bin/env python3
"""Debug cohesive forces implementation."""

import numpy as np
from sph.core.particles import ParticleArrays
from sph.physics.materials import MaterialType

# Create two particles
particles = ParticleArrays.allocate(10)
n_active = 2

# Place particles
particles.position_x[0] = 0.0
particles.position_y[0] = 0.0
particles.position_x[1] = 1.5  # Further apart
particles.position_y[1] = 0.0

# Set properties
particles.mass[:n_active] = 1.0
particles.material_id[:n_active] = MaterialType.ROCK.value
particles.smoothing_h[:n_active] = 2.0

# Set up neighbors
particles.neighbor_count[0] = 1
particles.neighbor_ids[0, 0] = 1
particles.neighbor_distances[0, 0] = 1.5

particles.neighbor_count[1] = 1
particles.neighbor_ids[1, 0] = 0
particles.neighbor_distances[1, 0] = 1.5

# Reset forces
particles.force_x[:n_active] = 0.0
particles.force_y[:n_active] = 0.0

# Manually compute cohesive force
cohesion_strength = 100.0
cutoff_factor = 1.5

for i in range(n_active):
    n_neighbors = particles.neighbor_count[i]
    print(f"\nParticle {i}: {n_neighbors} neighbors")
    
    if n_neighbors == 0:
        continue
    
    neighbor_ids = particles.neighbor_ids[i, :n_neighbors]
    distances = particles.neighbor_distances[i, :n_neighbors]
    
    h_i = particles.smoothing_h[i]
    close_mask = distances < (cutoff_factor * h_i)
    print(f"  h={h_i}, cutoff={cutoff_factor * h_i}")
    print(f"  distances={distances}")
    print(f"  close_mask={close_mask}")
    
    if not np.any(close_mask):
        print("  No close neighbors!")
        continue
    
    close_neighbors = neighbor_ids[close_mask]
    close_distances = distances[close_mask]
    
    dx = particles.position_x[i] - particles.position_x[close_neighbors]
    dy = particles.position_y[i] - particles.position_y[close_neighbors]
    
    r_eq = 0.5 * h_i
    print(f"  r_eq={r_eq}")
    print(f"  close_distances={close_distances}")
    
    # Force calculation
    force_mags = -cohesion_strength * (close_distances - r_eq)
    print(f"  force_mags before filter={force_mags}")
    
    # Only attractive forces
    force_mags[close_distances < r_eq] = 0.0
    print(f"  force_mags after filter={force_mags}")
    
    force_mags = force_mags / close_distances
    print(f"  force_mags normalized={force_mags}")
    
    # Force components
    fx = force_mags * dx / close_distances
    fy = force_mags * dy / close_distances
    
    print(f"  dx={dx}, dy={dy}")
    print(f"  fx={fx}, fy={fy}")
    
    # Apply forces
    particles.force_x[i] += np.sum(fx * particles.mass[close_neighbors])
    particles.force_y[i] += np.sum(fy * particles.mass[close_neighbors])
    
    print(f"  Final force: ({particles.force_x[i]:.3f}, {particles.force_y[i]:.3f})")

print(f"\nFinal forces:")
print(f"  Particle 0: F = ({particles.force_x[0]:.3f}, {particles.force_y[0]:.3f})")
print(f"  Particle 1: F = ({particles.force_x[1]:.3f}, {particles.force_y[1]:.3f})")