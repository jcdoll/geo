#!/usr/bin/env python3
"""Test cohesive forces implementation."""

import numpy as np
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.physics.cohesion import compute_cohesive_forces_simple
from sph.physics.materials import MaterialType

# Create two particles close together
particles = ParticleArrays.allocate(10)
n_active = 2

# Place particles close together but not at equilibrium
particles.position_x[0] = 0.0
particles.position_y[0] = 0.0
particles.position_x[1] = 1.5  # Slightly farther than equilibrium
particles.position_y[1] = 0.0

# Set properties
particles.mass[:n_active] = 1.0
particles.material_id[:n_active] = MaterialType.ROCK.value
particles.smoothing_h[:n_active] = 2.0

# Set up neighbors manually
particles.neighbor_count[0] = 1
particles.neighbor_ids[0, 0] = 1
particles.neighbor_distances[0, 0] = 1.5

particles.neighbor_count[1] = 1
particles.neighbor_ids[1, 0] = 0
particles.neighbor_distances[1, 0] = 1.5

# Reset forces
particles.force_x[:n_active] = 0.0
particles.force_y[:n_active] = 0.0

print("Before cohesion:")
print(f"  Particle 0: F = ({particles.force_x[0]:.3f}, {particles.force_y[0]:.3f})")
print(f"  Particle 1: F = ({particles.force_x[1]:.3f}, {particles.force_y[1]:.3f})")

# Apply cohesive forces
compute_cohesive_forces_simple(particles, n_active, cohesion_strength=100.0)

print("\nAfter cohesion:")
print(f"  Particle 0: F = ({particles.force_x[0]:.3f}, {particles.force_y[0]:.3f})")
print(f"  Particle 1: F = ({particles.force_x[1]:.3f}, {particles.force_y[1]:.3f})")

# Forces should be attractive (pulling particles together)
# Particle 0 should have positive x force (toward particle 1)
# Particle 1 should have negative x force (toward particle 0)
if particles.force_x[0] > 0 and particles.force_x[1] < 0:
    print("\n✓ Cohesive forces working correctly!")
else:
    print("\n✗ Cohesive forces not working as expected")