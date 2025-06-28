#!/usr/bin/env python3
"""Test the main entry point without opening a window."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import sph
from sph import scenarios

# Test backend
print("Testing SPH main entry point...")
sph.print_backend_info()

# Create a small scenario
print("\nCreating planet scenario...")
particles, n_active = scenarios.create_planet_simple(
    radius=20,
    particle_spacing=2,
    center=(50, 50)
)

print(f"Created {n_active} particles")
print(f"Domain: 100x100 m")

# Test physics step
print("\nTesting physics step...")
kernel = sph.CubicSplineKernel(dim=2)
spatial_hash = sph.create_spatial_hash((100, 100), 4.0)

# Build spatial hash
spatial_hash.build_vectorized(particles, n_active)
spatial_hash.query_neighbors_vectorized(particles, n_active, 4.0)

# Compute density
sph.compute_density(particles, kernel, n_active)
print(f"Average density: {particles.density[:n_active].mean():.1f} kg/m³")

# Compute forces
sph.compute_forces(particles, kernel, n_active, gravity=np.array([0, -9.81]))
print("Forces computed successfully")

print("\n✓ All systems operational!")
print("You can now run: python main.py")