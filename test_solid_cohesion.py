#!/usr/bin/env python3
"""
Test script to verify solid particle cohesion.

Creates two rock blocks and checks if they stick together.
"""

import numpy as np
import sph
from sph.core.particles import ParticleArrays
from sph.physics import MaterialType
from sph.visualizer import SPHVisualizer


def create_rock_block(center_x: float, center_y: float, width: float, height: float,
                     particles: ParticleArrays, start_idx: int, spacing: float = 1.3) -> int:
    """Create a block of rock particles."""
    nx = int(width / spacing)
    ny = int(height / spacing)
    
    idx = start_idx
    for i in range(nx):
        for j in range(ny):
            x = center_x - width/2 + i * spacing + spacing/2
            y = center_y - height/2 + j * spacing + spacing/2
            
            particles.position_x[idx] = x
            particles.position_y[idx] = y
            particles.velocity_x[idx] = 0.0
            particles.velocity_y[idx] = 0.0
            particles.material_id[idx] = MaterialType.ROCK.value
            particles.smoothing_h[idx] = 1.3
            particles.mass[idx] = 2700.0 * np.pi * 1.3**2  # density * volume
            particles.density[idx] = 2700.0
            particles.pressure[idx] = 0.0
            particles.temperature[idx] = 288.0
            particles.force_x[idx] = 0.0
            particles.force_y[idx] = 0.0
            particles.neighbor_count[idx] = 0
            
            idx += 1
    
    return idx - start_idx


def main():
    # Create particle arrays
    max_particles = 10000
    particles = ParticleArrays.allocate(max_particles)
    
    # Create two rock blocks that should stick together
    n_active = 0
    
    # First block (stationary)
    n1 = create_rock_block(-5, 0, 10, 10, particles, n_active)
    n_active += n1
    print(f"Created first rock block with {n1} particles")
    
    # Second block (moving slowly toward first)
    n2 = create_rock_block(8, 0, 10, 10, particles, n_active)
    # Give it a small leftward velocity
    for i in range(n_active, n_active + n2):
        particles.velocity_x[i] = -2.0  # 2 m/s leftward
    n_active += n2
    print(f"Created second rock block with {n2} particles")
    
    print(f"Total particles: {n_active}")
    
    # Create visualizer
    visualizer = SPHVisualizer(
        particles=particles,
        n_active=n_active,
        domain_size=(50.0, 50.0),
        window_size=(800, 800),
        target_fps=60
    )
    
    # Configure for this test
    visualizer.enable_external_gravity = False  # No gravity
    visualizer.enable_self_gravity = False      # No self-gravity
    visualizer.enable_pressure = True
    visualizer.enable_viscosity = True
    
    # Store initial state for reset
    visualizer.store_initial_state()
    
    print("\nControls:")
    print("- SPACE: Pause/Resume")
    print("- R: Reset to initial state")
    print("- M: Cycle display modes")
    print("- ESC: Exit")
    print("\nThe two rock blocks should collide and stick together due to cohesion.")
    
    # Run visualization
    visualizer.run()


if __name__ == "__main__":
    main()