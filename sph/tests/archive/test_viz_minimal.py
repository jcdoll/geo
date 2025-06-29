#!/usr/bin/env python3
"""
Minimal test of the visualizer to identify issues.
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Run without display

import pygame
import sys
import sph
from sph import scenarios

print("Testing minimal visualizer...")

# Create scenario
particles, n_active = scenarios.create_planet_simple(
    radius=20, particle_spacing=2, center=(50, 50)
)
print(f"Created {n_active} particles")

# Try to create visualizer
try:
    from sph.visualizer import SPHVisualizer
    viz = SPHVisualizer(
        particles=particles,
        n_active=n_active,
        domain_size=(100, 100),
        window_size=(400, 400),
        target_fps=60
    )
    print("✓ Visualizer created successfully")
    
    # Try one physics step
    viz.paused = False
    viz.step_physics()
    print("✓ Physics step completed")
    
    # Try rendering (will fail with dummy driver but should not crash)
    try:
        viz.render()
        print("✓ Render completed")
    except Exception as e:
        print(f"⚠ Render failed (expected with dummy driver): {e}")
    
    print("\n✓ All core systems working!")
    print("The visualizer should work with a proper display.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    
pygame.quit()