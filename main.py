#!/usr/bin/env python3
"""
Main entry point for SPH geological simulation.

Usage:
    python main.py                    # Run with default planet scenario
    python main.py --scenario water   # Run water drop scenario
    python main.py --size 100         # Set simulation size
    python main.py --backend numba    # Use Numba backend
"""

import argparse
import sys
import os

# Add sph to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sph.visualizer import SPHVisualizer
from sph import scenarios
import sph


def main():
    parser = argparse.ArgumentParser(description="SPH Geological Simulation")
    parser.add_argument(
        "--scenario", 
        default="planet",
        choices=["planet", "water", "thermal", "volcanic", "layered"],
        help="Initial scenario to load (default: planet)"
    )
    parser.add_argument(
        "--size", 
        type=int, 
        default=100,
        help="Domain size in meters (default: 100)"
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=None,
        help="Number of particles (default: auto based on scenario)"
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "numba", "gpu", "auto"],
        default="auto",
        help="Computation backend (default: auto)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target FPS (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Set backend
    if args.backend == "auto":
        # Will be set based on particle count
        backend = None
    else:
        if not sph.set_backend(args.backend):
            print(f"Warning: Backend '{args.backend}' not available, using default")
            backend = None
        else:
            backend = args.backend
            print(f"Using {args.backend.upper()} backend")
    
    # Create scenario
    print(f"Loading scenario: {args.scenario}")
    
    # Import enhanced planet scenario
    from sph.scenarios.planet_with_atmosphere import (
        create_planet_with_atmosphere, create_simple_ocean_world
    )
    
    # Scenario functions with better planets
    scenario_funcs = {
        "planet": lambda: scenarios.create_planet_simple(
            radius=args.size * 0.4,
            particle_spacing=args.size * 0.01,  # Finer spacing for more particles
            center=(0.0, 0.0)
        ),
        "water": lambda: create_simple_ocean_world(
            radius=args.size * 0.3,
            particle_spacing=args.size * 0.02,
            ocean_fraction=0.4,
            center=(0.0, 0.0)
        ),
        "thermal": lambda: scenarios.create_planet_simple(
            radius=args.size * 0.3,
            particle_spacing=args.size * 0.02,
            center=(0.0, 0.0)
        ),
        "volcanic": lambda: scenarios.create_planet_earth_like(
            radius=args.size * 0.4,
            particle_spacing=args.size * 0.02,
            center=(0.0, 0.0)
        ),
        "layered": lambda: scenarios.create_planet_earth_like(
            radius=args.size * 0.4,
            particle_spacing=args.size * 0.02,
            center=(0.0, 0.0)
        ),
    }
    
    # Create particles
    particles, n_active = scenario_funcs[args.scenario]()
    
    # Override particle count if specified
    if args.particles and args.particles < n_active:
        n_active = args.particles
        print(f"Using {n_active} particles (reduced from {particles.position_x.shape[0]})")
    
    # Auto-select backend if not specified
    if backend is None:
        backend = sph.auto_select_backend(n_active)
        print(f"Auto-selected {backend.upper()} backend for {n_active} particles")
    
    # Print info
    sph.print_backend_info()
    print(f"\nSimulation info:")
    print(f"  Particles: {n_active}")
    print(f"  Domain: {args.size}x{args.size} m")
    print(f"  Scenario: {args.scenario}")
    print(f"  Target FPS: {args.fps}")
    
    # Create and run visualizer
    viz = SPHVisualizer(
        particles=particles,
        n_active=n_active,
        domain_size=(args.size, args.size),
        window_size=(800, 900),  # Extra height for toolbar
        target_fps=args.fps
    )
    
    print("\nStarting visualization...")
    print("Press H for help, ESC to exit")
    
    viz.run()


if __name__ == "__main__":
    main()