#!/usr/bin/env python3
"""
Main entry point for the 2D Geology Simulator.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the visualizer and game
try:
    from .visualizer import GeologyVisualizer
    from .geo_game import GeoGame
except ImportError:
    from visualizer import GeologyVisualizer
    from geo_game import GeoGame

def main():
    """Main entry point."""
    # Create simulation
    sim = GeoGame(200, 200, setup_planet=True)
    
    # Create and run visualizer with 1.5x scale
    viz = GeologyVisualizer(sim, window_width=800, window_height=800, scale_factor=1.5)
    viz.run()

if __name__ == "__main__":
    print("=" * 60)
    print("2D GEOLOGICAL SIMULATION SYSTEM (CA)")
    print("=" * 60)
    print()
    print("Features:")
    print("• Real-time heat transfer simulation")
    print("• Rock metamorphism based on P-T conditions") 
    print("• Interactive gravity-aware fluid dynamics")
    print("• Forward and backward time stepping")
    print("• Multiple visualization modes (materials, temperature, pressure, gravity)")
    print("• Solar illumination and day/night cycles")
    print()
    print("Press H for help once the simulation starts")
    print()
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 