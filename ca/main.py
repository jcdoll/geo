#!/usr/bin/env python3
"""
Main entry point for the 2D Geology Simulator.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle both package and direct execution
try:
    from .visualizer import main
except ImportError:
    from visualizer import main

if __name__ == "__main__":
    print("=" * 60)
    print("2D GEOLOGICAL SIMULATION SYSTEM")
    print("=" * 60)
    print()
    print("Features:")
    print("• Real-time heat transfer simulation")
    print("• Rock metamorphism based on P-T conditions") 
    print("• Interactive tools for adding heat sources and pressure")
    print("• Forward and backward time stepping")
    print("• Multiple visualization modes (materials, temperature, pressure, power)")
    print()
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running simulation: {e}")
        sys.exit(1) 