"""
2D Geological Simulation System

A real-time physics-based geological simulation with interactive visualization.
Features heat transfer, metamorphism, and geological processes.
"""

from .simulation_engine import GeologySimulation
from .materials import MaterialType, MaterialDatabase
from .visualizer import GeologyVisualizer

__version__ = "1.0.0"
__author__ = "Geology Simulator Team"

__all__ = [
    'GeologySimulation',
    'MaterialType', 
    'MaterialDatabase',
    'GeologyVisualizer'
] 