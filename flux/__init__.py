"""
2D Geological Simulation System

A real-time physics-based geological simulation with interactive visualization.
Features heat transfer, metamorphism, and geological processes.
"""

"""Geo package – modular geological simulation engine."""

# Import the core simulation components
try:
    from .geo_game import GeoGame  # noqa: F401
    from .visualizer import GeologyVisualizer  # noqa: F401
except Exception:  # pragma: no cover – optional heavy subsystems missing
    GeoGame = None  # type: ignore
    GeologyVisualizer = None  # type: ignore

from .materials import MaterialType, MaterialDatabase

__version__ = "1.0.0"
__author__ = "Geology Simulator Team"

__all__ = [
    'GeoGame',
    'MaterialType', 
    'MaterialDatabase',
    'GeologyVisualizer'
] 