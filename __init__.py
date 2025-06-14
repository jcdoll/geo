"""
2D Geological Simulation System

A real-time physics-based geological simulation with interactive visualization.
Features heat transfer, metamorphism, and geological processes.
"""

"""Geo package – modular geological simulation engine."""

# Import heavy engine lazily – allows utility submodules (e.g. gravity_solver)
# to be imported without pulling in the full simulation stack (and its legacy
# dependencies) when only analysis utilities are required.
try:
    from .simulation_engine import GeologySimulation  # noqa: F401
    from .visualizer import GeologyVisualizer  # noqa: F401
except Exception:  # pragma: no cover – optional heavy subsystems missing
    GeologySimulation = None  # type: ignore
    GeologyVisualizer = None  # type: ignore

from .materials import MaterialType, MaterialDatabase

__version__ = "1.0.0"
__author__ = "Geology Simulator Team"

__all__ = [
    'GeologySimulation',
    'MaterialType', 
    'MaterialDatabase',
    'GeologyVisualizer'
] 