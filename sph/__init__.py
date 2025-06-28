"""SPH (Smoothed Particle Hydrodynamics) implementation for geological simulation."""

from . import core
from . import physics
from . import scenarios

# Import API to trigger backend registration
from . import api

# Import unified API
from .api import (
    # Core functions
    compute_density,
    compute_forces,
    compute_gravity,
    create_spatial_hash,
    
    # Backend management
    set_backend,
    get_backend,
    auto_select_backend,
    print_backend_info,
    
    # Core classes
    ParticleArrays,
    CubicSplineKernel
)

# Import visualizer for main.py
from .visualizer import SPHVisualizer

__version__ = "0.2.0"

__all__ = [
    # Modules
    'core', 
    'physics',
    'scenarios',
    
    # API functions
    'compute_density',
    'compute_forces',
    'compute_gravity',
    'create_spatial_hash',
    
    # Backend management
    'set_backend',
    'get_backend',
    'auto_select_backend',
    'print_backend_info',
    
    # Core classes
    'ParticleArrays',
    'CubicSplineKernel'
]