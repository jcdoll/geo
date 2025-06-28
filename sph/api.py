"""
Unified API for SPH with automatic backend dispatch.

This module provides a clean interface that automatically dispatches
to CPU, Numba, or GPU implementations based on the current backend.
"""

import numpy as np
from typing import Optional
from .core.backend import dispatch, set_backend, get_backend, auto_select_backend, print_backend_info
from .core.particles import ParticleArrays
from .core.kernel_vectorized import CubicSplineKernel

# Import and register all implementations
from .core.backend import backend_function, for_backend, Backend

# CPU implementations
from .physics.density_vectorized import compute_density_vectorized
from .physics.forces_vectorized import compute_forces_vectorized
from .physics.gravity_vectorized import compute_gravity_direct_batched
from .core.spatial_hash_vectorized import VectorizedSpatialHash

# Register CPU implementations
@backend_function("compute_density")
@for_backend(Backend.CPU)
def _compute_density_cpu(particles: ParticleArrays, kernel: CubicSplineKernel, n_active: int):
    compute_density_vectorized(particles, kernel, n_active)

@backend_function("compute_forces")
@for_backend(Backend.CPU)
def _compute_forces_cpu(particles: ParticleArrays, kernel: CubicSplineKernel, n_active: int,
                       gravity: np.ndarray = None, alpha_visc: float = 0.1):
    compute_forces_vectorized(particles, kernel, n_active, gravity, alpha_visc)

@backend_function("compute_gravity")
@for_backend(Backend.CPU)
def _compute_gravity_cpu(particles: ParticleArrays, n_active: int,
                        G: float = 6.67430e-11, softening: float = 0.1):
    compute_gravity_direct_batched(particles, n_active, G, softening)

# Try to import and register Numba implementations
try:
    from .physics.density_numba import compute_density_numba_wrapper
    from .physics.forces_numba import compute_forces_numba_wrapper
    from .physics.gravity_numba import compute_gravity_numba_wrapper
    from .core.spatial_hash_numba import NumbaOptimizedSpatialHash
    
    @backend_function("compute_density")
    @for_backend(Backend.NUMBA)
    def _compute_density_numba(particles: ParticleArrays, kernel: CubicSplineKernel, n_active: int):
        compute_density_numba_wrapper(particles, n_active)
    
    @backend_function("compute_forces")
    @for_backend(Backend.NUMBA)
    def _compute_forces_numba(particles: ParticleArrays, kernel: CubicSplineKernel, n_active: int,
                             gravity: np.ndarray = None, alpha_visc: float = 0.1):
        compute_forces_numba_wrapper(particles, n_active, gravity, alpha_visc)
    
    @backend_function("compute_gravity")
    @for_backend(Backend.NUMBA)
    def _compute_gravity_numba(particles: ParticleArrays, n_active: int,
                              G: float = 6.67430e-11, softening: float = 0.1):
        compute_gravity_numba_wrapper(particles, n_active, G, softening)
    
except ImportError:
    pass

# Try to import and register GPU implementations
try:
    from .physics.density_gpu import compute_density_gpu
    from .physics.forces_gpu import compute_forces_gpu
    # Note: These are already decorated with @backend_function
    
except ImportError:
    pass


# Public API functions that dispatch to appropriate backend
def compute_density(particles: ParticleArrays, kernel: CubicSplineKernel = None, 
                   n_active: int = None, backend: Optional[str] = None):
    """Compute SPH density using current or specified backend.
    
    Args:
        particles: Particle arrays
        kernel: SPH kernel (not used by Numba/GPU backends)
        n_active: Number of active particles
        backend: Override backend ('cpu', 'numba', 'gpu', or None for current)
    """
    if n_active is None:
        n_active = len(particles.position_x)
    if kernel is None:
        kernel = CubicSplineKernel()
    
    dispatch("compute_density", particles, kernel, n_active, backend=backend)


def compute_forces(particles: ParticleArrays, kernel: CubicSplineKernel = None,
                  n_active: int = None, gravity: np.ndarray = None,
                  alpha_visc: float = 0.1, backend: Optional[str] = None):
    """Compute SPH forces using current or specified backend.
    
    Args:
        particles: Particle arrays
        kernel: SPH kernel (not used by Numba/GPU backends)
        n_active: Number of active particles
        gravity: Gravity vector [gx, gy]
        alpha_visc: Artificial viscosity parameter
        backend: Override backend
    """
    if n_active is None:
        n_active = len(particles.position_x)
    if kernel is None:
        kernel = CubicSplineKernel()
    
    dispatch("compute_forces", particles, kernel, n_active, gravity, alpha_visc, backend=backend)


def compute_gravity(particles: ParticleArrays, n_active: int = None,
                   G: float = 6.67430e-11, softening: float = 0.1,
                   backend: Optional[str] = None):
    """Compute self-gravity using current or specified backend.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        G: Gravitational constant
        softening: Softening length
        backend: Override backend
    """
    if n_active is None:
        n_active = len(particles.position_x)
    
    dispatch("compute_gravity", particles, n_active, G, softening, backend=backend)


# Spatial hash factory that respects backend
def create_spatial_hash(domain_size: tuple, cell_size: float,
                       max_per_cell: int = 100, domain_min: tuple = None) -> object:
    """Create spatial hash using current backend.
    
    Args:
        domain_size: (width, height) of simulation domain
        cell_size: Size of each cell
        max_per_cell: Maximum particles per cell
        domain_min: (min_x, min_y) of domain. If None, assumes (0, 0)
    
    Returns appropriate implementation based on backend.
    """
    backend = get_backend()
    
    if backend == "numba":
        try:
            from .core.spatial_hash_numba import NumbaOptimizedSpatialHash
            return NumbaOptimizedSpatialHash(domain_size, cell_size, max_per_cell, domain_min)
        except ImportError:
            pass
    
    # Default to CPU
    return VectorizedSpatialHash(domain_size, cell_size, max_per_cell, domain_min)


# Re-export backend management functions
__all__ = [
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