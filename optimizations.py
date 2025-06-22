"""
Performance optimizations for flux-based simulation.

This module provides Numba JIT-compiled versions of performance-critical
functions for significant speedup.
"""

import numpy as np
from typing import Tuple

# Optional Numba import
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define dummy decorator
    class numba:
        @staticmethod
        def njit(*args, **kwargs):
            def decorator(func):
                return func
            return decorator


@numba.njit(parallel=True, fastmath=True)
def compute_mass_flux_numba(
    density: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    dt: float,
    dx: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized mass flux computation.
    
    Args:
        density: Density field
        velocity_x, velocity_y: Velocity components
        dt: Time step
        dx: Grid spacing
        
    Returns:
        (flux_x, flux_y): Mass fluxes at cell faces
    """
    ny, nx = density.shape
    
    # X-direction flux
    flux_x = np.zeros((ny, nx+1), dtype=np.float32)
    for j in numba.prange(ny):
        for i in range(1, nx):
            # Face velocity
            v_face = 0.5 * (velocity_x[j, i-1] + velocity_x[j, i])
            
            # Upwind scheme
            if v_face > 0:
                flux_x[j, i] = density[j, i-1] * v_face * dt / dx
            else:
                flux_x[j, i] = density[j, i] * v_face * dt / dx
                
    # Y-direction flux
    flux_y = np.zeros((ny+1, nx), dtype=np.float32)
    for j in range(1, ny):
        for i in numba.prange(nx):
            # Face velocity
            v_face = 0.5 * (velocity_y[j-1, i] + velocity_y[j, i])
            
            # Upwind scheme
            if v_face > 0:
                flux_y[j, i] = density[j-1, i] * v_face * dt / dx
            else:
                flux_y[j, i] = density[j, i] * v_face * dt / dx
                
    return flux_x, flux_y


@numba.njit(parallel=True, fastmath=True)
def apply_flux_divergence_numba(
    quantity: np.ndarray,
    flux_x: np.ndarray,
    flux_y: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized flux divergence application.
    
    Args:
        quantity: The quantity to update
        flux_x, flux_y: Fluxes at cell faces
        
    Returns:
        Updated quantity
    """
    ny, nx = quantity.shape
    result = quantity.copy()
    
    for j in numba.prange(ny):
        for i in range(nx):
            # X-direction divergence
            div_x = flux_x[j, i+1] - flux_x[j, i]
            
            # Y-direction divergence
            div_y = flux_y[j+1, i] - flux_y[j, i]
            
            # Update
            result[j, i] -= (div_x + div_y)
            
    return result


@numba.njit(parallel=True, fastmath=True)
def normalize_volume_fractions_numba(vol_frac: np.ndarray) -> np.ndarray:
    """
    Numba-optimized volume fraction normalization.
    
    Args:
        vol_frac: Volume fractions [n_materials, ny, nx]
        
    Returns:
        Normalized volume fractions
    """
    n_mat, ny, nx = vol_frac.shape
    result = vol_frac.copy()
    
    for j in numba.prange(ny):
        for i in range(nx):
            total = 0.0
            for m in range(n_mat):
                total += vol_frac[m, j, i]
                
            if total > 0:
                for m in range(n_mat):
                    result[m, j, i] = vol_frac[m, j, i] / total
            else:
                # All space
                result[0, j, i] = 1.0
                for m in range(1, n_mat):
                    result[m, j, i] = 0.0
                    
    return result


@numba.njit(parallel=True, fastmath=True)
def compute_mixture_density_numba(
    vol_frac: np.ndarray,
    material_densities: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized mixture density calculation.
    
    Args:
        vol_frac: Volume fractions [n_materials, ny, nx]
        material_densities: Density for each material type
        
    Returns:
        Mixture density field
    """
    n_mat, ny, nx = vol_frac.shape
    density = np.zeros((ny, nx), dtype=np.float32)
    
    for j in numba.prange(ny):
        for i in range(nx):
            for m in range(n_mat):
                density[j, i] += vol_frac[m, j, i] * material_densities[m]
                
    return density


def get_optimized_functions():
    """
    Get dictionary of optimized functions if Numba is available.
    
    Returns:
        Dict mapping function names to optimized implementations,
        or None if Numba is not available.
    """
    if not HAS_NUMBA:
        return None
        
    return {
        'compute_mass_flux': compute_mass_flux_numba,
        'apply_flux_divergence': apply_flux_divergence_numba,
        'normalize_volume_fractions': normalize_volume_fractions_numba,
        'compute_mixture_density': compute_mixture_density_numba,
    }