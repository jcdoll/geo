"""
Numba-optimized force computation for SPH.

This is the biggest performance bottleneck - Numba provides 20-50x speedup.
"""

import numpy as np
import numba as nb
from typing import Tuple
from ..core.particles import ParticleArrays


@nb.njit(fastmath=True, cache=True)
def cubic_spline_gradient(dx: float, dy: float, r: float, h: float) -> Tuple[float, float]:
    """Cubic spline kernel gradient (2D)."""
    if r < 1e-10:
        return 0.0, 0.0
    
    q = r / h
    norm_2d = 10.0 / (7.0 * np.pi * h * h * h)
    
    if q <= 1.0:
        grad_mag = norm_2d * (-3 * q + 2.25 * q * q)
    elif q <= 2.0:
        grad_mag = -norm_2d * 0.75 * (2 - q) * (2 - q)
    else:
        return 0.0, 0.0
    
    # Apply direction
    factor = grad_mag / r
    return factor * dx, factor * dy


@nb.njit(parallel=True, fastmath=True, cache=True)
def compute_forces_numba(position_x: np.ndarray, position_y: np.ndarray,
                        velocity_x: np.ndarray, velocity_y: np.ndarray,
                        mass: np.ndarray, density: np.ndarray, pressure: np.ndarray,
                        smoothing_h: np.ndarray,
                        neighbor_ids: np.ndarray, neighbor_distances: np.ndarray,
                        neighbor_count: np.ndarray,
                        force_x: np.ndarray, force_y: np.ndarray,
                        n_active: int, gravity_x: float, gravity_y: float,
                        alpha_visc: float = 0.1):
    """Numba-optimized pressure and viscosity forces.
    
    This single function replaces the main bottleneck with 20-50x speedup.
    """
    # Process each particle in parallel
    for i in nb.prange(n_active):
        # Reset forces
        force_x[i] = mass[i] * gravity_x
        force_y[i] = mass[i] * gravity_y
        
        # Skip if no neighbors
        n_neighbors = neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        # Particle i properties
        px_i = position_x[i]
        py_i = position_y[i]
        vx_i = velocity_x[i]
        vy_i = velocity_y[i]
        h_i = smoothing_h[i]
        pressure_i = pressure[i] / (density[i] * density[i])
        
        # Process neighbors
        for j_idx in range(n_neighbors):
            j = neighbor_ids[i, j_idx]
            if j < 0:
                continue
            
            # Position and velocity differences
            dx = px_i - position_x[j]
            dy = py_i - position_y[j]
            dvx = vx_i - velocity_x[j]
            dvy = vy_i - velocity_y[j]
            r = neighbor_distances[i, j_idx]
            
            # Skip if too close
            if r < 1e-6:
                continue
            
            # Pressure gradient term
            pressure_j = pressure[j] / (density[j] * density[j])
            pressure_term = pressure_i + pressure_j
            
            # Artificial viscosity for stability
            visc_term = 0.0
            if alpha_visc > 0:
                v_dot_r = dvx * dx + dvy * dy
                if v_dot_r < 0:  # Approaching
                    # Simplified sound speed estimate
                    c_i = 10.0 * np.sqrt(abs(pressure[i]) / density[i] + 1e-6)
                    c_j = 10.0 * np.sqrt(abs(pressure[j]) / density[j] + 1e-6)
                    c_ij = 0.5 * (c_i + c_j)
                    
                    # Average smoothing length
                    h_j = smoothing_h[j]
                    h_ij = 0.5 * (h_i + h_j)
                    
                    # Monaghan viscosity
                    mu_ij = h_ij * v_dot_r / (r * r + 0.01 * h_ij * h_ij)
                    rho_ij = 0.5 * (density[i] + density[j])
                    visc_term = -alpha_visc * c_ij * mu_ij / rho_ij
            
            # Kernel gradient
            grad_x, grad_y = cubic_spline_gradient(dx, dy, r, h_i)
            
            # Total force
            force_factor = -mass[j] * (pressure_term + visc_term)
            force_x[i] += force_factor * grad_x
            force_y[i] += force_factor * grad_y


def compute_forces_numba_wrapper(particles: ParticleArrays, n_active: int,
                                gravity: np.ndarray = None, alpha_visc: float = 0.1):
    """Wrapper for Numba force computation."""
    if gravity is None:
        gravity = np.array([0.0, 0.0], dtype=np.float32)
    
    compute_forces_numba(
        particles.position_x, particles.position_y,
        particles.velocity_x, particles.velocity_y,
        particles.mass, particles.density, particles.pressure,
        particles.smoothing_h,
        particles.neighbor_ids, particles.neighbor_distances,
        particles.neighbor_count,
        particles.force_x, particles.force_y,
        n_active, gravity[0], gravity[1], alpha_visc
    )