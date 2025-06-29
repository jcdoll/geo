"""
Additional repulsive forces to prevent particle overlap in SPH.

These forces supplement the standard pressure forces when particles
get too close together.
"""

import numpy as np
from ..core.particles import ParticleArrays


def add_overlap_prevention_forces(particles: ParticleArrays, n_active: int,
                                overlap_distance: float = 0.5,
                                repulsion_strength: float = 1000.0) -> None:
    """
    Add short-range repulsive forces to prevent particle overlap.
    
    This implements a Lennard-Jones style repulsive force that activates
    when particles are closer than overlap_distance * smoothing_h.
    
    Args:
        particles: Particle arrays (forces will be added in-place)
        n_active: Number of active particles
        overlap_distance: Fraction of smoothing length where repulsion starts
        repulsion_strength: Strength of repulsive force
    """
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
            
        # Get neighbor data
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        distances = particles.neighbor_distances[i, neighbor_slice]
        
        # Position differences
        dx = particles.position_x[i] - particles.position_x[neighbor_ids]
        dy = particles.position_y[i] - particles.position_y[neighbor_ids]
        
        # Check for overlapping particles
        h_i = particles.smoothing_h[i]
        overlap_threshold = overlap_distance * h_i
        
        # Find particles that are too close
        too_close = distances < overlap_threshold
        if not np.any(too_close):
            continue
            
        # Compute repulsive force for close particles
        # F = strength * (1 - r/r_threshold)^2 * r_hat
        close_distances = distances[too_close]
        close_dx = dx[too_close]
        close_dy = dy[too_close]
        
        # Normalized separation (avoid division by zero)
        r_safe = np.maximum(close_distances, 0.01 * h_i)
        dx_norm = close_dx / r_safe
        dy_norm = close_dy / r_safe
        
        # Repulsive force magnitude
        # Quadratic repulsion: gets stronger as particles approach
        force_magnitude = repulsion_strength * particles.mass[i] * \
                         (1.0 - close_distances / overlap_threshold)**2
        
        # Apply forces (action-reaction pair)
        fx_repulsive = force_magnitude * dx_norm
        fy_repulsive = force_magnitude * dy_norm
        
        # Add to particle i
        particles.force_x[i] += np.sum(fx_repulsive)
        particles.force_y[i] += np.sum(fy_repulsive)
        
        # Newton's third law - add opposite force to neighbors
        close_neighbors = neighbor_ids[too_close]
        for j, nid in enumerate(close_neighbors):
            particles.force_x[nid] -= fx_repulsive[j]
            particles.force_y[nid] -= fy_repulsive[j]


def compute_overlap_prevention_pressure(distances: np.ndarray,
                                      smoothing_h: np.ndarray,
                                      overlap_fraction: float = 0.4) -> np.ndarray:
    """
    Compute additional pressure term for overlap prevention.
    
    This can be added to the standard pressure to create extra repulsion
    at short distances.
    
    Args:
        distances: Array of distances to neighbors
        smoothing_h: Smoothing lengths
        overlap_fraction: Fraction of h where pressure boost activates
        
    Returns:
        Additional pressure values
    """
    # Threshold distance
    if np.isscalar(smoothing_h):
        h_threshold = overlap_fraction * smoothing_h
    else:
        h_threshold = overlap_fraction * smoothing_h[:, np.newaxis]
    
    # Compute pressure boost for close particles
    pressure_boost = np.zeros_like(distances)
    
    # Find overlapping particles
    too_close = distances < h_threshold
    
    if np.any(too_close):
        # Exponential repulsion
        # P_boost = P_0 * exp(-r/r_0)
        r_0 = 0.1 * h_threshold  # Characteristic distance
        pressure_boost[too_close] = 1e6 * np.exp(-distances[too_close] / r_0)
    
    return pressure_boost