"""
Enhanced pressure calculation with overlap prevention for SPH.

Includes:
- Short-range repulsive force to prevent particle overlap
- Improved artificial pressure
- Better handling of high compression
"""

import numpy as np
from ..physics.materials import MaterialType


def compute_pressure_with_overlap_prevention(
    density: np.ndarray,
    density_ref: np.ndarray,
    bulk_modulus: np.ndarray,
    smoothing_h: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_count: np.ndarray,
    gamma: float = 7.0,
    overlap_threshold: float = 0.3,
    repulsion_strength: float = 100.0) -> np.ndarray:
    """
    Compute pressure with overlap prevention.
    
    Uses modified Tait equation plus short-range repulsive force
    when particles get too close (< overlap_threshold * h).
    
    Args:
        density: Current density array
        density_ref: Reference density array  
        bulk_modulus: Bulk modulus array (Pa)
        smoothing_h: Smoothing length array
        neighbor_distances: Distance to neighbors (N, K)
        neighbor_count: Number of neighbors per particle
        gamma: Tait equation exponent (typically 7)
        overlap_threshold: Distance ratio below which repulsion activates
        repulsion_strength: Strength of overlap prevention force
        
    Returns:
        Pressure array with overlap prevention
    """
    n_particles = len(density)
    
    # Standard Tait pressure (but less aggressive reduction)
    B_stable = bulk_modulus * 0.1  # Only reduce by 10x instead of 100x
    ratio = density / density_ref
    
    # Clamp density ratio more aggressively
    ratio = np.clip(ratio, 0.5, 2.0)
    
    # Base pressure from Tait equation
    pressure = B_stable * (ratio**gamma - 1.0)
    
    # Improved artificial pressure (Monaghan 2000)
    # P_art = ε * B * (ρ/ρ₀)^2
    epsilon = 0.1  # Artificial pressure coefficient
    artificial_pressure = epsilon * B_stable * ratio**2
    pressure += artificial_pressure
    
    # Add short-range repulsion when particles get too close
    repulsion_pressure = np.zeros_like(pressure)
    
    for i in range(n_particles):
        if neighbor_count[i] == 0:
            continue
            
        # Check distances to neighbors
        h_i = smoothing_h[i]
        threshold_dist = overlap_threshold * h_i
        
        for j in range(neighbor_count[i]):
            dist = neighbor_distances[i, j]
            
            # If neighbor is too close, add repulsive pressure
            if dist < threshold_dist and dist > 0:
                # Repulsive pressure increases sharply as distance decreases
                # P_rep = strength * (1 - dist/threshold)^2
                overlap_factor = 1.0 - dist / threshold_dist
                repulsion_pressure[i] += repulsion_strength * B_stable[i] * overlap_factor**2
    
    # Combine all pressure components
    total_pressure = pressure + repulsion_pressure
    
    # Clamp to reasonable range
    max_pressure = 50.0 * B_stable  # Allow higher pressure for overlap prevention
    total_pressure = np.clip(total_pressure, -0.1 * B_stable, max_pressure)
    
    return total_pressure


def compute_pressure_simple_repulsive(
    density: np.ndarray,
    density_ref: np.ndarray, 
    bulk_modulus: np.ndarray,
    gamma: float = 7.0) -> np.ndarray:
    """
    Simpler pressure with built-in repulsion through modified Tait equation.
    
    Uses a modified equation that naturally provides more repulsion
    at high densities.
    """
    # Use lower effective bulk modulus for numerical stability
    B_effective = bulk_modulus * 0.01  # 1% for much larger timesteps
    
    # Density ratio with tighter bounds
    ratio = np.clip(density / density_ref, 0.8, 1.5)
    
    # Modified Tait with additional repulsive term
    # Standard term
    pressure_tait = B_effective * (ratio**gamma - 1.0)
    
    # Additional repulsive term that activates at high density
    # This provides extra repulsion when ρ > ρ₀
    pressure_repulsive = B_effective * np.maximum(0, ratio - 1.0)**3
    
    # Background pressure to maintain particle spacing
    pressure_background = 0.05 * B_effective
    
    total_pressure = pressure_tait + pressure_repulsive + pressure_background
    
    # Asymmetric clamping - allow high pressure but limit negative
    return np.clip(total_pressure, -0.01 * B_effective, 100.0 * B_effective)


def get_stable_bulk_modulus_improved(material_id: int) -> float:
    """Get improved bulk modulus for better particle spacing."""
    # Much lower values for reasonable timesteps
    # These are artificial values for numerical stability, not physical
    bulk_modulus_map = {
        MaterialType.WATER.value: 2e6,      # 100x lower
        MaterialType.ICE.value: 9e6,        # 100x lower
        MaterialType.ROCK.value: 1e7,       # 100x lower
        MaterialType.MAGMA.value: 5e6,      # 100x lower
        MaterialType.SAND.value: 2e6,       # 100x lower
        MaterialType.URANIUM.value: 2e7,    # 100x lower
        MaterialType.AIR.value: 1e3,        # Very low for gas
        MaterialType.WATER_VAPOR.value: 1e3,# Very low for gas
        MaterialType.SPACE.value: 1e2,      # Minimal
    }
    return bulk_modulus_map.get(material_id, 1e6)