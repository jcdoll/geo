"""
Numba-optimized cohesive forces for maximum CPU performance.
"""

import numpy as np
import numba as nb
from typing import Tuple
from ..core.particles import ParticleArrays
from .materials import MaterialDatabase, MaterialType


@nb.njit(fastmath=True, cache=True)
def compute_cohesive_force_pair(
    dx: float, dy: float, r: float, 
    h_avg: float, cohesion: float,
    cutoff_factor: float = 1.5
) -> Tuple[float, float]:
    """Compute cohesive force between a pair of particles."""
    if r < 1e-6:
        return 0.0, 0.0
    
    # Reference distances
    r_min = 0.8 * h_avg
    r_cut = cutoff_factor * h_avg
    
    if r > r_cut:
        return 0.0, 0.0
    
    # Normalized direction
    rx = dx / r
    ry = dy / r
    
    # Force magnitude
    if r < r_min:
        # Repulsive
        force_mag = cohesion * (1.0 - r / r_min) / r_min
    else:
        # Attractive
        force_mag = -cohesion * (1.0 - (r - r_min) / (r_cut - r_min)) / r_min
    
    return force_mag * rx, force_mag * ry


@nb.njit(parallel=True, fastmath=True, cache=True)
def compute_cohesive_forces_numba_kernel(
    position_x: np.ndarray, position_y: np.ndarray,
    mass: np.ndarray, temperature: np.ndarray,
    material_id: np.ndarray, smoothing_h: np.ndarray,
    neighbor_ids: np.ndarray, neighbor_distances: np.ndarray,
    neighbor_count: np.ndarray,
    force_x: np.ndarray, force_y: np.ndarray,
    cohesion_strengths: np.ndarray,
    melting_points: np.ndarray,
    compatibility_matrix: np.ndarray,
    n_active: int,
    cutoff_factor: float = 1.5,
    temperature_softening: bool = True
):
    """
    Numba kernel for cohesive force computation.
    
    This provides maximum CPU performance through parallelization.
    """
    # Process each particle in parallel
    for i in nb.prange(n_active):
        if cohesion_strengths[i] <= 0:
            continue
            
        n_neighbors = neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        # Particle i properties
        px_i = position_x[i]
        py_i = position_y[i]
        h_i = smoothing_h[i]
        mat_i = material_id[i]
        T_i = temperature[i]
        T_melt_i = melting_points[i]
        mass_i = mass[i]
        
        # Process neighbors
        fx_total = 0.0
        fy_total = 0.0
        
        for j_idx in range(n_neighbors):
            j = neighbor_ids[i, j_idx]
            if j < 0 or cohesion_strengths[j] <= 0:
                continue
            
            r = neighbor_distances[i, j_idx]
            if r < 1e-6 or r > cutoff_factor * h_i:
                continue
            
            # Position difference
            dx = px_i - position_x[j]
            dy = py_i - position_y[j]
            
            # Average properties
            h_avg = 0.5 * (h_i + smoothing_h[j])
            cohesion = 0.5 * (cohesion_strengths[i] + cohesion_strengths[j])
            
            # Material compatibility
            mat_j = material_id[j]
            compatibility = compatibility_matrix[mat_i, mat_j]
            cohesion *= compatibility
            
            if cohesion <= 0:
                continue
            
            # Temperature softening
            if temperature_softening:
                T_j = temperature[j]
                T_melt_j = melting_points[j]
                
                temp_factor_i = max(0.0, 1.0 - T_i / T_melt_i)
                temp_factor_j = max(0.0, 1.0 - T_j / T_melt_j)
                temp_factor = 0.5 * (temp_factor_i + temp_factor_j)
                
                cohesion *= temp_factor
            
            # Compute force
            fx, fy = compute_cohesive_force_pair(dx, dy, r, h_avg, cohesion, cutoff_factor)
            
            # Mass weighting
            mass_j = mass[j]
            mass_factor = mass_j / (mass_i + mass_j)
            
            # Accumulate forces
            fx_total += fx * mass_factor
            fy_total += fy * mass_factor
            
            # Newton's third law (atomic operations would be ideal here)
            # For now, we'll handle this in a separate pass
        
        # Update particle i forces
        force_x[i] += fx_total
        force_y[i] += fy_total


def compute_cohesive_forces_numba(
    particles: ParticleArrays,
    n_active: int,
    material_db: MaterialDatabase,
    cutoff_factor: float = 1.5,
    temperature_softening: bool = True
):
    """
    Wrapper for Numba-optimized cohesive force computation.
    """
    # Pre-compute material properties
    max_particles = len(particles.position_x)
    cohesion_strengths = np.zeros(max_particles, dtype=np.float32)
    melting_points = np.full(max_particles, 1e6, dtype=np.float32)
    
    # Build compatibility matrix
    n_materials = len(MaterialType)
    compatibility_matrix = np.zeros((n_materials, n_materials), dtype=np.float32)
    
    # Same material: full compatibility
    for i in range(n_materials):
        compatibility_matrix[i, i] = 1.0
    
    # Special rules
    compatibility_matrix[MaterialType.ROCK.value, MaterialType.URANIUM.value] = 0.8
    compatibility_matrix[MaterialType.URANIUM.value, MaterialType.ROCK.value] = 0.8
    compatibility_matrix[MaterialType.WATER.value, MaterialType.ICE.value] = 0.5
    compatibility_matrix[MaterialType.ICE.value, MaterialType.WATER.value] = 0.5
    
    # Default for different solids
    solid_materials = [MaterialType.ROCK, MaterialType.ICE, MaterialType.URANIUM]
    for i, mat_i in enumerate(solid_materials):
        for j, mat_j in enumerate(solid_materials):
            if i != j and compatibility_matrix[mat_i.value, mat_j.value] == 0:
                compatibility_matrix[mat_i.value, mat_j.value] = 0.1
    
    # Get material properties
    for i in range(n_active):
        mat_type = MaterialType(particles.material_id[i])
        mat_props = material_db.get_properties(mat_type)
        
        if hasattr(mat_props, 'cohesion_strength') and mat_props.cohesion_strength is not None:
            cohesion_strengths[i] = mat_props.cohesion_strength
        
        if hasattr(mat_props, 'melting_point') and mat_props.melting_point is not None:
            melting_points[i] = mat_props.melting_point
    
    # Call Numba kernel
    compute_cohesive_forces_numba_kernel(
        particles.position_x, particles.position_y,
        particles.mass, particles.temperature,
        particles.material_id, particles.smoothing_h,
        particles.neighbor_ids, particles.neighbor_distances,
        particles.neighbor_count,
        particles.force_x, particles.force_y,
        cohesion_strengths, melting_points,
        compatibility_matrix,
        n_active, cutoff_factor, temperature_softening
    )
    
    # Handle Newton's third law in a separate pass
    # This is needed because Numba doesn't support atomic operations well
    for i in range(n_active):
        if cohesion_strengths[i] <= 0:
            continue
            
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        for j_idx in range(n_neighbors):
            j = particles.neighbor_ids[i, j_idx]
            if j < 0 or j >= i:  # Only process each pair once
                continue
            
            r = particles.neighbor_distances[i, j_idx]
            if r < 1e-6 or r > cutoff_factor * particles.smoothing_h[i]:
                continue
            
            if cohesion_strengths[j] <= 0:
                continue
            
            # Compute force between i and j
            dx = particles.position_x[i] - particles.position_x[j]
            dy = particles.position_y[i] - particles.position_y[j]
            
            h_avg = 0.5 * (particles.smoothing_h[i] + particles.smoothing_h[j])
            cohesion = 0.5 * (cohesion_strengths[i] + cohesion_strengths[j])
            
            # Material compatibility
            compatibility = compatibility_matrix[particles.material_id[i], particles.material_id[j]]
            cohesion *= compatibility
            
            if cohesion <= 0:
                continue
            
            # Temperature softening
            if temperature_softening:
                T_i = particles.temperature[i]
                T_j = particles.temperature[j]
                T_melt_i = melting_points[i]
                T_melt_j = melting_points[j]
                
                temp_factor_i = max(0.0, 1.0 - T_i / T_melt_i)
                temp_factor_j = max(0.0, 1.0 - T_j / T_melt_j)
                temp_factor = 0.5 * (temp_factor_i + temp_factor_j)
                
                cohesion *= temp_factor
            
            # Compute force
            fx, fy = compute_cohesive_force_pair(dx, dy, r, h_avg, cohesion, cutoff_factor)
            
            # Apply equal and opposite forces
            mass_i = particles.mass[i]
            mass_j = particles.mass[j]
            total_mass = mass_i + mass_j
            
            particles.force_x[i] += fx * mass_j / total_mass
            particles.force_y[i] += fy * mass_j / total_mass
            particles.force_x[j] -= fx * mass_i / total_mass
            particles.force_y[j] -= fy * mass_i / total_mass