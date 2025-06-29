"""
Vectorized cohesive forces for SPH particles.

Optimized for performance and GPU compatibility.
"""

import numpy as np
from typing import Optional
from ..core.particles import ParticleArrays
from ..core.kernel_vectorized import CubicSplineKernel
from .materials import MaterialDatabase, MaterialType


def compute_cohesive_forces_vectorized(
    particles: ParticleArrays,
    kernel: CubicSplineKernel,
    n_active: int,
    material_db: MaterialDatabase,
    cutoff_factor: float = 1.5,
    temperature_softening: bool = True
):
    """
    Vectorized cohesive force computation for better performance.
    
    This implementation minimizes loops and is more suitable for GPU porting.
    
    Args:
        particles: Particle arrays with neighbor information
        kernel: SPH kernel function
        n_active: Number of active particles
        material_db: Material property database
        cutoff_factor: Cutoff distance as fraction of smoothing length
        temperature_softening: Reduce cohesion at high temperatures
    """
    # Pre-compute material properties for all particles
    max_particles = len(particles.position_x)
    cohesion_strengths = np.zeros(max_particles, dtype=np.float32)
    melting_points = np.full(max_particles, 1e6, dtype=np.float32)
    
    # Material compatibility matrix (symmetric)
    # This could be precomputed once and stored
    n_materials = len(MaterialType)
    compatibility_matrix = np.zeros((n_materials, n_materials), dtype=np.float32)
    
    # Fill compatibility matrix
    for i in range(n_materials):
        compatibility_matrix[i, i] = 1.0  # Same material: full cohesion
    
    # Special compatibility rules
    compatibility_matrix[MaterialType.ROCK.value, MaterialType.URANIUM.value] = 0.8
    compatibility_matrix[MaterialType.URANIUM.value, MaterialType.ROCK.value] = 0.8
    compatibility_matrix[MaterialType.WATER.value, MaterialType.ICE.value] = 0.5
    compatibility_matrix[MaterialType.ICE.value, MaterialType.WATER.value] = 0.5
    
    # Default weak bonding between different solids
    solid_materials = [MaterialType.ROCK, MaterialType.ICE, MaterialType.URANIUM, MaterialType.SAND]
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
    
    # Process all particles at once
    for i in range(n_active):
        if cohesion_strengths[i] <= 0:
            continue
            
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        # Get neighbor data (vectorized)
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        distances = particles.neighbor_distances[i, neighbor_slice]
        
        # Filter by distance
        h_i = particles.smoothing_h[i]
        cutoff_dist = cutoff_factor * h_i
        close_mask = (distances < cutoff_dist) & (distances > 1e-6)
        
        if not np.any(close_mask):
            continue
        
        # Work with close neighbors
        close_neighbors = neighbor_ids[close_mask]
        close_distances = distances[close_mask]
        
        # Skip neighbors with no cohesion
        neighbor_cohesion = cohesion_strengths[close_neighbors]
        valid_mask = neighbor_cohesion > 0
        
        if not np.any(valid_mask):
            continue
        
        # Final neighbor set
        valid_neighbors = close_neighbors[valid_mask]
        valid_distances = close_distances[valid_mask]
        n_valid = len(valid_neighbors)
        
        # Vectorized computation of position differences
        dx = particles.position_x[i] - particles.position_x[valid_neighbors]
        dy = particles.position_y[i] - particles.position_y[valid_neighbors]
        
        # Average cohesion strengths
        cohesion = 0.5 * (cohesion_strengths[i] + cohesion_strengths[valid_neighbors])
        
        # Material compatibility (vectorized lookup)
        mat_i = particles.material_id[i]
        mat_j = particles.material_id[valid_neighbors]
        compatibility = compatibility_matrix[mat_i, mat_j]
        cohesion *= compatibility
        
        # Temperature softening (if enabled)
        if temperature_softening:
            T_i = particles.temperature[i]
            T_j = particles.temperature[valid_neighbors]
            T_melt_i = melting_points[i]
            T_melt_j = melting_points[valid_neighbors]
            
            # Temperature factors
            temp_factor_i = np.maximum(0.0, 1.0 - T_i / T_melt_i)
            temp_factor_j = np.maximum(0.0, 1.0 - T_j / T_melt_j)
            temp_factor = 0.5 * (temp_factor_i + temp_factor_j)
            
            cohesion *= temp_factor
        
        # Smoothing lengths
        h_j = particles.smoothing_h[valid_neighbors]
        h_avg = 0.5 * (h_i + h_j)
        
        # Reference distances
        r_min = 0.8 * h_avg  # Equilibrium distance
        r_cut = cutoff_factor * h_avg  # Cutoff distance
        
        # Normalized directions
        r_norm = valid_distances
        rx = dx / r_norm
        ry = dy / r_norm
        
        # Force calculation (vectorized)
        # Short-range repulsion
        repulsive_mask = valid_distances < r_min
        attractive_mask = ~repulsive_mask
        
        force_mag = np.zeros_like(valid_distances)
        
        # Convert cohesion strength (Pa) to force scale
        # Cohesion acts over an area ~ h²
        # Use much smaller scaling for numerical stability
        area_scale = h_avg**2
        cohesion_force = cohesion * area_scale * 1e-6  # Much smaller scale for stability
        
        # Repulsive forces
        if np.any(repulsive_mask):
            force_mag[repulsive_mask] = cohesion_force[repulsive_mask] * \
                (1.0 - valid_distances[repulsive_mask] / r_min[repulsive_mask]) / r_min[repulsive_mask]
        
        # Attractive forces
        if np.any(attractive_mask):
            force_mag[attractive_mask] = -cohesion_force[attractive_mask] * \
                (1.0 - (valid_distances[attractive_mask] - r_min[attractive_mask]) / 
                 (r_cut[attractive_mask] - r_min[attractive_mask])) / r_min[attractive_mask]
        
        # Force components
        fx = force_mag * rx
        fy = force_mag * ry
        
        # Mass factors for equal and opposite forces
        mass_i = particles.mass[i]
        mass_j = particles.mass[valid_neighbors]
        mass_factor = mass_j / (mass_i + mass_j)
        
        # Apply forces to particle i (sum of all neighbor contributions)
        particles.force_x[i] += np.sum(fx * mass_factor)
        particles.force_y[i] += np.sum(fy * mass_factor)
        
        # Newton's third law (vectorized update)
        # This is the tricky part for full vectorization - we need to scatter-add
        # For now, we'll do a small loop here, but this can be optimized further
        for j_idx, j in enumerate(valid_neighbors):
            particles.force_x[j] -= fx[j_idx] * (1.0 - mass_factor[j_idx])
            particles.force_y[j] -= fy[j_idx] * (1.0 - mass_factor[j_idx])


def compute_cohesive_forces_simple_vectorized(
    particles: ParticleArrays,
    n_active: int,
    cohesion_strength: float = 1e5,
    cutoff_factor: float = 1.5
):
    """
    Simplified vectorized cohesive force with uniform strength.
    
    This version is optimized for GPU implementation.
    """
    # Process each particle
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        # Get all neighbor data at once
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        distances = particles.neighbor_distances[i, neighbor_slice]
        
        # Filter by distance
        h_i = particles.smoothing_h[i]
        close_mask = (distances < (cutoff_factor * h_i)) & (distances > 1e-6)
        
        if not np.any(close_mask):
            continue
        
        # Vectorized calculations
        close_neighbors = neighbor_ids[close_mask]
        close_distances = distances[close_mask]
        
        # Position differences
        dx = particles.position_x[i] - particles.position_x[close_neighbors]
        dy = particles.position_y[i] - particles.position_y[close_neighbors]
        
        # Equilibrium distance
        r_eq = 0.5 * h_i
        
        # Spring-like cohesive force (only attractive)
        force_mags = -cohesion_strength * (close_distances - r_eq)
        force_mags[close_distances < r_eq] = 0.0  # No repulsion
        
        # Normalize and apply
        force_mags = force_mags / close_distances
        fx = force_mags * dx * particles.mass[close_neighbors]
        fy = force_mags * dy * particles.mass[close_neighbors]
        
        # Sum forces
        particles.force_x[i] += np.sum(fx)
        particles.force_y[i] += np.sum(fy)