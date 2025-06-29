"""
Short-range repulsion forces to prevent particle interpenetration.

Implements strong repulsive forces when particles get too close,
especially important for preventing light particles (air) from
penetrating dense particle arrangements (rock).
"""

import numpy as np
from ..core.particles import ParticleArrays
from .materials import MaterialDatabase, MaterialType


def compute_repulsion_forces(
    particles: ParticleArrays,
    n_active: int,
    material_db: MaterialDatabase,
    repulsion_distance: float = 0.5,  # Fraction of smoothing length
    repulsion_strength: float = 1e6   # Strong repulsion
):
    """
    Add strong short-range repulsion to prevent interpenetration.
    
    This is especially important for:
    - Preventing air particles from entering rock/water regions
    - Maintaining proper particle spacing
    - Preventing numerical instabilities from overlapping particles
    
    Args:
        particles: Particle arrays with neighbor information
        n_active: Number of active particles
        material_db: Material property database
        repulsion_distance: Distance below which repulsion activates (fraction of h)
        repulsion_strength: Strength of repulsive force
    """
    
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
            
        # Get particle properties
        h_i = particles.smoothing_h[i]
        mat_i = particles.material_id[i]
        mass_i = particles.mass[i]
        
        # Material-specific repulsion distance
        # Use smaller distance for same material, larger for different materials
        
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        distances = particles.neighbor_distances[i, neighbor_slice]
        
        # Find particles that are too close
        for j_idx, j in enumerate(neighbor_ids):
            r = distances[j_idx]
            mat_j = particles.material_id[j]
            h_j = particles.smoothing_h[j]
            mass_j = particles.mass[j]
            
            # Average smoothing length
            h_avg = 0.5 * (h_i + h_j)
            
            # Determine repulsion distance based on material combination
            if mat_i == mat_j:
                # Same material - allow closer approach
                r_repel = repulsion_distance * h_avg * 0.8
            else:
                # Different materials - stronger separation
                # Especially important for air-solid interfaces
                if (MaterialType(mat_i) == MaterialType.AIR or 
                    MaterialType(mat_j) == MaterialType.AIR):
                    r_repel = repulsion_distance * h_avg * 1.5  # Extra separation for air
                else:
                    r_repel = repulsion_distance * h_avg * 1.2
            
            # Apply repulsion if too close
            if r < r_repel and r > 1e-6:
                # Exponential repulsion: F = A * exp(-r/r0)
                # This gives very strong force at small distances
                r0 = r_repel * 0.3  # Characteristic length scale
                
                # Material-based repulsion strength
                # Stronger repulsion for air trying to enter solids
                if (MaterialType(mat_i) == MaterialType.AIR and 
                    MaterialType(mat_j) in [MaterialType.ROCK, MaterialType.URANIUM]):
                    A = repulsion_strength * 10.0  # 10x stronger
                elif (MaterialType(mat_j) == MaterialType.AIR and 
                      MaterialType(mat_i) in [MaterialType.ROCK, MaterialType.URANIUM]):
                    A = repulsion_strength * 10.0
                else:
                    A = repulsion_strength
                
                # Exponential force magnitude
                force_mag = A * np.exp(-r / r0) / r0
                
                # Apply force along line of centers (repulsive)
                dx = particles.position_x[i] - particles.position_x[j]
                dy = particles.position_y[i] - particles.position_y[j]
                
                fx = force_mag * dx / r
                fy = force_mag * dy / r
                
                # Apply with mass weighting
                mass_factor = mass_j / (mass_i + mass_j)
                
                particles.force_x[i] += fx * mass_factor
                particles.force_y[i] += fy * mass_factor
                
                # Newton's third law
                particles.force_x[j] -= fx * (1.0 - mass_factor)
                particles.force_y[j] -= fy * (1.0 - mass_factor)


def compute_boundary_force(
    particles: ParticleArrays,
    n_active: int,
    material_db: MaterialDatabase,
    interface_strength: float = 1e5
):
    """
    Add interface tension forces between different materials.
    
    This creates an effective surface tension that helps prevent
    material mixing, especially at air-liquid and air-solid interfaces.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        material_db: Material database
        interface_strength: Strength of interface forces
    """
    
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
            
        mat_i = MaterialType(particles.material_id[i])
        
        # Count neighbors of same vs different materials
        same_material_count = 0
        different_material_count = 0
        
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        
        # Compute material gradient direction
        gradient_x = 0.0
        gradient_y = 0.0
        
        for j_idx, j in enumerate(neighbor_ids):
            mat_j = MaterialType(particles.material_id[j])
            
            if mat_i == mat_j:
                same_material_count += 1
            else:
                different_material_count += 1
                
                # Add to gradient - points away from different materials
                dx = particles.position_x[i] - particles.position_x[j]
                dy = particles.position_y[i] - particles.position_y[j]
                r = particles.neighbor_distances[i, j_idx]
                
                if r > 1e-6:
                    gradient_x += dx / r
                    gradient_y += dy / r
        
        # If at interface (has neighbors of different materials)
        if different_material_count > 0:
            # Normalize gradient
            grad_mag = np.sqrt(gradient_x**2 + gradient_y**2)
            if grad_mag > 1e-6:
                gradient_x /= grad_mag
                gradient_y /= grad_mag
                
                # Interface force proportional to fraction of different neighbors
                interface_fraction = different_material_count / n_neighbors
                
                # Special handling for air interfaces
                if mat_i == MaterialType.AIR:
                    # Air particles pushed away from solids/liquids
                    force_mag = interface_strength * interface_fraction * 2.0
                else:
                    # Other materials have moderate interface tension
                    force_mag = interface_strength * interface_fraction
                
                # Apply force in gradient direction (away from different materials)
                particles.force_x[i] += force_mag * gradient_x
                particles.force_y[i] += force_mag * gradient_y