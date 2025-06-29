"""
Cohesive forces for SPH particles.

Implements short-range attractive forces between particles to simulate
material cohesion, allowing solids to hold together.
"""

import numpy as np
from typing import Optional
from ..core.particles import ParticleArrays
from ..core.kernel_vectorized import CubicSplineKernel
from .materials import MaterialDatabase, MaterialType


def compute_cohesive_forces(
    particles: ParticleArrays,
    kernel: CubicSplineKernel,
    n_active: int,
    material_db: MaterialDatabase,
    cutoff_factor: float = 1.5,
    temperature_softening: bool = True
):
    """
    Add cohesive forces between particles.
    
    Cohesive forces provide short-range attraction between particles of
    compatible materials, allowing solids to hold together. The force
    is attractive at intermediate distances and becomes repulsive at
    very short distances to prevent particle overlap.
    
    Args:
        particles: Particle arrays with neighbor information
        kernel: SPH kernel function
        n_active: Number of active particles
        material_db: Material property database
        cutoff_factor: Cutoff distance as fraction of smoothing length
        temperature_softening: Reduce cohesion at high temperatures
    """
    # Get material properties for all particles
    max_particles = len(particles.position_x)
    cohesion_strengths = np.zeros(max_particles, dtype=np.float32)
    melting_points = np.zeros(max_particles, dtype=np.float32)
    
    for i in range(n_active):
        mat_type = MaterialType(particles.material_id[i])
        mat_props = material_db.get_properties(mat_type)
        
        # Use cohesion_strength if available, otherwise 0
        if hasattr(mat_props, 'cohesion_strength') and mat_props.cohesion_strength is not None:
            cohesion_strengths[i] = mat_props.cohesion_strength
        
        # Get melting point for temperature softening
        if hasattr(mat_props, 'melting_point') and mat_props.melting_point is not None:
            melting_points[i] = mat_props.melting_point
        else:
            melting_points[i] = 1e6  # Very high default
    
    # Process each particle
    for i in range(n_active):
        if cohesion_strengths[i] <= 0:
            continue  # No cohesion for this material
            
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        # Get neighbor data
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        distances = particles.neighbor_distances[i, neighbor_slice]
        
        # Skip particles that are too far
        h_i = particles.smoothing_h[i]
        close_mask = distances < (cutoff_factor * h_i)
        if not np.any(close_mask):
            continue
        
        # Work with close neighbors only
        close_neighbors = neighbor_ids[close_mask]
        close_distances = distances[close_mask]
        
        # Position differences
        dx = particles.position_x[i] - particles.position_x[close_neighbors]
        dy = particles.position_y[i] - particles.position_y[close_neighbors]
        
        # Compute cohesive force for each neighbor
        for j_idx, j in enumerate(close_neighbors):
            # Skip if neighbor has no cohesion
            if cohesion_strengths[j] <= 0:
                continue
            
            # Use average cohesion strength
            cohesion = 0.5 * (cohesion_strengths[i] + cohesion_strengths[j])
            
            # Material compatibility factor
            # Same material: full cohesion
            # Different materials: reduced cohesion
            if particles.material_id[i] == particles.material_id[j]:
                compatibility = 1.0
            else:
                # Different material cohesion rules
                mat_i = MaterialType(particles.material_id[i])
                mat_j = MaterialType(particles.material_id[j])
                
                # Special cases for material pairs
                if {mat_i, mat_j} == {MaterialType.ROCK, MaterialType.URANIUM}:
                    compatibility = 0.8  # Rock and uranium bond well
                elif {mat_i, mat_j} == {MaterialType.WATER, MaterialType.ICE}:
                    compatibility = 0.5  # Water-ice moderate bonding
                elif mat_i in [MaterialType.AIR, MaterialType.WATER_VAPOR] or \
                     mat_j in [MaterialType.AIR, MaterialType.WATER_VAPOR]:
                    compatibility = 0.0  # Gases don't bond with solids
                else:
                    compatibility = 0.1  # Weak bonding between different materials
            
            cohesion *= compatibility
            
            if cohesion <= 0:
                continue
            
            # Temperature softening
            if temperature_softening:
                # Reduce cohesion as temperature approaches melting point
                T_i = particles.temperature[i]
                T_j = particles.temperature[j]
                T_melt_i = melting_points[i]
                T_melt_j = melting_points[j]
                
                # Temperature factor (1.0 at T=0, 0.0 at T=T_melt)
                temp_factor_i = max(0.0, 1.0 - T_i / T_melt_i)
                temp_factor_j = max(0.0, 1.0 - T_j / T_melt_j)
                temp_factor = 0.5 * (temp_factor_i + temp_factor_j)
                
                cohesion *= temp_factor
            
            # Distance and direction
            r = close_distances[j_idx]
            if r < 1e-6:
                continue  # Skip overlapping particles
            
            rx = dx[j_idx] / r
            ry = dy[j_idx] / r
            
            # Cohesive force profile
            # Uses a Lennard-Jones-like potential:
            # - Repulsive at very short range (r < r_min)
            # - Attractive at intermediate range (r_min < r < r_cut)
            # - Zero beyond cutoff
            
            # Reference distances
            h_avg = 0.5 * (particles.smoothing_h[i] + particles.smoothing_h[j])
            r_min = 0.8 * h_avg  # Equilibrium distance
            r_cut = cutoff_factor * h_avg  # Cutoff distance
            
            if r < r_min:
                # Short-range repulsion to prevent overlap
                # This helps maintain particle spacing
                force_mag = cohesion * (1.0 - r / r_min) / r_min
                # Repulsive: positive force along r direction
                fx = force_mag * rx
                fy = force_mag * ry
            else:
                # Attractive cohesion
                # Linear decrease from r_min to r_cut
                force_mag = cohesion * (1.0 - (r - r_min) / (r_cut - r_min)) / r_min
                # Attractive: negative force along r direction
                fx = -force_mag * rx
                fy = -force_mag * ry
            
            # Apply equal and opposite forces
            mass_factor = particles.mass[j] / (particles.mass[i] + particles.mass[j])
            
            particles.force_x[i] += fx * mass_factor
            particles.force_y[i] += fy * mass_factor
            
            # Newton's third law
            particles.force_x[j] -= fx * (1.0 - mass_factor)
            particles.force_y[j] -= fy * (1.0 - mass_factor)


def compute_cohesive_forces_simple(
    particles: ParticleArrays,
    n_active: int,
    cohesion_strength: float = 1e5,
    cutoff_factor: float = 1.5
):
    """
    Simplified cohesive force implementation with uniform strength.
    
    Good for testing and scenarios where all materials have similar cohesion.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        cohesion_strength: Uniform cohesion strength for all materials
        cutoff_factor: Cutoff distance as fraction of smoothing length
    """
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        # Get neighbor data
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        distances = particles.neighbor_distances[i, neighbor_slice]
        
        # Skip distant particles
        h_i = particles.smoothing_h[i]
        close_mask = distances < (cutoff_factor * h_i)
        if not np.any(close_mask):
            continue
        
        close_neighbors = neighbor_ids[close_mask]
        close_distances = distances[close_mask]
        
        # Position differences
        dx = particles.position_x[i] - particles.position_x[close_neighbors]
        dy = particles.position_y[i] - particles.position_y[close_neighbors]
        
        # Equilibrium distance (fraction of smoothing length)
        # Use smaller value to ensure particles at normal spacing experience attraction
        r_eq = 0.5 * h_i
        
        # Spring-like cohesive force
        # F = -k * (r - r_eq) * r_hat
        # When r > r_eq: force is negative (attractive)
        # When r < r_eq: force is positive (repulsive)
        force_mags = -cohesion_strength * (close_distances - r_eq)
        
        # Only attractive forces when particles are farther than equilibrium
        # Set repulsive forces (when r < r_eq) to zero
        force_mags[close_distances < r_eq] = 0.0
        
        # Normalize by distance for direction
        force_mags = force_mags / close_distances
        
        # Force components (force_mags already normalized by distance)
        fx = force_mags * dx
        fy = force_mags * dy
        
        # Apply forces
        particles.force_x[i] += np.sum(fx * particles.mass[close_neighbors])
        particles.force_y[i] += np.sum(fy * particles.mass[close_neighbors])