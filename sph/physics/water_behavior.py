"""
Special handling for water behavior in SPH simulations.

Handles:
- Water in vacuum/low pressure (evaporation)
- Surface tension approximation
- Water-air interface
"""

import numpy as np
from ..core.particles import ParticleArrays
from ..physics.materials import MaterialType, MaterialDatabase


def handle_water_in_vacuum(particles: ParticleArrays, n_active: int,
                          material_db: MaterialDatabase,
                          dt: float,
                          evaporation_rate: float = 0.1) -> int:
    """
    Handle water particles in vacuum or very low pressure environments.
    
    Water exposed to vacuum should evaporate (convert to water vapor).
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        material_db: Material database
        dt: Time step
        evaporation_rate: Rate of evaporation (fraction per second)
        
    Returns:
        Number of particles that evaporated
    """
    n_evaporated = 0
    
    for i in range(n_active):
        # Only process water particles
        if particles.material_id[i] != MaterialType.WATER.value:
            continue
            
        # Check if particle is in low pressure environment
        # Look at neighboring particles
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            # No neighbors - definitely in vacuum
            should_evaporate = True
        else:
            # Check neighbor materials
            neighbor_ids = particles.neighbor_ids[i, :n_neighbors]
            neighbor_materials = particles.material_id[neighbor_ids]
            
            # Count non-liquid neighbors
            n_gas_neighbors = np.sum(
                (neighbor_materials == MaterialType.AIR.value) |
                (neighbor_materials == MaterialType.SPACE.value) |
                (neighbor_materials == MaterialType.WATER_VAPOR.value)
            )
            
            # If mostly surrounded by gas/vacuum, evaporate
            should_evaporate = n_gas_neighbors > n_neighbors * 0.7
            
        if should_evaporate:
            # Probability of evaporation
            if np.random.random() < evaporation_rate * dt:
                # Convert to water vapor
                particles.material_id[i] = MaterialType.WATER_VAPOR.value
                
                # Update properties
                vapor_props = material_db.get_properties(MaterialType.WATER_VAPOR)
                water_props = material_db.get_properties(MaterialType.WATER)
                
                # Conserve mass but update density
                particles.density[i] *= vapor_props.density_ref / water_props.density_ref
                
                # Add thermal energy (cooling effect)
                particles.temperature[i] -= 50.0  # Evaporative cooling
                
                n_evaporated += 1
    
    return n_evaporated


def add_water_cohesion_forces(particles: ParticleArrays, n_active: int,
                            cohesion_strength: float = 0.5) -> None:
    """
    Add cohesive forces between water particles to simulate surface tension.
    
    This is a simplified model that adds attractive forces between water
    particles that are close but not too close.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles  
        cohesion_strength: Strength of cohesive forces
    """
    for i in range(n_active):
        # Only water particles have cohesion
        if particles.material_id[i] != MaterialType.WATER.value:
            continue
            
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
        
        # Check for water neighbors
        water_neighbors = particles.material_id[neighbor_ids] == MaterialType.WATER.value
        if not np.any(water_neighbors):
            continue
            
        # Cohesive force for water-water interactions
        # Attractive at medium range, neutral at equilibrium
        h_i = particles.smoothing_h[i]
        equilibrium_dist = 0.8 * h_i
        
        # Only attract particles that are separated
        attracting = (distances > equilibrium_dist) & (distances < 1.5 * h_i) & water_neighbors
        
        if np.any(attracting):
            attract_distances = distances[attracting]
            attract_dx = dx[attracting]
            attract_dy = dy[attracting]
            
            # Normalized directions
            dx_norm = attract_dx / attract_distances
            dy_norm = attract_dy / attract_distances
            
            # Cohesive force magnitude (decreases with distance)
            force_magnitude = cohesion_strength * particles.mass[i] * \
                            (1.0 - (attract_distances - equilibrium_dist) / (0.7 * h_i))
            
            # Apply forces
            fx_cohesive = -force_magnitude * dx_norm  # Negative because attractive
            fy_cohesive = -force_magnitude * dy_norm
            
            particles.force_x[i] += np.sum(fx_cohesive)
            particles.force_y[i] += np.sum(fy_cohesive)
            
            # Newton's third law
            attract_neighbors = neighbor_ids[attracting]
            for j, nid in enumerate(attract_neighbors):
                particles.force_x[nid] -= fx_cohesive[j]
                particles.force_y[nid] -= fy_cohesive[j]


def adjust_water_pressure_near_gas(particles: ParticleArrays, n_active: int) -> None:
    """
    Adjust water pressure near gas interfaces to prevent explosive expansion.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
    """
    for i in range(n_active):
        if particles.material_id[i] != MaterialType.WATER.value:
            continue
            
        # Check neighbors
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
            
        neighbor_ids = particles.neighbor_ids[i, :n_neighbors]
        neighbor_materials = particles.material_id[neighbor_ids]
        
        # Count gas neighbors
        n_gas = np.sum(
            (neighbor_materials == MaterialType.AIR.value) |
            (neighbor_materials == MaterialType.SPACE.value) |
            (neighbor_materials == MaterialType.WATER_VAPOR.value)
        )
        
        # If near gas interface, reduce pressure
        if n_gas > 0:
            gas_fraction = n_gas / n_neighbors
            # Reduce pressure near interface
            particles.pressure[i] *= (1.0 - 0.5 * gas_fraction)