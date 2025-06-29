"""
Improved pressure calculation for mixed material systems.

Handles the extreme density ratios between air and rock/water
by using material-appropriate equations of state.
"""

import numpy as np
from ..core.particles import ParticleArrays
from .materials import MaterialDatabase, MaterialType


def compute_pressure_mixed(
    particles: ParticleArrays,
    material_db: MaterialDatabase,
    n_active: int,
    background_pressure: float = 101325.0  # 1 atm
):
    """
    Compute pressure with proper handling of mixed materials.
    
    Key features:
    - Ideal gas law for gases (air, water vapor)
    - Modified Tait equation for liquids/solids
    - Background pressure to prevent vacuum collapse
    - Smooth transitions at material interfaces
    
    Args:
        particles: Particle arrays
        material_db: Material database
        n_active: Number of active particles
        background_pressure: Ambient pressure (Pa)
    """
    
    # Constants
    R_air = 287.0      # Specific gas constant for air [J/(kg·K)]
    R_vapor = 461.5    # Specific gas constant for water vapor [J/(kg·K)]
    
    # Reset pressure to background
    particles.pressure[:n_active] = background_pressure
    
    # Process each material type separately for efficiency
    for mat_type in MaterialType:
        mask = particles.material_id[:n_active] == mat_type.value
        if not np.any(mask):
            continue
            
        indices = np.where(mask)[0]
        mat_props = material_db.get_properties(mat_type)
        
        if mat_type == MaterialType.AIR:
            # Ideal gas law: P = ρRT
            # But with limits to prevent extreme pressures
            density = particles.density[indices]
            temperature = particles.temperature[indices]
            
            # Limit density to reasonable range for air
            # This prevents huge pressures from SPH density artifacts
            density_clamped = np.clip(density, 0.1, 10.0)  # 0.1 to 10 kg/m³
            
            # Ideal gas pressure
            pressure = density_clamped * R_air * temperature
            
            # Add background pressure and limit total
            particles.pressure[indices] = np.clip(
                pressure + background_pressure,
                0.5 * background_pressure,  # Min: 0.5 atm
                2.0 * background_pressure   # Max: 2 atm
            )
            
        elif mat_type == MaterialType.WATER_VAPOR:
            # Similar to air but different gas constant
            density = particles.density[indices]
            temperature = particles.temperature[indices]
            density_clamped = np.clip(density, 0.01, 1.0)
            
            pressure = density_clamped * R_vapor * temperature
            particles.pressure[indices] = pressure + background_pressure
            
        elif mat_type == MaterialType.SPACE:
            # Near vacuum
            particles.pressure[indices] = 0.01 * background_pressure
            
        else:
            # Liquids and solids use modified Tait equation
            density = particles.density[indices]
            density_ref = mat_props.density_ref
            
            # Get appropriate bulk modulus
            # Reduce it significantly for numerical stability
            if mat_type == MaterialType.WATER:
                B = mat_props.bulk_modulus * 0.001  # Very soft water
                gamma = 7.0
                density_ratio_max = 1.1  # Water barely compressible
            else:
                # Solids (rock, uranium, etc)
                B = mat_props.bulk_modulus * 0.0001  # Very soft for stability
                gamma = 3.0  # Lower exponent
                density_ratio_max = 1.05  # Nearly incompressible
            
            # Compute density ratio with strict limits
            density_ratio = density / density_ref
            density_ratio = np.clip(density_ratio, 0.95, density_ratio_max)
            
            # Modified Tait equation
            pressure = B * (density_ratio**gamma - 1.0)
            
            # Add background pressure
            particles.pressure[indices] = pressure + background_pressure
    
    # Post-processing: smooth pressure at material interfaces
    # This helps prevent pressure discontinuities that cause instabilities
    smooth_interface_pressure(particles, n_active, background_pressure)


def smooth_interface_pressure(
    particles: ParticleArrays,
    n_active: int,
    background_pressure: float
):
    """
    Smooth pressure at material interfaces to prevent instabilities.
    
    When particles of very different materials (like air and rock) are neighbors,
    their pressures can be very different, causing large forces. This function
    applies smoothing at interfaces.
    """
    
    # Store original pressures
    original_pressure = particles.pressure[:n_active].copy()
    
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
            
        mat_i = particles.material_id[i]
        
        # Check if at interface
        at_interface = False
        pressure_sum = original_pressure[i]
        weight_sum = 1.0
        
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        
        for j in neighbor_ids:
            mat_j = particles.material_id[j]
            
            if mat_i != mat_j:
                at_interface = True
                
                # Different weighting for different material pairs
                if (MaterialType(mat_i) == MaterialType.AIR and 
                    MaterialType(mat_j) in [MaterialType.ROCK, MaterialType.URANIUM]):
                    # Air-solid interface: use mostly solid pressure
                    weight = 0.1
                elif (MaterialType(mat_j) == MaterialType.AIR and 
                      MaterialType(mat_i) in [MaterialType.ROCK, MaterialType.URANIUM]):
                    # Solid-air interface: keep solid pressure
                    weight = 0.0
                else:
                    # Other interfaces: moderate smoothing
                    weight = 0.3
                
                if weight > 0:
                    pressure_sum += original_pressure[j] * weight
                    weight_sum += weight
        
        # Apply smoothed pressure at interfaces
        if at_interface and weight_sum > 0:
            # Blend original and smoothed pressure
            smoothed = pressure_sum / weight_sum
            blend_factor = 0.5  # How much smoothing to apply
            
            particles.pressure[i] = (
                (1 - blend_factor) * original_pressure[i] + 
                blend_factor * smoothed
            )


def compute_artificial_pressure(
    particles: ParticleArrays,
    n_active: int,
    delta: float = 0.1
):
    """
    Add artificial pressure to prevent particle clumping.
    
    This is the "delta-SPH" approach that adds a small background pressure
    to prevent the tensile instability.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        delta: Artificial pressure factor (typically 0.1)
    """
    
    if delta <= 0:
        return
        
    # Compute reference pressure based on typical velocity
    v_max = 0.0
    for i in range(n_active):
        v = np.sqrt(particles.velocity_x[i]**2 + particles.velocity_y[i]**2)
        v_max = max(v_max, v)
    
    # Artificial pressure scale
    # P_art = delta * ρ * v_max²
    for i in range(n_active):
        p_artificial = delta * particles.density[i] * v_max * v_max
        particles.pressure[i] += p_artificial