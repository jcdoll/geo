"""
Proper solar heating implementation using material absorption properties.
Space naturally doesn't absorb energy due to zero absorption coefficient.
"""

import numpy as np
from materials import MaterialType, MaterialDatabase


def apply_solar_heating_proper(state, dt: float, solar_angle: float, solar_constant: float = 1361.0):
    """
    Apply solar heating using proper material absorption coefficients.
    
    Space has zero absorption, so it naturally doesn't heat up.
    Rays continue through transparent materials until hitting opaque surfaces.
    
    Args:
        state: FluxState instance
        dt: Time step in seconds
        solar_angle: Angle of sun from vertical (radians)
        solar_constant: Solar irradiance (W/m²)
    """
    # Solar direction
    sun_x = np.sin(solar_angle)
    sun_y = np.cos(solar_angle)
    
    # Only apply heating when sun is above horizon
    if sun_y <= 0:
        return
    
    # Get material database
    mat_db = MaterialDatabase()
    
    # Create absorption coefficient array from material properties
    # This properly uses the absorption_coeff from each material
    absorption = np.zeros_like(state.density)
    albedo = np.zeros_like(state.density)
    
    for mat_type in MaterialType:
        mat_idx = mat_type.value
        if mat_idx < state.n_materials:
            vol_frac = state.vol_frac[mat_idx]
            mat_props = mat_db.get_properties(mat_type)
            
            # Use actual material absorption coefficient
            absorption += vol_frac * mat_props.absorption_coeff
            albedo += vol_frac * mat_props.albedo
    
    # Effective absorption (accounting for albedo)
    # Space will have absorption = 0, so effective_absorption = 0
    effective_absorption = absorption * (1.0 - albedo)
    
    # For vertical sun, use optimized column processing
    if abs(sun_x) < 0.1:
        _apply_vertical_rays(state, dt, effective_absorption, solar_constant)
    else:
        # Angled sun: trace rays properly
        _apply_angled_rays(state, dt, effective_absorption, solar_constant, sun_x, sun_y)


def _apply_vertical_rays(state, dt, effective_absorption, solar_constant):
    """
    Apply solar heating for vertical rays.
    
    Process each column from top to bottom, with rays attenuating
    as they pass through materials.
    """
    ny, nx = state.temperature.shape
    
    # Process each column
    for i in range(nx):
        # Start with full solar intensity at top
        intensity = solar_constant
        
        # March down the column
        for j in range(ny):
            # Only process if there's still significant intensity
            if intensity < 1e-6:
                break
            
            # Energy absorbed in this cell
            absorbed = intensity * effective_absorption[j, i]
            
            if absorbed > 0:
                # Convert to volumetric power density (W/m³)
                volumetric_power = absorbed / state.cell_depth
                
                # Update power density tracking
                state.power_density[j, i] += volumetric_power
                
                # Temperature change
                # No need to check density - if absorption is 0 (like space), no heating occurs
                if state.density[j, i] > 0 and state.specific_heat[j, i] > 0:
                    dT = volumetric_power * dt / (state.density[j, i] * state.specific_heat[j, i])
                    state.temperature[j, i] += dT
            
            # Attenuate intensity for next cell
            # If effective_absorption is 0 (space), intensity passes through unchanged
            intensity *= (1.0 - effective_absorption[j, i])


def _apply_angled_rays(state, dt, effective_absorption, solar_constant, sun_x, sun_y):
    """
    Apply solar heating for angled rays using DDA ray marching.
    """
    ny, nx = state.temperature.shape
    
    # Determine which boundary rays enter from
    if sun_y > 0:  # Sun above horizon
        # Spawn rays from top boundary
        for i in range(nx):
            _trace_solar_ray(state, i, 0, sun_x, sun_y, dt, effective_absorption, solar_constant)
    
    # Also spawn rays from side boundaries if sun is at an angle
    if abs(sun_x) > 0.1:
        if sun_x > 0:  # Sun from right
            for j in range(ny):
                _trace_solar_ray(state, nx-1, j, sun_x, sun_y, dt, effective_absorption, solar_constant)
        else:  # Sun from left
            for j in range(ny):
                _trace_solar_ray(state, 0, j, sun_x, sun_y, dt, effective_absorption, solar_constant)


def _trace_solar_ray(state, start_x: int, start_y: int, sun_x: float, sun_y: float, 
                     dt: float, effective_absorption: np.ndarray, solar_constant: float):
    """
    Trace a single solar ray using DDA algorithm.
    
    Args:
        start_x, start_y: Starting position
        sun_x, sun_y: Sun direction (normalized)
        dt: Time step
        effective_absorption: Pre-computed absorption array
        solar_constant: Initial ray intensity
    """
    # Initial intensity
    intensity = solar_constant
    
    # Ray direction (opposite of sun direction - rays go from sun to ground)
    dx = -sun_x
    dy = -sun_y
    
    # Current position
    x, y = float(start_x), float(start_y)
    
    # DDA parameters
    if abs(dx) > abs(dy):
        # X-major
        step_x = 1.0 if dx > 0 else -1.0
        step_y = dy / abs(dx) if abs(dx) > 0 else 0
    else:
        # Y-major
        step_y = 1.0 if dy > 0 else -1.0
        step_x = dx / abs(dy) if abs(dy) > 0 else 0
    
    # March along ray
    while 0 <= x < state.nx and 0 <= y < state.ny and intensity > 1e-6:
        ix, iy = int(x), int(y)
        
        # Energy absorbed in this cell
        absorbed = intensity * effective_absorption[iy, ix]
        
        if absorbed > 0:
            # Convert to volumetric power density
            volumetric_power = absorbed / state.cell_depth
            
            # Update power density
            state.power_density[iy, ix] += volumetric_power
            
            # Temperature change
            if state.density[iy, ix] > 0 and state.specific_heat[iy, ix] > 0:
                dT = volumetric_power * dt / (state.density[iy, ix] * state.specific_heat[iy, ix])
                state.temperature[iy, ix] += dT
        
        # Attenuate ray
        intensity *= (1.0 - effective_absorption[iy, ix])
        
        # Stop if we hit opaque material (high absorption)
        if effective_absorption[iy, ix] > 0.99:
            break
        
        # Step to next cell
        x += step_x
        y += step_y