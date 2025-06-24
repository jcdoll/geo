"""
Solar heating using proper material absorption coefficients with safety checks.

This implementation uses the absorption_coeff property from the material
database, ensuring that space (with absorption=0) naturally doesn't heat up.
"""

import numpy as np
from materials import MaterialType, MaterialDatabase

# Create material database instance
material_db = MaterialDatabase()


def apply_solar_heating_proper(state, dt: float, solar_angle: float, solar_constant: float):
    """
    Apply solar heating using material absorption coefficients.
    
    Args:
        state: FluxState instance
        dt: Time step in seconds
        solar_angle: Angle of sun from vertical (radians)
        solar_constant: Solar irradiance (W/m²)
    """
    # Solar direction
    sun_x = np.sin(solar_angle)
    sun_y = np.cos(solar_angle)
    
    # Skip if sun is below horizon
    if sun_y < 0:
        return
    
    # Pre-compute effective absorption for each cell
    ny, nx = state.temperature.shape
    effective_absorption = np.zeros((ny, nx))
    
    # For each cell, compute weighted absorption based on material fractions
    for j in range(ny):
        for i in range(nx):
            absorption = 0.0
            albedo = 0.0
            
            # Weight by volume fraction
            for mat_idx in range(state.n_materials):
                vol_frac = state.vol_frac[mat_idx, j, i]
                if vol_frac > 0:
                    mat_type = MaterialType(mat_idx)
                    mat_props = material_db.get_properties(mat_type)
                    
                    # Use actual material absorption coefficient
                    absorption += vol_frac * mat_props.absorption_coeff
                    albedo += vol_frac * mat_props.albedo
            
            # Clamp values to valid ranges
            absorption = np.clip(absorption, 0.0, 1.0)
            albedo = np.clip(albedo, 0.0, 1.0)
            
            # Effective absorption (accounting for albedo)
            # Space will have absorption = 0, so effective_absorption = 0
            effective_absorption[j, i] = absorption * (1.0 - albedo)
    
    # Ensure no NaN or extreme values
    effective_absorption = np.nan_to_num(effective_absorption, nan=0.0, posinf=1.0, neginf=0.0)
    effective_absorption = np.clip(effective_absorption, 0.0, 1.0)
    
    # No artificial solar constant limits  # Max 5x normal solar constant
    
    # For vertical sun, use optimized column processing
    if abs(sun_x) < 0.1:
        _apply_vertical_rays_safe(state, dt, effective_absorption, solar_constant)
    else:
        # Angled sun: trace rays properly
        _apply_angled_rays_safe(state, dt, effective_absorption, solar_constant, sun_x, sun_y)


def _apply_vertical_rays_safe(state, dt, effective_absorption, solar_constant):
    """
    Apply solar heating for vertical rays with safety checks.
    
    Process each column from top to bottom, with rays attenuating
    as they pass through materials.
    """
    ny, nx = state.temperature.shape
    
    # No artificial temperature limits
    
    # Process each column
    for i in range(nx):
        # Start with full solar intensity at top
        intensity = solar_constant
        
        # March down the column
        for j in range(ny):
            # Only process if there's still significant intensity
            if intensity < 1e-6:
                break
            
            # Ensure absorption is in valid range
            eff_abs = np.clip(effective_absorption[j, i], 0.0, 1.0)
            
            # Energy absorbed in this cell
            absorbed = intensity * eff_abs
            
            # Safety check on absorbed energy
            if not np.isfinite(absorbed) or absorbed < 0:
                absorbed = 0.0
            
            if absorbed > 0 and state.cell_depth > 0:
                # Convert to volumetric power density (W/m³)
                volumetric_power = absorbed / state.cell_depth
                
                # Limit power density to prevent runaway
                max_power = 1e9  # 1 GW/m³ is extreme
                volumetric_power = np.clip(volumetric_power, 0.0, max_power)
                
                # Update power density tracking
                state.power_density[j, i] += volumetric_power
                
                # Temperature change
                if state.density[j, i] > 0 and state.specific_heat[j, i] > 0:
                    dT = volumetric_power * dt / (state.density[j, i] * state.specific_heat[j, i])
                    
                    # No artificial temperature change limits
                    
                    # Check for NaN before updating
                    if np.isfinite(dT):
                        state.temperature[j, i] += dT
            
            # Attenuate intensity for next cell
            # If effective_absorption is 0 (space), intensity passes through unchanged
            intensity *= (1.0 - eff_abs)
            
            # Ensure intensity doesn't go negative or NaN
            if not np.isfinite(intensity) or intensity < 0:
                intensity = 0.0


def _apply_angled_rays_safe(state, dt, effective_absorption, solar_constant, sun_x, sun_y):
    """
    Apply solar heating for angled rays using DDA ray marching with safety checks.
    
    Traces rays from sun direction through the grid.
    """
    ny, nx = state.temperature.shape
    
    # No artificial temperature limits
    
    # Ray direction (pointing towards sun)
    ray_x = -sun_x
    ray_y = -sun_y
    
    # Determine starting positions based on sun angle
    if sun_y > 0:  # Sun above horizon
        # Start from top edge
        start_positions = [(0, i) for i in range(nx)]
    else:
        return  # Sun below horizon
    
    # Additional starting positions for angled sun
    if abs(sun_x) > 0.1:
        if sun_x > 0:  # Sun from right
            start_positions.extend([(j, nx-1) for j in range(1, ny)])
        else:  # Sun from left
            start_positions.extend([(j, 0) for j in range(1, ny)])
    
    # Trace rays using DDA
    for start_j, start_i in start_positions:
        # Current position
        x = float(start_i) + 0.5
        y = float(start_j) + 0.5
        
        # Initial intensity
        intensity = solar_constant
        
        # Step sizes for DDA
        if abs(ray_x) > abs(ray_y):
            t_step = 1.0 / abs(ray_x)
            x_step = 1.0 if ray_x > 0 else -1.0
            y_step = ray_y / abs(ray_x)
        else:
            t_step = 1.0 / abs(ray_y)
            y_step = 1.0 if ray_y > 0 else -1.0
            x_step = ray_x / abs(ray_y)
        
        # March along ray
        steps = 0
        max_steps = 2 * max(ny, nx)  # Prevent infinite loops
        
        while steps < max_steps:
            steps += 1
            
            # Current cell
            i = int(x)
            j = int(y)
            
            # Check bounds
            if i < 0 or i >= nx or j < 0 or j >= ny:
                break
            
            # Only process if there's still significant intensity
            if intensity < 1e-6:
                break
            
            # Ensure absorption is in valid range
            eff_abs = np.clip(effective_absorption[j, i], 0.0, 1.0)
            
            # Energy absorbed in this cell
            absorbed = intensity * eff_abs * t_step
            
            # Safety check on absorbed energy
            if not np.isfinite(absorbed) or absorbed < 0:
                absorbed = 0.0
            
            if absorbed > 0 and state.cell_depth > 0:
                # Convert to volumetric power density
                volumetric_power = absorbed / state.cell_depth
                
                # No artificial power density limits
                
                # Update power density
                state.power_density[j, i] += volumetric_power
                
                # Temperature change
                if state.density[j, i] > 0 and state.specific_heat[j, i] > 0:
                    dT = volumetric_power * dt / (state.density[j, i] * state.specific_heat[j, i])
                    
                    # No artificial temperature change limits
                    
                    # Check for NaN before updating
                    if np.isfinite(dT):
                        state.temperature[j, i] += dT
            
            # Attenuate intensity
            intensity *= (1.0 - eff_abs * t_step)
            
            # Ensure intensity doesn't go negative or NaN
            if not np.isfinite(intensity) or intensity < 0:
                intensity = 0.0
            
            # Step to next cell
            x += x_step
            y += y_step