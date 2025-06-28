"""
Solar heating using proper material absorption coefficients with safety checks.

This implementation uses the solar_absorption property from the material
database, ensuring that space (with absorption=0) naturally doesn't heat up.
"""

import numpy as np
from typing import TYPE_CHECKING

from materials import MaterialType, MaterialDatabase

if TYPE_CHECKING:
    from state import FluxState

# Create material database instance
material_db = MaterialDatabase()


def apply_solar_heating(state: "FluxState", dt: float, solar_angle: float, solar_constant: float) -> None:
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
    
    # Skip if sun is below horizon AND not coming from sides
    if sun_y < 0 and abs(sun_x) < 0.1:
        return
    
    # Pre-compute effective absorption for each cell (vectorized)
    ny, nx = state.temperature.shape
    
    # Create array for material solar absorption
    solar_absorptions = np.zeros(state.n_materials)
    
    for mat_idx in range(state.n_materials):
        mat_type = MaterialType(mat_idx)
        mat_props = material_db.get_properties(mat_type)
        solar_absorptions[mat_idx] = mat_props.solar_absorption
    
    # Compute weighted solar absorption using einsum for efficiency
    # vol_frac shape: (n_materials, ny, nx)
    # solar_absorptions shape: (n_materials,)
    # Result shape: (ny, nx)
    effective_absorption = np.einsum('myx,m->yx', state.vol_frac, solar_absorptions)
    
    # Ensure no NaN or extreme values
    effective_absorption = np.nan_to_num(effective_absorption, nan=0.0, posinf=1.0, neginf=0.0)
    effective_absorption = np.clip(effective_absorption, 0.0, 1.0)
    
    # Apply ray tracing for all sun angles
    _apply_ray_traced_heating(state, dt, effective_absorption, solar_constant, sun_x, sun_y)


def _apply_ray_traced_heating(state: "FluxState", dt: float, effective_absorption: np.ndarray, 
                             solar_constant: float, sun_x: float, sun_y: float) -> None:
    """
    Apply solar heating using DDA ray marching for all sun angles.
    
    Traces rays from sun direction through the grid.
    """
    ny, nx = state.temperature.shape
    
    # No artificial temperature limits
    
    # Ray direction (from sun towards planet center)
    # The sun vector (sun_x, sun_y) points from planet center to sun
    # Rays go in opposite direction: from sun to planet center
    # When sun is at angle θ: sun_x = sin(θ), sun_y = cos(θ)
    # For sun at east (π/2): sun_x = 1, sun_y = 0
    # Rays should go west (negative x), so ray_x should be -sun_x
    ray_x = -sun_x  # Opposite of sun vector direction
    ray_y = -sun_y  # Opposite of sun vector direction
    
    # Determine starting positions based on sun angle
    start_positions = []
    
    if sun_y > 0:  # Sun above horizon
        # Rays go downward (negative y), so start from top edge
        start_positions.extend([(0, i) for i in range(nx)])
    elif sun_y < -0.1:  # Sun below horizon (significant angle)
        # Rays go upward (positive y), so start from bottom edge
        start_positions.extend([(ny-1, i) for i in range(nx)])
    
    # For angled sun, add side edge positions
    if abs(sun_x) > 0.1:
        if sun_x > 0:  # Sun from east (right side)
            # Rays go west (left), so start from right edge
            start_positions.extend([(j, nx-1) for j in range(ny)])
        else:  # Sun from west (left side)
            # Rays go east (right), so start from left edge
            start_positions.extend([(j, 0) for j in range(ny)])
    
    # If no starting positions (sun below horizon and not from sides), return
    if not start_positions:
        return
    
    # Trace rays using DDA
    # IMPORTANT: DO NOT add special cases for vertical/horizontal rays!
    # The general DDA algorithm handles ALL cases correctly.
    # Special cases just add complexity and maintenance burden.
    for start_j, start_i in start_positions:
        # Current position - start at cell center
        x = float(start_i) + 0.5
        y = float(start_j) + 0.5
        
        # Small offset to ensure rays traverse cells properly
        # Without this, vertical rays starting at y=0.5 step to y=-0.5 immediately
        if abs(ray_y) > abs(ray_x):
            y = float(start_j) + 0.1 if ray_y < 0 else float(start_j) + 0.9
        
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
        
        while steps < max_steps and intensity > 1e-6:
            # Current cell - use floor to handle negative coordinates properly
            i = int(np.floor(x))
            j = int(np.floor(y))
            
            # Check bounds
            if i < 0 or i >= nx or j < 0 or j >= ny:
                break
            
            # Ensure absorption is in valid range
            eff_abs = np.clip(effective_absorption[j, i], 0.0, 1.0)
            
            # Energy absorbed in this cell
            # Simple absorption: fraction of incoming intensity
            absorbed = intensity * eff_abs
            
            # Safety check on absorbed energy
            if not np.isfinite(absorbed) or absorbed < 0:
                absorbed = 0.0
            
            if absorbed > 0:
                # Convert to volumetric power density
                # Use a realistic absorption depth instead of full cell depth
                # Most solar absorption happens in top 1-10m for rock/water
                absorption_depth = min(10.0, state.cell_depth)  # 10m max absorption depth
                volumetric_power = absorbed / absorption_depth
                
                # Update power density
                state.power_density[j, i] += volumetric_power
                
                # Temperature change
                if state.density[j, i] > 0 and state.specific_heat[j, i] > 0:
                    dT = volumetric_power * dt / (state.density[j, i] * state.specific_heat[j, i])
                    
                    # Check for NaN before updating
                    if np.isfinite(dT):
                        state.temperature[j, i] += dT
            
            # Attenuate intensity by the absorbed amount
            intensity -= absorbed
            
            # Ensure intensity doesn't go negative or NaN
            if not np.isfinite(intensity) or intensity < 0:
                intensity = 0.0
            
            # Step to next cell
            x += x_step
            y += y_step
            
            steps += 1