"""
Vectorized solar heating implementation using array operations instead of loops.
"""

import numpy as np
from materials import MaterialType


def apply_solar_heating_vectorized(state, dt: float, solar_angle: float, solar_constant: float = 1361.0):
    """
    Apply solar heating using vectorized ray tracing.
    
    Instead of tracing individual rays, we process entire columns/rows at once.
    
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
        
    # Material absorption coefficients
    absorption_coeffs = {
        MaterialType.AIR: 0.001,
        MaterialType.WATER_VAPOR: 0.005,
        MaterialType.WATER: 0.02,
        MaterialType.ICE: 0.01,
        MaterialType.SPACE: 0.0,
        MaterialType.ROCK: 1.0,
        MaterialType.SAND: 1.0,
        MaterialType.URANIUM: 1.0,
        MaterialType.MAGMA: 1.0,
    }
    
    # Create absorption coefficient array from volume fractions
    absorption = np.zeros_like(state.density)
    albedo = np.zeros_like(state.density)
    
    for mat_type in MaterialType:
        mat_idx = mat_type.value
        if mat_idx < state.n_materials:
            vol_frac = state.vol_frac[mat_idx]
            absorption += vol_frac * absorption_coeffs.get(mat_type, 0.1)
            
            # Get material albedo
            from materials import MaterialDatabase
            mat_db = MaterialDatabase()
            mat_props = mat_db.get_properties(mat_type)
            albedo += vol_frac * mat_props.albedo
    
    # Effective absorption (accounting for albedo)
    effective_absorption = absorption * (1.0 - albedo)
    
    # For vertical sun (sun_y ≈ 1), process columns from top
    if abs(sun_x) < 0.1:
        # Simple case: vertical rays
        _apply_vertical_rays_vectorized(state, dt, effective_absorption, solar_constant)
    else:
        # Angled sun: need to handle ray paths
        _apply_angled_rays_vectorized(state, dt, effective_absorption, solar_constant, sun_x, sun_y)


def _apply_vertical_rays_vectorized(state, dt, effective_absorption, solar_constant):
    """
    Apply solar heating for vertical rays (sun directly overhead).
    
    This is fully vectorized - process all columns simultaneously.
    """
    ny, nx = state.temperature.shape
    
    # Initialize intensity for all columns at top boundary
    intensity = np.ones(nx) * solar_constant
    
    # Process each row from top to bottom
    for j in range(ny):
        # Skip low density cells (space and sparse matter)
        # Space now has density 0.001 kg/m³, so use threshold above that
        non_space_mask = state.density[j, :] >= 1.0  # kg/m³
        
        if np.any(non_space_mask):
            # Energy absorbed in this row
            absorbed = intensity * effective_absorption[j, :] * non_space_mask
            
            # Convert to volumetric power density
            volumetric_power = absorbed / state.cell_depth
            
            # Update power density
            state.power_density[j, :] += volumetric_power
            
            # Temperature change
            # Only apply to cells with sufficient density to avoid extreme heating
            valid_cells = (state.density[j, :] > 1.0) & (state.specific_heat[j, :] > 0)
            if np.any(valid_cells):
                dT = np.zeros(nx)
                denominator = state.density[j, valid_cells] * state.specific_heat[j, valid_cells]
                # Additional safety check
                safe_cells = denominator > 1e-10
                if np.any(safe_cells):
                    valid_indices = np.where(valid_cells)[0][safe_cells]
                    dT[valid_indices] = volumetric_power[valid_indices] * dt / denominator[safe_cells]
                    state.temperature[j, :] += dT
            
            # Attenuate intensity for next row
            intensity *= (1.0 - effective_absorption[j, :] * non_space_mask)
            
            # Stop if intensity is negligible
            if np.max(intensity) < 1e-6:
                break


def _apply_angled_rays_vectorized(state, dt, effective_absorption, solar_constant, sun_x, sun_y):
    """
    Apply solar heating for angled rays.
    
    This uses a simplified approach: compute optical depth along ray paths
    and apply heating based on Beer-Lambert law.
    """
    ny, nx = state.temperature.shape
    
    # Determine primary direction
    if abs(sun_y) > abs(sun_x):
        # Primarily vertical - process from top with slight horizontal shift
        _apply_near_vertical_rays(state, dt, effective_absorption, solar_constant, sun_x, sun_y)
    else:
        # Primarily horizontal - process from side
        _apply_near_horizontal_rays(state, dt, effective_absorption, solar_constant, sun_x, sun_y)


def _apply_near_vertical_rays(state, dt, effective_absorption, solar_constant, sun_x, sun_y):
    """
    Handle rays that are mostly vertical with slight angle.
    """
    ny, nx = state.temperature.shape
    dx = state.dx
    
    # Ray slope in grid units
    ray_slope = sun_x / sun_y  # dx/dy along ray
    
    # Initialize intensity array for top boundary
    intensity = np.ones((ny, nx)) * solar_constant
    
    # For each point, calculate which cell the ray came from
    for j in range(1, ny):
        for i in range(nx):
            # Where did this ray come from?
            source_x = i - ray_slope * j
            
            # Interpolate intensity from source
            if 0 <= source_x < nx - 1:
                i0 = int(source_x)
                i1 = i0 + 1
                frac = source_x - i0
                
                # Get intensity from previous row
                prev_intensity = (1 - frac) * intensity[j-1, i0] + frac * intensity[j-1, i1]
                
                # Apply absorption
                if state.density[j, i] >= 0.1:
                    absorbed = prev_intensity * effective_absorption[j, i]
                    
                    # Update power and temperature
                    volumetric_power = absorbed / state.cell_depth
                    state.power_density[j, i] += volumetric_power
                    
                    if state.density[j, i] > 1.0 and state.specific_heat[j, i] > 0:
                        denominator = state.density[j, i] * state.specific_heat[j, i]
                        if denominator > 1e-10:
                            dT = volumetric_power * dt / denominator
                            state.temperature[j, i] += dT
                    
                    # Store attenuated intensity
                    intensity[j, i] = prev_intensity * (1 - effective_absorption[j, i])
                else:
                    intensity[j, i] = prev_intensity


def _apply_near_horizontal_rays(state, dt, effective_absorption, solar_constant, sun_x, sun_y):
    """
    Handle rays that are mostly horizontal.
    
    Similar to vertical case but process from side boundary.
    """
    ny, nx = state.temperature.shape
    
    # Ray slope in grid units
    ray_slope = sun_y / abs(sun_x)  # dy/dx along ray
    
    # Determine which side rays enter from
    if sun_x > 0:
        # Rays from right side, going left
        start_col = nx - 1
        step = -1
    else:
        # Rays from left side, going right
        start_col = 0
        step = 1
    
    # Process column by column
    intensity = np.ones(ny) * solar_constant
    
    for offset in range(nx):
        i = start_col + offset * step
        if not (0 <= i < nx):
            break
            
        # For each cell in this column, trace back to boundary
        for j in range(ny):
            # Calculate source position
            source_y = j - ray_slope * offset
            
            if 0 <= source_y < ny - 1:
                j0 = int(source_y)
                j1 = j0 + 1
                frac = source_y - j0
                
                # Interpolate intensity
                ray_intensity = (1 - frac) * intensity[j0] + frac * intensity[j1]
                
                # Apply absorption and heating
                if state.density[j, i] >= 1.0 and ray_intensity > 1e-6:
                    absorbed = ray_intensity * effective_absorption[j, i]
                    
                    volumetric_power = absorbed / state.cell_depth
                    state.power_density[j, i] += volumetric_power
                    
                    if state.density[j, i] > 1.0 and state.specific_heat[j, i] > 0:
                        denominator = state.density[j, i] * state.specific_heat[j, i]
                        if denominator > 1e-10:
                            dT = volumetric_power * dt / denominator
                            state.temperature[j, i] += dT
                    
                    # Update intensity array for next column
                    intensity[j] *= (1 - effective_absorption[j, i])