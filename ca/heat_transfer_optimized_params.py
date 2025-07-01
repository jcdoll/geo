"""
Optimized heat transfer module with parameter tuning.
Based on the analysis, we keep the explicit method but optimize parameters.
"""

import numpy as np
from scipy import ndimage
try:
    from .materials import MaterialType, MaterialDatabase
    from .solar_heating import SolarHeating
except ImportError:
    from materials import MaterialType, MaterialDatabase
    from solar_heating import SolarHeating


class HeatTransferOptimized:
    """Heat transfer calculations with optimized parameters"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
        
        # Simple parameters - no artificial enhancements
        self.sim.max_diffusion_substeps = 50  # Enough substeps for stability
        
        # Cache for material properties
        self._emissivity_cache = {}
        self._last_material_update = -1
        
        # Solar heating module
        self.solar_heating = SolarHeating(simulation)
    
    def _update_material_cache(self):
        """Update cached material properties if needed"""
        # Check if materials have changed
        current_hash = hash(tuple(self.sim.material_types.flat[:1000]))  # Sample for performance
        if current_hash == self._last_material_update:
            return
            
        # Clear and rebuild cache
        self._emissivity_cache.clear()
        
        unique_materials = set(self.sim.material_types.flat)
        for mat in unique_materials:
            if mat != MaterialType.SPACE:
                props = self.sim.material_db.get_properties(mat)
                self._emissivity_cache[mat] = props.emissivity
        
        self._last_material_update = current_hash
    
    def solve_heat_diffusion(self):
        """Apply operator splitting to solve heat equation with sources."""
        
        
        # Reset power density to show instantaneous power, not accumulated
        self.sim.power_density.fill(0.0)
        
        # Update material cache
        self._update_material_cache()
        
        # Get mask of non-space cells
        non_space_mask = self.sim.material_types != MaterialType.SPACE
        
        # Step 1: Pure diffusion (if enabled)
        if self.sim.enable_heat_diffusion:
            working_temp, stability = self._solve_pure_diffusion(self.sim.temperature, non_space_mask)
        else:
            working_temp = self.sim.temperature.copy()
            stability = 1.0
        
        # Step 2: Radiative cooling (if enabled)
        if self.sim.enable_radiative_cooling:
            working_temp = self._solve_radiative_cooling(working_temp, non_space_mask)
        
        # Step 3: Non-radiative heat sources (if enabled)
        working_temp = self._solve_non_radiative_sources(working_temp, non_space_mask)
        
        return working_temp, stability
    
    def _solve_pure_diffusion(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve pure diffusion with optimized parameters"""
        # Get thermal diffusivity for all cells (α = k / (ρ * cp))
        valid_thermal = (self.sim.density > 0) & (self.sim.specific_heat > 0) & (self.sim.thermal_conductivity > 0)
        thermal_diffusivity = np.zeros_like(self.sim.thermal_conductivity)
        thermal_diffusivity[valid_thermal] = (
            self.sim.thermal_conductivity[valid_thermal] /
            (self.sim.density[valid_thermal] * self.sim.specific_heat[valid_thermal])
        )  # m²/s

        # No enhancements - just use material properties directly

        # Stability analysis for PURE DIFFUSION only
        dx_squared = self.sim.cell_size ** 2
        max_alpha = np.max(thermal_diffusivity[non_space_mask]) if np.any(non_space_mask) else 0.0

        # Pure diffusion stability limit depends on stencil
        stencil_denominator = 4.0 if self.sim.diffusion_stencil == "radius1" else 16.0
        diffusion_dt_limit = dx_squared / (stencil_denominator * max_alpha) if max_alpha > 0 else float('inf')

        # Adaptive time step for diffusion
        min_dt_seconds = self.sim.dt / self.sim.max_diffusion_substeps
        target_dt_seconds = np.clip(diffusion_dt_limit, min_dt_seconds, self.sim.dt)
        adaptive_dt = target_dt_seconds

        # Use sub-steps for stability
        num_substeps = max(1, min(self.sim.max_diffusion_substeps, int(np.ceil(self.sim.dt / adaptive_dt))))
        actual_effective_dt = self.sim.dt / num_substeps
        actual_stability_factor = actual_effective_dt / self.sim.dt

        # Store debugging info
        self.sim._max_thermal_diffusivity = max_alpha
        self.sim._diffusion_substeps = num_substeps

        # Pure diffusion sub-stepping
        new_temp = temperature.copy()

        for step in range(num_substeps):
            # Pure diffusion step
            new_temp = self._solve_diffusion_step(new_temp, thermal_diffusivity, actual_effective_dt, non_space_mask)

        return new_temp, actual_stability_factor

    def _solve_diffusion_step(self, temperature: np.ndarray, thermal_diffusivity: np.ndarray,
                             dt: float, non_space_mask: np.ndarray) -> np.ndarray:
        """Explicit Euler thermal diffusion step"""
        
        dt_seconds = dt
        dx_squared = self.sim.cell_size ** 2

        if self.sim.diffusion_stencil == "radius1":
            laplacian = np.zeros_like(temperature)
            for i, (dy, dx) in enumerate(self.sim.neighbors_8):
                neighbor_temp = np.roll(np.roll(temperature, dy, axis=0), dx, axis=1)
                weight = self.sim.distance_factors_8[i]
                laplacian += weight * (neighbor_temp - temperature)
            laplacian /= np.sum(self.sim.distance_factors_8)
        else:  # radius2 isotropic default
            laplacian = ndimage.convolve(temperature, self.sim._laplacian_kernel_radius2, mode="nearest")

        # Zero out Laplacian for space cells
        laplacian[~non_space_mask] = 0.0

        # Combined diffusion update
        diffusion_change = thermal_diffusivity * dt_seconds * laplacian / dx_squared

        # Apply updates only to non-space cells
        new_temp = temperature.copy()
        if np.any(non_space_mask):
            new_temp[non_space_mask] += diffusion_change[non_space_mask]

        return new_temp
    
    
    def _solve_radiative_cooling(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Solve radiative cooling with cached properties"""
        working_temp = temperature.copy()
        
        # Identify cells that can radiate
        space_mask = (self.sim.material_types == MaterialType.SPACE)
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        
        # Outer atmosphere: atmospheric cells adjacent to space
        outer_atmo_mask = atmosphere_mask & ndimage.binary_dilation(space_mask, structure=self.sim._circular_kernel_5x5)
        
        # Surface solids
        solid_mask = non_space_mask & ~atmosphere_mask
        surface_candidates = ndimage.binary_dilation(outer_atmo_mask | space_mask, structure=self.sim._circular_kernel_5x5)
        surface_solid_mask = surface_candidates & solid_mask
        
        # Combine for radiation
        radiative_mask = outer_atmo_mask | surface_solid_mask
        
        if not np.any(radiative_mask):
            return working_temp
        
        # Get coordinates and properties
        radiative_coords = np.where(radiative_mask)
        T_radiating = working_temp[radiative_coords]
        T_space = self.sim.space_temperature
        
        # Only process cells that are actually cooling
        cooling_mask = T_radiating > T_space
        if not np.any(cooling_mask):
            return working_temp
        
        cooling_idx = np.where(cooling_mask)[0]
        T_cooling = T_radiating[cooling_idx]
        
        # Get material properties using cache
        material_types_cooling = self.sim.material_types[radiative_coords][cooling_idx]
        emissivity = np.array([self._emissivity_cache.get(mat, 0.8) for mat in material_types_cooling])
        density_cooling = self.sim.density[radiative_coords][cooling_idx]
        specific_heat_cooling = self.sim.specific_heat[radiative_coords][cooling_idx]
        
        # Calculate dynamic greenhouse effect
        water_vapor_mask = (self.sim.material_types == MaterialType.WATER_VAPOR)
        total_water_vapor_mass = np.sum(self.sim.density[water_vapor_mask]) if np.any(water_vapor_mask) else 0.0
        
        if total_water_vapor_mass > 0:
            vapor_factor = np.log1p(total_water_vapor_mass / self.sim.greenhouse_vapor_scaling) / 10.0
            greenhouse_factor = self.sim.base_greenhouse_effect + (self.sim.max_greenhouse_effect - self.sim.base_greenhouse_effect) * np.tanh(vapor_factor)
        else:
            greenhouse_factor = self.sim.base_greenhouse_effect
        
        # Linearized Stefan-Boltzmann
        T_reference = np.maximum(T_cooling, 300.0)
        h_effective = 4 * self.sim.stefan_boltzmann * (1.0 - greenhouse_factor) * emissivity * T_reference**3
        
        # Calculate cooling rate
        surface_thickness = self.sim.cell_size * self.sim.surface_radiation_depth_fraction
        cooling_rate = h_effective * (T_cooling - T_space) / (density_cooling * specific_heat_cooling * surface_thickness)
        
        # Apply cooling with time step
        dt_seconds = self.sim.dt
        T_new = T_cooling - dt_seconds * cooling_rate
        T_new = np.maximum(T_new, T_space)
        
        # Update temperatures
        final_coords_y = radiative_coords[0][cooling_idx]
        final_coords_x = radiative_coords[1][cooling_idx]
        working_temp[final_coords_y, final_coords_x] = T_new
        
        # Update power density
        volumetric_power_density = h_effective * (T_cooling - T_space) / surface_thickness
        self.sim.power_density[final_coords_y, final_coords_x] -= volumetric_power_density
        
        # Track total radiative output
        cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
        total_radiative_power = np.sum(volumetric_power_density * cell_volume)
        self.sim.thermal_fluxes['radiative_output'] = total_radiative_power
        
        return working_temp
    
    def _solve_non_radiative_sources(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Apply non-radiative heat sources"""
        working_temp = temperature.copy()
        
        # Calculate source terms only if enabled
        internal_source = self._calculate_internal_heating_source(non_space_mask) if self.sim.enable_internal_heating else 0.0
        solar_source = self._calculate_solar_heating_source(non_space_mask) if self.sim.enable_solar_heating else 0.0
        atmospheric_source = self._calculate_atmospheric_heating_source(non_space_mask)
        
        # Total volumetric power density
        total_source = internal_source + solar_source + atmospheric_source
        
        # Temperature change
        with np.errstate(divide='ignore', invalid='ignore'):
            source_change = (total_source * self.sim.dt / 
                           (self.sim.density * self.sim.specific_heat))
            source_change[~non_space_mask] = 0.0
            source_change = np.nan_to_num(source_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        working_temp += source_change
        
        return working_temp
    
    def _calculate_internal_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate internal heating source term with caching"""
        source_term = np.zeros_like(self.sim.temperature)
        
        if not np.any(non_space_mask):
            return source_term
        
        # Calculate heat generation from radioactive materials
        heating_rate = np.zeros_like(self.sim.temperature)
        
        # Vectorized heat generation calculation
        unique_materials = set(self.sim.material_types[non_space_mask].flat)
        for mat_type in unique_materials:
            if mat_type != MaterialType.SPACE:
                mat_props = self.sim.material_db.get_properties(mat_type)
                if mat_props.heat_generation > 0:
                    mat_mask = (self.sim.material_types == mat_type) & non_space_mask
                    heating_rate[mat_mask] = mat_props.heat_generation
        
        # Convert to temperature change rate
        valid_cells = non_space_mask & (self.sim.density > 0) & (self.sim.specific_heat > 0)
        if np.any(valid_cells):
            source_term[valid_cells] = heating_rate[valid_cells] / (
                self.sim.density[valid_cells] * self.sim.specific_heat[valid_cells]
            )
        
        # Update power density and flux tracking
        self.sim.power_density[valid_cells] += heating_rate[valid_cells]
        cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
        total_internal_power = np.sum(heating_rate[valid_cells] * cell_volume)
        self.sim.thermal_fluxes['internal_heating'] = total_internal_power
        
        return source_term
    
    def _calculate_solar_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate solar heating source term using separate solar heating module"""
        return self.solar_heating.calculate_solar_heating(non_space_mask)
    
    
    def _calculate_atmospheric_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate atmospheric heating source term"""
        return np.zeros_like(self.sim.temperature)
    
    def apply_atmospheric_convection(self, temperature: np.ndarray) -> np.ndarray:
        """Apply fast atmospheric convection mixing"""
        working_temp = temperature.copy()
        
        # Identify atmospheric materials
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        
        if not np.any(atmosphere_mask):
            return working_temp
        
        # Calculate average temperature of atmospheric neighbors
        kernel = self.sim._circular_kernel_3x3
        atmo_temp = np.where(atmosphere_mask, working_temp, 0)
        atmo_count = ndimage.convolve(atmosphere_mask.astype(float), kernel, mode='constant', cval=0)
        atmo_sum = ndimage.convolve(atmo_temp, kernel, mode='constant', cval=0)
        
        # Avoid division by zero
        valid_atmo_neighbors = atmo_count > 0
        avg_atmo_neighbor_temp = np.zeros_like(working_temp)
        avg_atmo_neighbor_temp[valid_atmo_neighbors] = atmo_sum[valid_atmo_neighbors] / atmo_count[valid_atmo_neighbors]
        
        # Apply mixing only to atmospheric cells that have atmospheric neighbors
        mixing_mask = atmosphere_mask & valid_atmo_neighbors
        
        if np.any(mixing_mask):
            # Vectorized mixing
            temp_diff = avg_atmo_neighbor_temp[mixing_mask] - working_temp[mixing_mask]
            working_temp[mixing_mask] += self.sim.atmospheric_convection_mixing * temp_diff
        
        return working_temp

    def inject_heat(self, y: int, x: int, radius: int, delta_T: float) -> None:
        """Increase temperature by *delta_T* Kelvin inside a circular blob."""
        yy, xx = np.ogrid[:self.sim.height, :self.sim.width]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
        self.sim.temperature[mask] += delta_T