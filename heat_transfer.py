"""
Heat transfer module for geological simulation.
Handles thermal diffusion, heat sources, and radiative cooling.
"""

import numpy as np
from scipy import ndimage
try:
    from .materials import MaterialType, MaterialDatabase
except ImportError:
    from materials import MaterialType, MaterialDatabase


class HeatTransfer:
    """Heat transfer calculations for geological simulation"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
    
    def solve_heat_diffusion(self) -> tuple[np.ndarray, float]:
        """
        Solve heat diffusion using operator splitting method.
        
        Returns:
            tuple: (new_temperature, stability_factor)
        """
        non_space_mask = (self.sim.material_types != MaterialType.SPACE)

        if not np.any(non_space_mask):
            return self.sim.temperature, 1.0

        # Start with current temperature
        working_temp = self.sim.temperature.copy()

        # Step 1: Solve pure diffusion (no sources)
        working_temp, diffusion_stability = self._solve_pure_diffusion(working_temp, non_space_mask)

        # Step 2: Solve radiative cooling using selected method (dispatcher)
        working_temp = self._solve_radiative_cooling(working_temp, non_space_mask)

        # Step 3: Solve other heat sources explicitly (internal, solar, atmospheric)
        working_temp = self._solve_non_radiative_sources(working_temp, non_space_mask)

        # Overall stability factor is dominated by diffusion (radiation and sources are stable)
        overall_stability = diffusion_stability

        # Store debugging info
        self.sim._actual_substeps = getattr(self.sim, '_diffusion_substeps', 1)
        self.sim._actual_effective_dt = self.sim.dt * overall_stability

        # Debug info
        avg_temp_before = np.mean(self.sim.temperature[non_space_mask]) - 273.15 if np.any(non_space_mask) else 0.0
        avg_temp_after = np.mean(working_temp[non_space_mask]) - 273.15 if np.any(non_space_mask) else 0.0
        if self.sim.logging_enabled:
            self.sim.logger.debug(
                f"Diffusion sub-steps: {self.sim._actual_substeps}, ΔT planet avg: {avg_temp_before:6.1f}→{avg_temp_after:6.1f} °C")

        return working_temp, overall_stability
    
    def _solve_pure_diffusion(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve pure diffusion (no sources) for maximum stability"""
        # Get thermal diffusivity for all cells (α = k / (ρ * cp))
        valid_thermal = (self.sim.density > 0) & (self.sim.specific_heat > 0) & (self.sim.thermal_conductivity > 0)
        thermal_diffusivity = np.zeros_like(self.sim.thermal_conductivity)
        thermal_diffusivity[valid_thermal] = (
            self.sim.thermal_conductivity[valid_thermal] /
            (self.sim.density[valid_thermal] * self.sim.specific_heat[valid_thermal])
        )  # m²/s

        # Enhanced atmospheric convection (much faster heat transfer in gases)
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        atmospheric_cells = atmosphere_mask & valid_thermal
        if np.any(atmospheric_cells):
            thermal_diffusivity[atmospheric_cells] *= self.sim.atmospheric_diffusivity_enhancement

        # Enhanced diffusion at material interfaces
        if hasattr(self.sim, 'interface_diffusivity_enhancement'):
            neighbors = self.sim._get_neighbors(4, shuffle=False)
            for dy, dx in neighbors:
                shifted_materials = np.roll(np.roll(self.sim.material_types, dy, axis=0), dx, axis=1)
                interface_mask = (self.sim.material_types != shifted_materials) & non_space_mask
                if np.any(interface_mask):
                    enhancement_mask = interface_mask & valid_thermal
                    if np.any(enhancement_mask):
                        thermal_diffusivity[enhancement_mask] *= self.sim.interface_diffusivity_enhancement

        # Clamp extreme thermal diffusivity values (e.g., near-vacuum cells)
        thermal_diffusivity = np.clip(thermal_diffusivity, 0.0, self.sim.max_thermal_diffusivity)

        # Stability analysis for PURE DIFFUSION only (no sources)
        dx_squared = self.sim.cell_size ** 2
        max_alpha = np.max(thermal_diffusivity[non_space_mask]) if np.any(non_space_mask) else 0.0

        # Pure diffusion stability limit depends on stencil
        stencil_denominator = 4.0 if self.sim.diffusion_stencil == "radius1" else 16.0
        diffusion_dt_limit = dx_squared / (stencil_denominator * max_alpha) if max_alpha > 0 else float('inf')

        # Adaptive time step for diffusion (clamp between min substep and full timestep)
        min_dt_seconds = self.sim.dt * self.sim.seconds_per_year / self.sim.max_diffusion_substeps
        target_dt_seconds = np.clip(diffusion_dt_limit, min_dt_seconds, self.sim.dt * self.sim.seconds_per_year)

        # Convert back to years
        adaptive_dt = target_dt_seconds / self.sim.seconds_per_year
        stability_factor = adaptive_dt / self.sim.dt

        # Use sub-steps for stability
        num_substeps = max(1, min(self.sim.max_diffusion_substeps, int(np.ceil(self.sim.dt / adaptive_dt))))
        actual_effective_dt = self.sim.dt / num_substeps
        actual_stability_factor = actual_effective_dt / self.sim.dt

        # Store debugging info
        self.sim._max_thermal_diffusivity = max_alpha
        self.sim._diffusion_substeps = num_substeps

        # Pure diffusion sub-stepping (much simpler without sources)
        new_temp = temperature.copy()

        for step in range(num_substeps):
            # Pure diffusion step using selected method
            new_temp = self._solve_diffusion_step(new_temp, thermal_diffusivity, actual_effective_dt, non_space_mask)

        return new_temp, actual_stability_factor

    def _solve_diffusion_step(self, temperature: np.ndarray, thermal_diffusivity: np.ndarray,
                             dt: float, non_space_mask: np.ndarray) -> np.ndarray:
        """Master thermal diffusion solver - dispatches to selected implementation method"""
        
        if self.sim.thermal_diffusion_method == "explicit_euler":
            return self._diffusion_step_explicit_euler(temperature, thermal_diffusivity, dt, non_space_mask)
        else:
            raise ValueError(f"Unknown thermal diffusion method: {self.sim.thermal_diffusion_method}. "
                           f"Available options: 'explicit_euler'")

    def _diffusion_step_explicit_euler(self, temperature: np.ndarray, thermal_diffusivity: np.ndarray,
                                      dt: float, non_space_mask: np.ndarray) -> np.ndarray:
        """
        Explicit Euler thermal diffusion step
        
        Method: Forward Euler finite difference for dT/dt = α∇²T
        Advantages: Simple, fast, well-tested
        Disadvantages: Requires small time steps for stability
        """
        # Convert dt to seconds for proper units
        dt_seconds = dt * self.sim.seconds_per_year
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

        # Zero out Laplacian for space cells (no diffusion)
        laplacian[~non_space_mask] = 0.0

        # Combined diffusion update (vectorized)
        # dT/dt = α∇²T
        diffusion_change = thermal_diffusivity * dt_seconds * laplacian / dx_squared

        # Apply updates only to non-space cells
        new_temp = temperature.copy()
        if np.any(non_space_mask):
            new_temp[non_space_mask] += diffusion_change[non_space_mask]

        return new_temp

    
    def _get_interface_mask(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Get mask of cells at material interfaces"""
        # Dilate non-space mask and find boundary
        dilated = ndimage.binary_dilation(non_space_mask, structure=self.sim._circular_kernel_3x3)
        interface_mask = dilated & ~non_space_mask
        
        # Also include cells adjacent to different materials
        material_boundaries = np.zeros_like(non_space_mask)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(self.sim.material_types, dy, axis=0), dx, axis=1)
                different_materials = (self.sim.material_types != shifted) & non_space_mask
                material_boundaries |= different_materials
        
        return interface_mask | material_boundaries
    
    def _solve_radiative_cooling(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Solve radiative cooling using selected method"""
        if self.sim.radiative_cooling_method == "linearized_stefan_boltzmann":
            return self._solve_radiative_cooling_linearized_stefan_boltzmann(temperature, non_space_mask)
        elif self.sim.radiative_cooling_method == "newton_raphson_implicit":
            return self._solve_radiative_cooling_newton_raphson_implicit(temperature, non_space_mask)
        else:
            raise ValueError(f"Unknown radiative cooling method: {self.sim.radiative_cooling_method}")
    
    def _solve_radiative_cooling_linearized_stefan_boltzmann(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Solve radiative cooling using linearized Stefan-Boltzmann law"""
        working_temp = temperature.copy()
        
        # Identify cells that can radiate (outer atmosphere + surface solids)
        space_mask = (self.sim.material_types == MaterialType.SPACE)
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        
        # Outer atmosphere: atmospheric cells adjacent to space
        outer_atmo_mask = atmosphere_mask & ndimage.binary_dilation(space_mask, structure=self.sim._circular_kernel_5x5)
        
        # Surface solids: non-atmospheric, non-space cells adjacent to outer atmosphere or space
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
        
        # Get material properties
        material_types_cooling = self.sim.material_types[radiative_coords][cooling_idx]
        emissivity = np.array([self.sim.material_db.get_properties(mat_type).emissivity for mat_type in material_types_cooling])
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
        
        # Linearized Stefan-Boltzmann: Q = h(T - T_space) where h = 4σεT₀³
        T_reference = np.maximum(T_cooling, 300.0)  # Use actual temperature as reference
        h_effective = 4 * self.sim.stefan_boltzmann_geological * (1.0 - greenhouse_factor) * emissivity * T_reference**3
        
        # Calculate cooling rate
        surface_thickness = self.sim.cell_size * self.sim.surface_radiation_depth_fraction
        cooling_rate = h_effective * (T_cooling - T_space) / (density_cooling * specific_heat_cooling * surface_thickness)
        
        # Apply cooling with time step
        dt_seconds = self.sim.dt * self.sim.seconds_per_year
        T_new = T_cooling - dt_seconds * cooling_rate
        T_new = np.maximum(T_new, T_space)  # Don't cool below space temperature
        
        # Update temperatures
        final_coords_y = radiative_coords[0][cooling_idx]
        final_coords_x = radiative_coords[1][cooling_idx]
        working_temp[final_coords_y, final_coords_x] = T_new
        
        # Update power density for visualization
        volumetric_power_density = h_effective * (T_cooling - T_space) / (surface_thickness * self.sim.seconds_per_year)
        self.sim.power_density[final_coords_y, final_coords_x] -= volumetric_power_density
        
        # Update thermal flux tracking
        total_radiative_power = np.sum(volumetric_power_density * self.sim.cell_size**2 * surface_thickness)
        self.sim.thermal_fluxes['radiative_output'] = total_radiative_power
        
        return working_temp
    
    def _solve_radiative_cooling_newton_raphson_implicit(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Solve radiative cooling using Newton-Raphson implicit method for full Stefan-Boltzmann"""
        working_temp = temperature.copy()
        
        # Identify radiating cells (same logic as linearized method)
        space_mask = (self.sim.material_types == MaterialType.SPACE)
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        
        outer_atmo_mask = atmosphere_mask & ndimage.binary_dilation(space_mask, structure=self.sim._circular_kernel_5x5)
        solid_mask = non_space_mask & ~atmosphere_mask
        surface_candidates = ndimage.binary_dilation(outer_atmo_mask | space_mask, structure=self.sim._circular_kernel_5x5)
        surface_solid_mask = surface_candidates & solid_mask
        radiative_mask = outer_atmo_mask | surface_solid_mask
        
        if not np.any(radiative_mask):
            return working_temp
        
        # Get coordinates and properties
        radiative_coords = np.where(radiative_mask)
        T_radiating = working_temp[radiative_coords]
        T_space = self.sim.space_temperature
        
        # Only process cooling cells
        cooling_mask = T_radiating > T_space
        if not np.any(cooling_mask):
            return working_temp
        
        cooling_idx = np.where(cooling_mask)[0]
        T_cooling = T_radiating[cooling_idx]
        
        # Get material properties
        material_types_cooling = self.sim.material_types[radiative_coords][cooling_idx]
        emissivity = np.array([self.sim.material_db.get_properties(mat_type).emissivity for mat_type in material_types_cooling])
        density_cooling = self.sim.density[radiative_coords][cooling_idx]
        specific_heat_cooling = self.sim.specific_heat[radiative_coords][cooling_idx]
        
        # Calculate greenhouse effect
        water_vapor_mask = (self.sim.material_types == MaterialType.WATER_VAPOR)
        total_water_vapor_mass = np.sum(self.sim.density[water_vapor_mask]) if np.any(water_vapor_mask) else 0.0
        
        if total_water_vapor_mass > 0:
            vapor_factor = np.log1p(total_water_vapor_mass / self.sim.greenhouse_vapor_scaling) / 10.0
            greenhouse_factor = self.sim.base_greenhouse_effect + (self.sim.max_greenhouse_effect - self.sim.base_greenhouse_effect) * np.tanh(vapor_factor)
        else:
            greenhouse_factor = self.sim.base_greenhouse_effect
        
        # Stefan-Boltzmann constant with greenhouse effect
        stefan_geological = self.sim.stefan_boltzmann_geological / 1000.0  # Conservative scaling
        surface_thickness = self.sim.cell_size * self.sim.surface_radiation_depth_fraction
        alpha = (stefan_geological * emissivity * self.sim.radiative_cooling_efficiency * (1.0 - greenhouse_factor)) / (density_cooling * specific_heat_cooling * surface_thickness)
        
        dt_seconds = self.sim.dt * self.sim.seconds_per_year
        
        # Newton-Raphson iteration for implicit radiation
        T_new = T_cooling.copy()
        for iteration in range(3):  # Usually converges quickly
            f = T_new - T_cooling + dt_seconds * alpha * (T_new**4 - T_space**4)
            df_dt = 1.0 + dt_seconds * alpha * 4.0 * T_new**3
            
            # Newton-Raphson update
            delta_T = -f / df_dt
            T_new += delta_T
            
            # Keep temperatures physical
            T_new = np.maximum(T_new, T_space)
            
            # Check convergence
            if np.max(np.abs(delta_T)) < 0.1:  # 0.1K tolerance
                break
        
        # Update temperatures
        final_coords_y = radiative_coords[0][cooling_idx]
        final_coords_x = radiative_coords[1][cooling_idx]
        working_temp[final_coords_y, final_coords_x] = T_new
        
        # Update power density for visualization
        effective_stefan = stefan_geological * (1.0 - greenhouse_factor)
        power_per_area = effective_stefan * emissivity * (T_new**4 - T_space**4)
        volumetric_power_density = power_per_area / (surface_thickness * self.sim.seconds_per_year)
        self.sim.power_density[final_coords_y, final_coords_x] -= volumetric_power_density
        
        return working_temp
    
    def _solve_non_radiative_sources(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Apply all non-radiative heat sources"""
        working_temp = temperature.copy()
        
        # Reset power density tracking
        self.sim.power_density.fill(0.0)
        
        # Calculate heat sources
        internal_source = self._calculate_internal_heating_source(non_space_mask)
        solar_source = self._calculate_solar_heating_source(non_space_mask)
        atmospheric_source = self._calculate_atmospheric_heating_source(non_space_mask)
        
        # Apply sources
        total_source = internal_source + solar_source + atmospheric_source
        working_temp += total_source * self.sim.dt
        
        return working_temp
    
    def _calculate_internal_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate internal heating source term Q/(ρcp) in K/year"""
        source_term = np.zeros_like(self.sim.temperature)
        
        if not np.any(non_space_mask):
            return source_term
        
        # Calculate depth-dependent heating
        distances = self.sim._get_distances_from_center()
        planet_radius = self.sim._get_planet_radius()
        
        # Normalized depth (0 at surface, 1 at center)
        depth_normalized = np.maximum(0, (planet_radius - distances) / planet_radius)
        
        # Exponential heating profile
        heating_rate = 1e-6 * np.exp(depth_normalized / self.sim.core_heating_depth_scale)  # W/m³
        
        # Convert to temperature change rate
        valid_cells = non_space_mask & (self.sim.density > 0) & (self.sim.specific_heat > 0)
        if np.any(valid_cells):
            source_term[valid_cells] = heating_rate[valid_cells] / (
                self.sim.density[valid_cells] * self.sim.specific_heat[valid_cells]
            ) * self.sim.seconds_per_year  # K/year
        
        # Update power density and flux tracking
        self.sim.power_density[valid_cells] += heating_rate[valid_cells]
        total_internal_power = np.sum(heating_rate[valid_cells] * self.sim.cell_size**2)
        self.sim.thermal_fluxes['internal_heating'] = total_internal_power
        
        return source_term
    
    def _calculate_solar_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate solar heating source term Q/(ρcp) in K/year"""
        source_term = np.zeros_like(self.sim.temperature)
        
        # Find surface cells that can receive solar radiation
        space_mask = ~non_space_mask
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        surface_candidates = ndimage.binary_dilation(space_mask, structure=self.sim._circular_kernel_5x5) & non_space_mask & ~atmosphere_mask
        
        if not np.any(surface_candidates):
            return source_term
        
        # Calculate solar heating based on angle from solar direction
        center_x, center_y = self.sim.center_of_mass
        solar_dir_x, solar_dir_y = self.sim._get_solar_direction()
        
        # Create coordinate grids
        y_coords = np.arange(self.sim.height).reshape(-1, 1)
        x_coords = np.arange(self.sim.width).reshape(1, -1)
        
        # Calculate position vectors from center
        pos_x = x_coords - center_x
        pos_y = y_coords - center_y
        
        # Calculate solar intensity factor
        distance_from_center = np.sqrt(pos_x**2 + pos_y**2)
        safe_distance = np.maximum(distance_from_center, 0.1)
        dot_product = (pos_x * solar_dir_x + pos_y * solar_dir_y) / safe_distance
        solar_intensity_factor = np.maximum(0.0, dot_product)
        
        # Calculate effective solar constant
        effective_solar_constant = self.sim.solar_constant * self.sim.planetary_distance_factor
        
        # Apply atmospheric absorption
        source_term = self._solve_atmospheric_absorption(non_space_mask, solar_intensity_factor, effective_solar_constant, source_term)
        
        return source_term
    
    def _solve_atmospheric_absorption(self, non_space_mask: np.ndarray, solar_intensity_factor: np.ndarray,
                                    effective_solar_constant: float, source_term: np.ndarray) -> np.ndarray:
        """Solve atmospheric absorption using selected method"""
        if self.sim.atmospheric_absorption_method == "directional_sweep":
            return self._atmospheric_absorption_directional_sweep(non_space_mask, solar_intensity_factor, effective_solar_constant, source_term)
        else:
            raise ValueError(f"Unknown atmospheric absorption method: {self.sim.atmospheric_absorption_method}")
    
    def _atmospheric_absorption_directional_sweep(self, non_space_mask: np.ndarray, solar_intensity_factor: np.ndarray,
                                                 effective_solar_constant: float, source_term: np.ndarray) -> np.ndarray:
        """Directional sweep atmospheric absorption using DDA ray marching"""
        # Get solar direction
        solar_dir_x, solar_dir_y = self.sim._get_solar_direction()
        
        # Determine entry boundary based on solar direction
        if solar_dir_x > 0:  # Sun from left, enter from left boundary
            entry_x = 0
            x_range = range(self.sim.width)
        else:  # Sun from right, enter from right boundary
            entry_x = self.sim.width - 1
            x_range = range(self.sim.width - 1, -1, -1)
        
        if solar_dir_y > 0:  # Sun from top, enter from top boundary
            entry_y = 0
            y_range = range(self.sim.height)
        else:  # Sun from bottom, enter from bottom boundary
            entry_y = self.sim.height - 1
            y_range = range(self.sim.height - 1, -1, -1)
        
        # March rays through the grid
        for y in y_range:
            for x in x_range:
                if self.sim.material_types[y, x] == MaterialType.SPACE:
                    continue
                
                # Initial intensity based on solar angle
                intensity = effective_solar_constant * solar_intensity_factor[y, x]
                
                if intensity <= 0:
                    continue
                
                # Get material absorption coefficient
                material_props = self.sim.material_db.get_properties(self.sim.material_types[y, x])
                absorption_coeff = getattr(material_props, 'optical_absorption', 0.1)
                
                # Calculate absorbed energy
                absorbed = intensity * absorption_coeff
                intensity -= absorbed
                
                # Convert to temperature change rate
                if self.sim.density[y, x] > 0 and self.sim.specific_heat[y, x] > 0:
                    # Apply albedo
                    albedo = getattr(material_props, 'albedo', 0.3)
                    effective_absorbed = absorbed * (1.0 - albedo)
                    
                    # Convert to volumetric power density
                    surface_thickness = self.sim.cell_size * self.sim.surface_radiation_depth_fraction
                    volumetric_power = effective_absorbed / surface_thickness
                    
                    # Convert to temperature change rate
                    temp_change_rate = volumetric_power / (self.sim.density[y, x] * self.sim.specific_heat[y, x]) * self.sim.seconds_per_year
                    source_term[y, x] += temp_change_rate
                    
                    # Update power density
                    if (self.sim.material_types[y, x] == MaterialType.AIR or 
                        self.sim.material_types[y, x] == MaterialType.WATER_VAPOR):
                        # Atmospheric heating
                        self.sim.thermal_fluxes['atmospheric_heating'] += effective_absorbed * self.sim.cell_size**2
                    else:
                        # Surface heating
                        self.sim.power_density[y, x] += volumetric_power
                        self.sim.thermal_fluxes['solar_input'] += effective_absorbed * self.sim.cell_size**2
        
        return source_term
    
    def _calculate_atmospheric_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate atmospheric heating source term (placeholder for future expansion)"""
        # Currently handled in solar heating with atmospheric absorption
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
            # Vectorized mixing: T_new = T_old + mixing_fraction * (T_avg_atmo_neighbors - T_old)
            temp_diff = avg_atmo_neighbor_temp[mixing_mask] - working_temp[mixing_mask]
            working_temp[mixing_mask] += self.sim.atmospheric_convection_mixing * temp_diff
        
        return working_temp 