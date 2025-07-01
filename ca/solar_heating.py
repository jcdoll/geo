"""
Solar heating module with atmospheric absorption via ray marching.
Separated out for performance analysis and potential optimization.
"""

import numpy as np
from scipy import ndimage
import time

try:
    from .materials import MaterialType, MaterialDatabase
except ImportError:
    from materials import MaterialType, MaterialDatabase


class SolarHeating:
    """Handles solar radiation and atmospheric absorption"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
        self.material_db = MaterialDatabase()
        
        # Cache for material optical properties
        self._absorption_cache = {}
        self._last_cache_update = -1
        
        # Performance tracking
        self.last_ray_march_time = 0.0
        self.last_surface_calc_time = 0.0
        self.last_total_time = 0.0
    
    def update_material_cache(self):
        """Update cached optical properties if materials have changed"""
        # Simple change detection using material types hash
        current_hash = hash(tuple(self.sim.material_types.flat[:1000]))  # Sample first 1000 cells
        if current_hash == self._last_cache_update:
            return
            
        self._absorption_cache.clear()
        
        unique_materials = set(self.sim.material_types.flat)
        for mat in unique_materials:
            if mat != MaterialType.SPACE:
                props = self.material_db.get_properties(mat)
                self._absorption_cache[mat] = getattr(props, 'optical_absorption', 0.1)
        
        self._last_cache_update = current_hash
    
    def calculate_solar_heating(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate solar heating source term with detailed timing"""
        start_total = time.perf_counter()
        
        # Initialize temporary power density array for this step
        self._temp_power_density = np.zeros_like(self.sim.power_density)
        
        # Update material property cache
        self.update_material_cache()
        
        source_term = np.zeros_like(self.sim.temperature)
        
        # Step 1: Calculate solar intensity based on angle
        start_surface = time.perf_counter()
        
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
        
        # Use solar constant directly (no albedo)
        effective_solar_constant = self.sim.solar_constant
        
        self.last_surface_calc_time = time.perf_counter() - start_surface
        
        # Step 2: Apply atmospheric absorption via ray marching
        start_ray_march = time.perf_counter()
        source_term = self._apply_ray_marching_absorption(
            non_space_mask, 
            solar_intensity_factor, 
            effective_solar_constant, 
            source_term
        )
        self.last_ray_march_time = time.perf_counter() - start_ray_march
        
        self.last_total_time = time.perf_counter() - start_total
        
        # Apply mild smoothing to reduce banding artifacts from discrete ray marching
        # Only smooth non-space cells to avoid spreading energy into vacuum
        if np.any(non_space_mask):
            from scipy import ndimage
            # Create a small smoothing kernel
            kernel = np.array([[0.05, 0.1, 0.05],
                              [0.1,  0.4, 0.1],
                              [0.05, 0.1, 0.05]])
            # Apply smoothing only where there's non-zero source
            source_mask = (source_term > 0) & non_space_mask
            if np.any(source_mask):
                smoothed = ndimage.convolve(source_term * source_mask, kernel, mode='constant')
                # Preserve total energy by rescaling
                total_before = np.sum(source_term[source_mask])
                if total_before > 0:
                    smoothed_masked = smoothed * non_space_mask
                    total_after = np.sum(smoothed_masked[source_mask])
                    if total_after > 0:
                        scale_factor = total_before / total_after
                        source_term = smoothed_masked * scale_factor
            
            # Also smooth power density for visualization
            if hasattr(self, '_temp_power_density'):
                power_mask = (self._temp_power_density > 0) & non_space_mask
                if np.any(power_mask):
                    smoothed_power = ndimage.convolve(self._temp_power_density * power_mask, kernel, mode='constant')
                    # Update the main power density with smoothed values
                    self.sim.power_density += smoothed_power
                else:
                    # No smoothing needed, just add directly
                    self.sim.power_density += self._temp_power_density
        
        return source_term
    
    def _apply_ray_marching_absorption(self, non_space_mask: np.ndarray, 
                                      solar_intensity_factor: np.ndarray,
                                      effective_solar_constant: float, 
                                      source_term: np.ndarray) -> np.ndarray:
        """
        DDA ray marching atmospheric absorption algorithm.
        Properly traces rays from the edge of the domain based on solar angle.
        """
        # Calculate initial flux
        initial_flux = effective_solar_constant * self.sim.planetary_distance_factor * solar_intensity_factor
        
        # Get solar direction
        ux, uy = self.sim._get_solar_direction()
        if ux == 0 and uy == 0:
            return source_term
        
        
        # DDA stepping direction: move OPPOSITE to incoming solar vector
        step_x = -1 if ux > 0 else (1 if ux < 0 else 0)
        step_y = -1 if uy > 0 else (1 if uy < 0 else 0)
        inv_dx = abs(1.0 / ux) if ux != 0 else float('inf')
        inv_dy = abs(1.0 / uy) if uy != 0 else float('inf')
        
        # Select entry cells from edges based on solar direction
        # Always use both edges to avoid masking and ensure complete coverage
        entry_cells = []
        
        # Determine which edges to use based on solar direction
        use_vertical_edge = abs(ux) > 1e-6 or abs(uy) > 0.9  # Use vertical edge if horizontal component OR nearly vertical
        use_horizontal_edge = abs(uy) > 1e-6 or abs(ux) > 0.9  # Use horizontal edge if vertical component OR nearly horizontal
        
        # Add vertical edge (left or right)
        if use_vertical_edge:
            entry_x = self.sim.width - 1 if ux > 0 else 0
            entry_cells.extend((entry_x, y) for y in range(self.sim.height))
        
        # Add horizontal edge (top or bottom)
        if use_horizontal_edge:
            entry_y = self.sim.height - 1 if uy > 0 else 0
            entry_cells.extend((x, entry_y) for x in range(self.sim.width))
        
        # Fallback (should never happen)
        if not entry_cells:
            entry_cells = [(x, 0) for x in range(self.sim.width)]
        
        # Main DDA march
        for sx, sy in entry_cells:
            I = initial_flux[sy, sx]
            if I <= 0:
                continue
                
            t_max_x = inv_dx
            t_max_y = inv_dy
            
            while 0 <= sx < self.sim.width and 0 <= sy < self.sim.height and I > 0:
                mat = self.sim.material_types[sy, sx]
                
                if mat != MaterialType.SPACE:
                    # Get absorption coefficient (1.0 for solids, partial for atmosphere)
                    k = self.sim.material_db.get_solar_absorption(mat)
                    
                    # Calculate absorbed energy (no albedo)
                    absorbed = I * k
                    
                    if absorbed > 0 and self.sim.density[sy, sx] > 0 and self.sim.specific_heat[sy, sx] > 0:
                        # Convert to volumetric power density
                        # absorbed (W/m²) × face_area (cell_size × cell_depth) / cell_volume (cell_size² × cell_depth)
                        # = absorbed / cell_size
                        vol_power = absorbed / self.sim.cell_size
                        
                        # Temperature change rate
                        temp_change_rate = vol_power / (self.sim.density[sy, sx] * self.sim.specific_heat[sy, sx])
                        source_term[sy, sx] += temp_change_rate
                        
                        # Update power density tracking (accumulate in temporary array)
                        if not hasattr(self, '_temp_power_density'):
                            self._temp_power_density = np.zeros_like(self.sim.power_density)
                        self._temp_power_density[sy, sx] += vol_power
                        
                        # Categorise energy deposition for diagnostics
                        cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
                        if mat in (MaterialType.AIR, MaterialType.WATER_VAPOR):
                            if hasattr(self.sim, 'thermal_fluxes'):
                                self.sim.thermal_fluxes['atmospheric_heating'] += vol_power * cell_volume
                        else:
                            if hasattr(self.sim, 'thermal_fluxes'):
                                self.sim.thermal_fluxes['solar_input'] += vol_power * cell_volume
                    
                    # Reduce intensity for next cell
                    I -= absorbed
                    
                    # Check if ray is terminated (opaque surface or fully absorbed)
                    if k >= 1.0 or I <= 0:
                        break
                
                # Advance to next grid cell using DDA
                if t_max_x < t_max_y:
                    sx += step_x
                    t_max_x += inv_dx
                else:
                    sy += step_y
                    t_max_y += inv_dy
        
        return source_term
    
    def get_timing_stats(self) -> dict:
        """Return timing statistics for performance analysis"""
        return {
            'total_time': self.last_total_time,
            'surface_calc_time': self.last_surface_calc_time,
            'ray_march_time': self.last_ray_march_time,
            'ray_march_percentage': (self.last_ray_march_time / self.last_total_time * 100 
                                   if self.last_total_time > 0 else 0)
        }