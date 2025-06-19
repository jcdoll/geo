"""
Material processes module for geological simulation.
Handles metamorphism, weathering, and phase transitions.
"""

import numpy as np
from scipy import ndimage
try:
    from .materials import MaterialType
    from .simulation_utils import SimulationUtils
except ImportError:
    from materials import MaterialType
    from simulation_utils import SimulationUtils


class MaterialProcesses:
    """Handles material transformations including metamorphism and weathering"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
    
    def apply_metamorphism(self):
        """Apply metamorphic transformations based on pressure and temperature"""
        # Find all non-space cells
        non_space_mask = (self.sim.material_types != MaterialType.SPACE)
        non_space_coords = np.where(non_space_mask)
        
        if len(non_space_coords[0]) == 0:
            return
        
        # Check each cell for possible transitions
        for i in range(len(non_space_coords[0])):
            y, x = non_space_coords[0][i], non_space_coords[1][i]
            current_material = self.sim.material_types[y, x]
            current_temp = self.sim.temperature[y, x] - 273.15  # Convert to Celsius
            current_pressure = self.sim.pressure[y, x]
            
            # Get applicable transition
            material_props = self.sim.material_db.get_properties(current_material)
            transition = material_props.get_applicable_transition(current_temp, current_pressure)
            
            if transition:
                # Apply transition
                self.sim.material_types[y, x] = transition.target
                self.sim._properties_dirty = True
                
                # Special handling for melting - set temperature appropriately
                if transition.target == MaterialType.MAGMA:
                    # Set magma temperature just above melting point
                    melting_temp = transition.min_temp + 273.15
                    self.sim.temperature[y, x] = max(self.sim.temperature[y, x], melting_temp + 50)
                
                # Special handling for cooling magma
                elif current_material == MaterialType.MAGMA:
                    # Determine cooling product based on conditions
                    cooling_product = self.sim.material_db.get_cooling_product(
                        current_temp, current_pressure, "mafic"
                    )
                    self.sim.material_types[y, x] = cooling_product
                    
                    # Set temperature to room temperature for new solid
                    self.sim.temperature[y, x] = 300.0  # Room temperature
    
    def _get_surface_mask(self):
        """Get mask of surface cells (adjacent to space or atmosphere)"""
        # Find space and atmospheric cells
        space_mask = (self.sim.material_types == MaterialType.SPACE)
        atmosphere_mask = ((self.sim.material_types == MaterialType.AIR) | 
                          (self.sim.material_types == MaterialType.WATER_VAPOR))
        
        # Non-space cells (solid/liquid materials)
        non_space_mask = ~space_mask
        
        # Surface cells are non-space cells adjacent to space or atmosphere
        # Use binary dilation to find cells adjacent to space/atmosphere
        adjacent_to_void = ndimage.binary_dilation(
            space_mask | atmosphere_mask, 
            structure=np.ones((3, 3), dtype=bool)
        )
        
        # Surface mask: non-space cells that are adjacent to space/atmosphere
        surface_mask = non_space_mask & adjacent_to_void
        
        return surface_mask

    def apply_weathering(self):
        """Apply weathering processes to surface materials.
        
        Uses vectorized operations for fast processing of all surface cells.
        """
        # Skip if weathering is disabled
        if not getattr(self.sim, 'enable_weathering', True):
            return False
            
        # Find surface cells (adjacent to space or atmosphere)
        surface_mask = self._get_surface_mask()
        
        if not np.any(surface_mask):
            return False
            
        # Get temperature and pressure arrays
        temp_celsius = self.sim.temperature - 273.15
        pressure_mpa = self.sim.pressure / 1e6  # Convert Pa to MPa
        
        changes_made = False
        
        # Process each material type that's present on the surface
        unique_surface_materials = set(self.sim.material_types[surface_mask].flat)
        
        for material_type in unique_surface_materials:
            # Skip atmospheric materials
            if material_type in {MaterialType.SPACE, MaterialType.AIR, MaterialType.WATER_VAPOR, 
                               MaterialType.WATER, MaterialType.ICE}:
                continue
                
            # Get material mask for this type on the surface
            material_surface_mask = surface_mask & (self.sim.material_types == material_type)
            
            if not np.any(material_surface_mask):
                continue
                
            # Get material properties and weathering transitions
            material_props = self.sim.material_db.get_properties(material_type)
            weathering_transitions = [t for t in material_props.transitions 
                                    if t.min_pressure <= 10 and t.probability < 1.0]
            
            if not weathering_transitions:
                continue
                
            # Get coordinates for vectorized operations
            surface_coords = np.where(material_surface_mask)
            if len(surface_coords[0]) == 0:
                continue
                
            # Vectorized environmental enhancement calculations
            temps = temp_celsius[surface_coords]
            pressures = pressure_mpa[surface_coords]
            
            # 1. Chemical weathering enhancement (vectorized)
            chemical_enhancement = np.ones_like(temps)
            above_freezing = temps > 0
            chemical_enhancement[above_freezing] = np.exp((temps[above_freezing] - 15) / 14.4)
             
            # 2. Water presence enhancement (fully vectorized using convolution)
            water_mask = (self.sim.material_types == MaterialType.WATER)
            
            # Use 3x3 kernel to detect water neighbors
            kernel_3x3 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
            water_neighbors = ndimage.convolve(water_mask.astype(np.float32), kernel_3x3, mode='constant', cval=0)
             
            # Get water enhancement for surface cells of this material
            has_water_neighbor = water_neighbors[surface_coords] > 0
            water_enhancement = np.where(has_water_neighbor, 3.0, 1.0)
                        
            # 3. Physical weathering enhancement (vectorized)
            physical_enhancement = np.ones_like(temps)
            
            # Freeze-thaw weathering
            freeze_thaw_zone = (-10 <= temps) & (temps <= 10)
            freeze_thaw_intensity = 1.0 - np.abs(temps[freeze_thaw_zone] / 10.0)
            physical_enhancement[freeze_thaw_zone] += freeze_thaw_intensity * 2.0
            
            # Thermal expansion weathering
            extreme_temp = (temps > 40) | (temps < -20)
            thermal_stress = np.minimum(np.abs(temps[extreme_temp] - 20) / 100.0, 1.0)
            physical_enhancement[extreme_temp] += thermal_stress
            
            # 4. Total enhancement factor (vectorized)
            total_enhancement = chemical_enhancement * water_enhancement * physical_enhancement
            
            # Apply weathering transitions (vectorized probability checks)
            for transition in weathering_transitions:
                # Check which cells meet P-T conditions for this transition
                applicable_mask = ((temps >= transition.min_temp) & 
                                 (temps <= transition.max_temp) & 
                                 (pressures >= transition.min_pressure) & 
                                 (pressures <= transition.max_pressure))
                
                if not np.any(applicable_mask):
                    continue
                    
                # Enhanced probability for applicable cells
                enhanced_probabilities = np.minimum(0.1, transition.probability * total_enhancement[applicable_mask])
                
                # Vectorized random check
                random_values = np.random.random(np.sum(applicable_mask))
                weathering_occurs = random_values < enhanced_probabilities
                
                if np.any(weathering_occurs):
                    # Get indices of cells that weather
                    applicable_indices = np.where(applicable_mask)[0]
                    weathering_indices = applicable_indices[weathering_occurs]
                    
                    # Apply weathering to selected cells
                    weather_y = surface_coords[0][weathering_indices]
                    weather_x = surface_coords[1][weathering_indices]
                    
                    # Update material types
                    self.sim.material_types[weather_y, weather_x] = transition.target
                    self.sim._properties_dirty = True
                    changes_made = True
                    
                    # # Set appropriate temperatures for new materials
                    # if transition.target == MaterialType.WATER:
                    #     # Keep current temperature but ensure it's above freezing
                    #     self.sim.temperature[weather_y, weather_x] = np.maximum(
                    #         self.sim.temperature[weather_y, weather_x], 275.0)
                    
                    # # Slightly reduce temperature due to endothermic weathering reactions
                    # self.sim.temperature[weather_y, weather_x] = np.maximum(
                    #     self.sim.temperature[weather_y, weather_x] - 2, 2.7)
                    
                    # Only one transition per cell per timestep - remove weathered cells from further processing
                    remaining_mask = np.ones(len(surface_coords[0]), dtype=bool)
                    remaining_mask[weathering_indices] = False
                    if np.any(remaining_mask):
                        surface_coords = (surface_coords[0][remaining_mask], surface_coords[1][remaining_mask])
                        temps = temps[remaining_mask]
                        pressures = pressures[remaining_mask]
                        chemical_enhancement = chemical_enhancement[remaining_mask]
                        water_enhancement = water_enhancement[remaining_mask]
                        physical_enhancement = physical_enhancement[remaining_mask]
                        total_enhancement = total_enhancement[remaining_mask]
                    else:
                        break  # All cells have been weathered
                        
        return changes_made
    
    def apply_phase_transitions(self):
        """Apply phase transitions (water/ice/vapor, etc.)"""
        # Find water-related materials
        water_mask = (self.sim.material_types == MaterialType.WATER)
        ice_mask = (self.sim.material_types == MaterialType.ICE)
        vapor_mask = (self.sim.material_types == MaterialType.WATER_VAPOR)
        
        # Water to ice (freezing)
        if np.any(water_mask):
            water_coords = np.where(water_mask)
            freezing_mask = self.sim.temperature[water_coords] < 273.15
            
            if np.any(freezing_mask):
                freeze_y = water_coords[0][freezing_mask]
                freeze_x = water_coords[1][freezing_mask]
                self.sim.material_types[freeze_y, freeze_x] = MaterialType.ICE
                self.sim._properties_dirty = True
        
        # Ice to water (melting)
        if np.any(ice_mask):
            ice_coords = np.where(ice_mask)
            melting_mask = self.sim.temperature[ice_coords] > 273.15
            
            if np.any(melting_mask):
                melt_y = ice_coords[0][melting_mask]
                melt_x = ice_coords[1][melting_mask]
                self.sim.material_types[melt_y, melt_x] = MaterialType.WATER
                self.sim._properties_dirty = True
        
        # Water to vapor (evaporation)
        if np.any(water_mask):
            water_coords = np.where(water_mask)
            evaporation_mask = self.sim.temperature[water_coords] > 373.15
            
            if np.any(evaporation_mask):
                evap_y = water_coords[0][evaporation_mask]
                evap_x = water_coords[1][evaporation_mask]
                self.sim.material_types[evap_y, evap_x] = MaterialType.WATER_VAPOR
                self.sim._properties_dirty = True
        
        # Vapor to water (condensation)
        if np.any(vapor_mask):
            vapor_coords = np.where(vapor_mask)
            condensation_mask = self.sim.temperature[vapor_coords] < 373.15
            
            if np.any(condensation_mask):
                cond_y = vapor_coords[0][condensation_mask]
                cond_x = vapor_coords[1][condensation_mask]
                self.sim.material_types[cond_y, cond_x] = MaterialType.WATER
                self.sim._properties_dirty = True
    
    @staticmethod
    def create_binding_matrix(solid_binding_force: float = 2e-4) -> tuple[np.ndarray, dict]:
        """Create material binding matrix for force-based swapping calculations.
        
        Args:
            solid_binding_force: Reference cohesion force density between solid voxels (N/m³)
            
        Returns:
            Tuple of (binding_matrix, material_index_map)
            - binding_matrix: N×N array where N is number of MaterialTypes (force density in N/m³)
            - material_index_map: Dict mapping MaterialType to matrix index
        """
        # Get all material types
        mt_list = list(MaterialType)
        n_mat = len(mt_list)
        binding_matrix = np.zeros((n_mat, n_mat), dtype=np.float32)
        
        # Create index mapping
        material_index_map = {m: i for i, m in enumerate(mt_list)}
        
        # Define fluid materials (including SPACE which acts as fluid)
        fluid_set = {MaterialType.AIR, MaterialType.WATER_VAPOR, MaterialType.WATER, MaterialType.MAGMA, MaterialType.SPACE}
        
        # Fill binding matrix
        for i, material_a in enumerate(mt_list):
            for j, material_b in enumerate(mt_list):
                if (material_a in fluid_set) and (material_b in fluid_set):
                    # Fluid-fluid interactions: no binding
                    binding_matrix[i, j] = 0.0
                elif (material_a in fluid_set) ^ (material_b in fluid_set):
                    # Fluid-solid interactions: half binding (SPACE counts as fluid)
                    binding_matrix[i, j] = 0.5 * solid_binding_force
                else:
                    # Solid-solid interactions: full binding
                    binding_matrix[i, j] = solid_binding_force
        
        return binding_matrix, material_index_map

    def apply_thermal_expansion(self):
        """Apply thermal expansion effects (placeholder for future implementation)"""
        # This could modify density based on temperature changes
        # For now, this is a placeholder for future thermal expansion physics
        pass 
