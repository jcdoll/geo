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
    
    def apply_weathering(self, enable_weathering: bool = False):
        """Apply surface weathering processes"""
        if not enable_weathering:
            return
        
        # Find surface cells (non-space cells adjacent to space)
        space_mask = (self.sim.material_types == MaterialType.SPACE)
        non_space_mask = ~space_mask
        
        # Surface cells are adjacent to space
        surface_candidates = ndimage.binary_dilation(space_mask, structure=self.sim._circular_kernel_3x3) & non_space_mask
        
        # Exclude atmospheric materials from weathering
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR) |
            (self.sim.material_types == MaterialType.WATER) |
            (self.sim.material_types == MaterialType.ICE)
        )
        surface_solid_mask = surface_candidates & ~atmosphere_mask
        
        if not np.any(surface_solid_mask):
            return
        
        # Apply weathering to a fraction of surface cells each step
        surface_coords = np.where(surface_solid_mask)
        num_surface = len(surface_coords[0])
        
        # Weather a small fraction each step (slow process)
        weathering_rate = 0.001  # 0.1% per step
        num_to_weather = max(1, int(num_surface * weathering_rate))
        
        if num_to_weather > 0:
            # Randomly select cells to weather
            selected_indices = np.random.choice(num_surface, num_to_weather, replace=False)
            
            for idx in selected_indices:
                y, x = surface_coords[0][idx], surface_coords[1][idx]
                current_material = self.sim.material_types[y, x]
                
                # Get weathering products
                weathering_products = self.sim.material_db.get_weathering_products(current_material)
                
                if weathering_products:
                    # Choose first weathering product (could be randomized)
                    new_material = weathering_products[0]
                    self.sim.material_types[y, x] = new_material
                    self.sim._properties_dirty = True
                    
                    # Set temperature to surface temperature
                    self.sim.temperature[y, x] = self.sim.surface_temperature
    
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
    
    def apply_thermal_expansion(self):
        """Apply thermal expansion effects (placeholder for future implementation)"""
        # This could modify density based on temperature changes
        # For now, this is a placeholder for future thermal expansion physics
        pass 