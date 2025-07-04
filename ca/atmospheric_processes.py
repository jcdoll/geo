"""
Atmospheric processes module for geological simulation.
Handles solar radiation, atmospheric absorption, and convection.
"""

import numpy as np
from scipy import ndimage
try:
    from .materials import MaterialType
    from .simulation_utils import SimulationUtils
except ImportError:
    from materials import MaterialType
    from simulation_utils import SimulationUtils


class AtmosphericProcesses:
    """Handles atmospheric processes including solar radiation and convection"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
    
    def apply_atmospheric_convection(self):
        """Atmospheric convection mixing - DISABLED
        
        Convection effects are now handled through enhanced thermal conductivity
        of atmospheric materials rather than explicit mixing.
        """
        pass
    
    
    def calculate_greenhouse_effect(self) -> float:
        """Calculate dynamic greenhouse effect based on water vapor content"""
        water_vapor_mask = (self.sim.material_types == MaterialType.WATER_VAPOR)
        total_water_vapor_mass = np.sum(self.sim.density[water_vapor_mask]) if np.any(water_vapor_mask) else 0.0
        
        # Scale greenhouse effect by water vapor content (logarithmic to prevent runaway)
        if total_water_vapor_mass > 0:
            vapor_factor = np.log1p(total_water_vapor_mass / self.sim.greenhouse_vapor_scaling) / 10.0
            greenhouse_factor = self.sim.base_greenhouse_effect + (self.sim.max_greenhouse_effect - self.sim.base_greenhouse_effect) * np.tanh(vapor_factor)
        else:
            greenhouse_factor = self.sim.base_greenhouse_effect
        
        return greenhouse_factor
    
    def solve_atmospheric_absorption_directional_sweep(self, non_space_mask: np.ndarray, 
                                                      solar_intensity_factor: np.ndarray,
                                                      effective_solar_constant: float, 
                                                      source_term: np.ndarray) -> np.ndarray:
        """Directional sweep (DDA) atmospheric absorption working for any solar angle"""
        initial_flux = effective_solar_constant * self.sim.planetary_distance_factor * solar_intensity_factor
        remaining_flux = np.zeros_like(initial_flux)
        
        ux, uy = self.sim._get_solar_direction()
        if ux == 0 and uy == 0:
            return initial_flux
        
        # DDA stepping direction: move OPPOSITE to incoming solar vector
        step_x = -1 if ux > 0 else 1
        step_y = -1 if uy > 0 else 1
        inv_dx = abs(1.0 / ux) if ux != 0 else float('inf')
        inv_dy = abs(1.0 / uy) if uy != 0 else float('inf')
        
        # Select all entry cells on the day-side boundary
        if abs(ux) >= abs(uy):  # shallow ray → enter from side opposite to ray direction
            entry_x = self.sim.width - 1 if ux > 0 else 0
            entry_cells = ((entry_x, y) for y in range(self.sim.height))
        else:  # steep ray → enter from top/bottom opposite to ray direction
            entry_y = self.sim.height - 1 if uy > 0 else 0
            entry_cells = ((x, entry_y) for x in range(self.sim.width))
        
        # Main DDA march
        for sx, sy in entry_cells:
            I = initial_flux[sy, sx]
            t_max_x = inv_dx
            t_max_y = inv_dy
            
            while 0 <= sx < self.sim.width and 0 <= sy < self.sim.height and I > 0:
                mat = self.sim.material_types[sy, sx]
                
                if mat != MaterialType.SPACE:
                    k = self.sim.material_db.get_solar_absorption(mat)
                    absorbed = I * k
                    
                    if absorbed > 0 and self.sim.density[sy, sx] > 0 and self.sim.specific_heat[sy, sx] > 0:
                        vol_power = absorbed / self.sim.cell_size
                        source_term[sy, sx] += vol_power / (self.sim.density[sy, sx] * self.sim.specific_heat[sy, sx])
                        self.sim.power_density[sy, sx] += vol_power
                        
                        # Categorise energy deposition for diagnostics
                        if mat in (MaterialType.AIR, MaterialType.WATER_VAPOR):
                            self.sim.thermal_fluxes['atmospheric_heating'] += vol_power * (self.sim.cell_size ** 3)
                        else:
                            self.sim.thermal_fluxes['solar_input'] += vol_power * (self.sim.cell_size ** 3)
                    
                    I -= absorbed
                    
                    if k >= 1.0 or I <= 0:
                        remaining_flux[sy, sx] += I
                        break  # ray terminated at opaque surface or fully absorbed
                
                # advance to next grid cell using DDA
                if t_max_x < t_max_y:
                    sx += step_x
                    t_max_x += inv_dx
                else:
                    sy += step_y
                    t_max_y += inv_dy
        
        return remaining_flux 
