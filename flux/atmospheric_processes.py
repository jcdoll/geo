"""
Atmospheric processes module for flux-based geological simulation.
Handles atmospheric convection, mixing, and greenhouse effects.
"""

import numpy as np
from typing import Optional
from state import FluxState
from materials import MaterialType, MaterialDatabase


class AtmosphericProcesses:
    """Handles atmospheric processes including convection and greenhouse effects."""
    
    def __init__(self, state: FluxState):
        """
        Initialize atmospheric processes module.
        
        Args:
            state: FluxState instance
        """
        self.state = state
        
        # Configuration parameters
        self.convection_mixing_rate = 0.1  # Fraction of temperature difference to mix per step
        self.buoyancy_threshold = 0.1  # Temperature difference threshold for buoyancy-driven mixing (K)
        self.greenhouse_vapor_scaling = 100.0  # Scaling factor for water vapor greenhouse effect
        self.base_greenhouse_factor = 0.1  # Base greenhouse effect (fraction absorbed)
        self.max_greenhouse_factor = 0.9  # Maximum greenhouse effect
        
        # Material database
        self.material_db = MaterialDatabase()
        
    def apply_convection(self) -> None:
        """
        Apply atmospheric convection mixing based on temperature gradients.
        
        This implements buoyancy-driven convection where warmer air rises
        and cooler air sinks, leading to vertical mixing.
        """
        st = self.state
        
        # Find atmospheric cells (air and water vapor)
        is_atmosphere = np.zeros(st.n_materials, dtype=bool)
        is_atmosphere[MaterialType.AIR.value] = True
        is_atmosphere[MaterialType.WATER_VAPOR.value] = True
        
        # Get total atmospheric volume fraction per cell
        atmosphere_mask = np.sum(st.vol_frac[is_atmosphere], axis=0) > 0.5
        
        if not np.any(atmosphere_mask):
            return
        
        # Apply vertical mixing for unstable stratification
        # Work from bottom to top to handle rising warm air
        for y in range(st.ny - 2, 0, -1):  # Skip boundaries
            for x in range(1, st.nx - 1):
                if not atmosphere_mask[y, x] or not atmosphere_mask[y-1, x]:
                    continue
                
                # Check if lower cell is warmer (unstable stratification)
                temp_below = st.temperature[y, x]
                temp_above = st.temperature[y-1, x]
                
                if temp_below > temp_above + self.buoyancy_threshold:
                    # Mix temperatures
                    temp_diff = temp_below - temp_above
                    mixing_amount = temp_diff * self.convection_mixing_rate
                    
                    # Safety check - ensure finite values
                    if np.isfinite(mixing_amount) and np.isfinite(temp_below) and np.isfinite(temp_above):
                        # Apply mixing
                        st.temperature[y, x] -= mixing_amount * 0.5
                        st.temperature[y-1, x] += mixing_amount * 0.5
                    
                    # Also mix material fractions
                    for mat_idx in np.where(is_atmosphere)[0]:
                        frac_below = st.vol_frac[mat_idx, y, x]
                        frac_above = st.vol_frac[mat_idx, y-1, x]
                        frac_avg = 0.5 * (frac_below + frac_above)
                        
                        mix_frac = self.convection_mixing_rate
                        st.vol_frac[mat_idx, y, x] = (1 - mix_frac) * frac_below + mix_frac * frac_avg
                        st.vol_frac[mat_idx, y-1, x] = (1 - mix_frac) * frac_above + mix_frac * frac_avg
        
        # Horizontal mixing for atmospheric cells (weather systems)
        if np.any(atmosphere_mask):
            # Use convolution for efficient neighbor averaging
            kernel = np.array([[0.0, 0.25, 0.0],
                              [0.25, 0.0, 0.25],
                              [0.0, 0.25, 0.0]])
            
            # Average temperature of atmospheric neighbors
            atmo_temp = np.where(atmosphere_mask, st.temperature, 0)
            neighbor_sum = np.zeros_like(st.temperature)
            neighbor_count = np.zeros_like(st.temperature)
            
            # Manual convolution to handle boundaries properly
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    if abs(dy) + abs(dx) == 2:  # Skip diagonals
                        continue
                        
                    # Shift indices
                    y_src = slice(max(0, dy), min(st.ny, st.ny + dy))
                    x_src = slice(max(0, dx), min(st.nx, st.nx + dx))
                    y_dst = slice(max(0, -dy), min(st.ny, st.ny - dy))
                    x_dst = slice(max(0, -dx), min(st.nx, st.nx - dx))
                    
                    # Add neighbor contributions
                    neighbor_mask = atmosphere_mask[y_src, x_src]
                    neighbor_sum[y_dst, x_dst] += atmo_temp[y_src, x_src] * neighbor_mask
                    neighbor_count[y_dst, x_dst] += neighbor_mask
            
            # Apply horizontal mixing where we have atmospheric neighbors
            valid_mixing = (neighbor_count > 0) & atmosphere_mask
            if np.any(valid_mixing):
                avg_neighbor_temp = neighbor_sum[valid_mixing] / neighbor_count[valid_mixing]
                temp_diff = avg_neighbor_temp - st.temperature[valid_mixing]
                
                # Safety check - filter out NaN/inf values
                finite_mask = np.isfinite(temp_diff) & np.isfinite(avg_neighbor_temp)
                if np.any(finite_mask):
                    valid_indices = np.where(valid_mixing)
                    y_idx = valid_indices[0][finite_mask]
                    x_idx = valid_indices[1][finite_mask]
                    st.temperature[y_idx, x_idx] += self.convection_mixing_rate * 0.5 * temp_diff[finite_mask]
    
    def calculate_greenhouse_factor(self) -> float:
        """
        Calculate dynamic greenhouse effect based on atmospheric composition.
        
        Returns:
            greenhouse_factor: Fraction of outgoing radiation absorbed (0-1)
        """
        st = self.state
        
        # Get water vapor index
        vapor_idx = MaterialType.WATER_VAPOR.value
        
        # Calculate total water vapor mass
        total_vapor_mass = 0.0
        if vapor_idx < st.n_materials:
            # Water vapor mass = volume fraction * density * cell volume
            vapor_density = self.material_db.get_properties(MaterialType.WATER_VAPOR).density
            vapor_mass_per_cell = st.vol_frac[vapor_idx] * vapor_density * (st.dx ** 2)
            total_vapor_mass = np.sum(vapor_mass_per_cell)
        
        # Calculate greenhouse factor using logarithmic scaling to prevent runaway
        if total_vapor_mass > 0:
            # Use tanh to smoothly saturate the greenhouse effect
            vapor_factor = np.log1p(total_vapor_mass / self.greenhouse_vapor_scaling) / 10.0
            greenhouse_factor = (self.base_greenhouse_factor + 
                               (self.max_greenhouse_factor - self.base_greenhouse_factor) * 
                               np.tanh(vapor_factor))
        else:
            greenhouse_factor = self.base_greenhouse_factor
        
        return greenhouse_factor
    
    
    def get_atmospheric_column_density(self, x: int) -> float:
        """
        Calculate total atmospheric column density at a given x position.
        
        Args:
            x: X coordinate
            
        Returns:
            column_density: Total atmospheric mass per unit area (kg/m²)
        """
        st = self.state
        
        # Get atmospheric material indices
        is_atmosphere = np.zeros(st.n_materials, dtype=bool)
        is_atmosphere[MaterialType.AIR.value] = True
        is_atmosphere[MaterialType.WATER_VAPOR.value] = True
        
        # Calculate column density
        column_density = 0.0
        for y in range(st.ny):
            for mat_idx in np.where(is_atmosphere)[0]:
                mat_type = MaterialType(mat_idx)
                mat_density = self.material_db.get_properties(mat_type).density
                column_density += st.vol_frac[mat_idx, y, x] * mat_density * st.dx
        
        return column_density