"""
Flux-based simulation state management.

This module manages the continuous field representations for the flux-based
geological simulation, replacing the discrete cell-based approach.
"""

from typing import Optional, Tuple, Dict
import numpy as np


class FluxState:
    """
    Manages continuous field state for flux-based simulation.
    
    Key differences from CA approach:
    - Volume fractions φᵢ(x,y,t) for each material instead of single material per cell
    - Continuous density field computed from mixture properties
    - Face-centered flux arrays for conservation
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 50.0,
        n_materials: int = len(MaterialType),
    ):
        """
        Initialize simulation state.
        
        Args:
            nx: Number of grid cells in x direction
            ny: Number of grid cells in y direction
            dx: Cell size in meters
            n_materials: Number of material types
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.n_materials = n_materials
        
        # Time tracking
        self.time = 0.0
        self.dt = 0.0
        
        # Cell-centered quantities
        self.density = np.zeros((ny, nx), dtype=np.float32)
        self.velocity_x = np.zeros((ny, nx), dtype=np.float32)
        self.velocity_y = np.zeros((ny, nx), dtype=np.float32)
        self.temperature = np.zeros((ny, nx), dtype=np.float32)
        self.pressure = np.zeros((ny, nx), dtype=np.float32)
        
        # Volume fractions for each material [n_materials, ny, nx]
        # Constraint: sum over materials equals 1 at each point
        self.vol_frac = np.zeros((n_materials, ny, nx), dtype=np.float32)
        
        # Initialize with space/vacuum everywhere (index 0)
        self.vol_frac[0] = 1.0
        
        # Face-centered flux arrays for conservation
        # x-direction fluxes at (i+1/2, j) faces
        self.mass_flux_x = np.zeros((ny, nx+1), dtype=np.float32)
        self.heat_flux_x = np.zeros((ny, nx+1), dtype=np.float32)
        
        # y-direction fluxes at (i, j+1/2) faces  
        self.mass_flux_y = np.zeros((ny+1, nx), dtype=np.float32)
        self.heat_flux_y = np.zeros((ny+1, nx), dtype=np.float32)
        
        # Material property arrays (mixture properties)
        self.thermal_conductivity = np.zeros((ny, nx), dtype=np.float32)
        self.specific_heat = np.zeros((ny, nx), dtype=np.float32)
        self.viscosity = np.zeros((ny, nx), dtype=np.float32)
        self.emissivity = np.zeros((ny, nx), dtype=np.float32)
        
        # Heat source/sink tracking
        self.heat_source = np.zeros((ny, nx), dtype=np.float32)
        
        # Gravity field
        self.gravity_x = np.zeros((ny, nx), dtype=np.float32)
        self.gravity_y = np.zeros((ny, nx), dtype=np.float32)
        
    def normalize_volume_fractions(self):
        """Ensure volume fractions sum to 1 at each point."""
        total = self.vol_frac.sum(axis=0)
        # Avoid division by zero
        total = np.where(total > 0, total, 1.0)
        self.vol_frac /= total[np.newaxis, :, :]
        
    def update_mixture_properties(self, material_db):
        """
        Update mixture properties from volume fractions.
        
        Args:
            material_db: Material database with properties for each material type
        """
        # Reset arrays
        self.density.fill(0.0)
        self.thermal_conductivity.fill(0.0)
        self.specific_heat.fill(0.0)
        self.viscosity.fill(0.0)
        self.emissivity.fill(0.0)
        
        # Compute mixture properties
        for mat_idx in range(self.n_materials):
            if mat_idx == 0:  # Skip space (index 0)
                continue
                
            props = material_db.get_properties_by_index(mat_idx)
            phi = self.vol_frac[mat_idx]
            
            # Volume-weighted properties
            self.density += phi * props.density
            self.specific_heat += phi * props.specific_heat
            self.viscosity += phi * props.viscosity
            self.emissivity += phi * props.emissivity
            
        # Harmonic mean for thermal conductivity
        # k_mix = 1 / Σ(φᵢ/kᵢ)
        for mat_idx in range(self.n_materials):
            if mat_idx == 0:  # Skip space
                continue
                
            props = material_db.get_properties_by_index(mat_idx)
            phi = self.vol_frac[mat_idx]
            
            # Add small epsilon to avoid division by zero
            self.thermal_conductivity += phi / (props.thermal_conductivity + 1e-10)
            
        # Invert for harmonic mean
        self.thermal_conductivity = np.where(
            self.thermal_conductivity > 0,
            1.0 / self.thermal_conductivity,
            0.0
        )
        
    def get_total_mass(self) -> float:
        """Calculate total mass in the system."""
        return np.sum(self.density) * self.dx * self.dx
        
    def get_total_energy(self) -> float:
        """Calculate total thermal energy in the system."""
        return np.sum(self.density * self.specific_heat * self.temperature) * self.dx * self.dx
        
    def get_material_inventory(self) -> Dict[int, float]:
        """Get total volume of each material type by index."""
        inventory = {}
        cell_volume = self.dx * self.dx
        
        for mat_idx in range(self.n_materials):
            total_volume = np.sum(self.vol_frac[mat_idx]) * cell_volume
            if total_volume > 0:
                inventory[mat_idx] = total_volume
                
        return inventory