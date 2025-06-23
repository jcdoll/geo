"""
Flux-based simulation state management.

This module manages the continuous field representations for the flux-based
geological simulation, replacing the discrete cell-based approach.
"""

from typing import Optional, Tuple, Dict
import numpy as np

from materials import MaterialType, MaterialDatabase


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
        n_materials: int = 9,  # Number of material types
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
        
        # Face-centered velocity components (MAC grid)
        # vx at vertical faces (ny, nx+1)
        self.velocity_x_face = np.zeros((ny, nx + 1), dtype=np.float32)
        # vy at horizontal faces (ny+1, nx)
        self.velocity_y_face = np.zeros((ny + 1, nx), dtype=np.float32)

        # Face-centred inverse-density coefficients β = 1/ρ used in variable-coefficient Poisson
        # β_x aligned with vx faces, β_y aligned with vy faces
        self.beta_x = np.zeros((ny, nx + 1), dtype=np.float32)
        self.beta_y = np.zeros((ny + 1, nx), dtype=np.float32)

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
            
        # Invert for harmonic mean (avoid division by zero)
        mask = self.thermal_conductivity > 1e-10
        self.thermal_conductivity[mask] = 1.0 / self.thermal_conductivity[mask]
        self.thermal_conductivity[~mask] = 0.0
        
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

    # ------------------------------------------------------------------
    # MAC-grid helper utilities
    # ------------------------------------------------------------------
    def update_face_coefficients(self):
        """Compute face-centred β = 1/ρ coefficients.

        β_x lives at vertical faces between cells (i-1,j) and (i,j).
        β_y lives at horizontal faces between cells (i,j-1) and (i,j).
        Uses harmonic averaging which is the correct discretisation for
        variable-coefficient Poisson problems in a staggered grid.
        """
        rho = self.density + 1e-12  # avoid divide-by-zero

        # β_x: shape (ny, nx+1)
        # Interior faces: harmonic mean of left & right cell densities
        # Special handling for space regions
        space_mask = rho < 0.1  # Space has essentially zero density
        
        # For faces between space and matter, use the matter density
        # For faces between two space cells, use a small value
        self.beta_x[:, 1:-1] = np.zeros((self.ny, self.nx-1))
        for j in range(self.ny):
            for i in range(1, self.nx):
                left_space = space_mask[j, i-1]
                right_space = space_mask[j, i]
                
                if left_space and right_space:
                    # Both cells are space - use small beta
                    self.beta_x[j, i] = 1000.0  # Large but finite
                elif left_space:
                    # Left is space, right is matter
                    self.beta_x[j, i] = 1.0 / rho[j, i]
                elif right_space:
                    # Right is space, left is matter
                    self.beta_x[j, i] = 1.0 / rho[j, i-1]
                else:
                    # Both are matter - harmonic mean
                    self.beta_x[j, i] = 2.0 / (rho[j, i-1] + rho[j, i])

        # Boundaries
        self.beta_x[:, 0] = 1.0 / np.maximum(rho[:, 0], 0.001)
        self.beta_x[:, -1] = 1.0 / np.maximum(rho[:, -1], 0.001)

        # β_y: shape (ny+1, nx)
        self.beta_y[1:-1, :] = np.zeros((self.ny-1, self.nx))
        for j in range(1, self.ny):
            for i in range(self.nx):
                top_space = space_mask[j-1, i]
                bottom_space = space_mask[j, i]
                
                if top_space and bottom_space:
                    # Both cells are space
                    self.beta_y[j, i] = 1000.0
                elif top_space:
                    # Top is space, bottom is matter
                    self.beta_y[j, i] = 1.0 / rho[j, i]
                elif bottom_space:
                    # Bottom is space, top is matter
                    self.beta_y[j, i] = 1.0 / rho[j-1, i]
                else:
                    # Both are matter - harmonic mean
                    self.beta_y[j, i] = 2.0 / (rho[j-1, i] + rho[j, i])
                    
        # Boundaries
        self.beta_y[0, :] = 1.0 / np.maximum(rho[0, :], 0.001)
        self.beta_y[-1, :] = 1.0 / np.maximum(rho[-1, :], 0.001)

    def update_face_velocities_from_cell(self):
        """Populate face-centred velocities by averaging neighbouring cells.

        This helper lets legacy cell-centred kernels keep writing
        `velocity_x` / `velocity_y` while new MAC-based routines can read the
        corresponding face arrays without incurring a large refactor in one
        step.
        """
        # vx faces (ny, nx+1)
        self.velocity_x_face[:, 1:-1] = 0.5 * (
            self.velocity_x[:, :-1] + self.velocity_x[:, 1:]
        )
        # For the two domain boundaries, just copy neighbouring cell value
        self.velocity_x_face[:, 0] = self.velocity_x[:, 0]
        self.velocity_x_face[:, -1] = self.velocity_x[:, -1]

        # vy faces (ny+1, nx)
        self.velocity_y_face[1:-1, :] = 0.5 * (
            self.velocity_y[:-1, :] + self.velocity_y[1:, :]
        )
        self.velocity_y_face[0, :] = self.velocity_y[0, :]
        self.velocity_y_face[-1, :] = self.velocity_y[-1, :]

    def update_cell_velocities_from_face(self):
        """Reconstruct cell-centred velocities by averaging neighbouring faces."""
        # vx cell-centred: average two surrounding vertical faces
        self.velocity_x = 0.5 * (
            self.velocity_x_face[:, :-1] + self.velocity_x_face[:, 1:]
        )

        # vy cell-centred: average two surrounding horizontal faces
        self.velocity_y = 0.5 * (
            self.velocity_y_face[:-1, :] + self.velocity_y_face[1:, :]
        )