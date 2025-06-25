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
        cell_depth: Optional[float] = None,  # Cell depth for 3D calculations
    ):
        """
        Initialize simulation state.
        
        Args:
            nx: Number of grid cells in x direction
            ny: Number of grid cells in y direction
            dx: Cell size in meters
            n_materials: Number of material types
            cell_depth: Cell depth in meters (defaults to domain width for cubic simulation)
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.n_materials = n_materials
        # Cell depth for 3D calculations (default to domain width for cubic simulation)
        self.cell_depth = cell_depth if cell_depth is not None else nx * dx
        
        # Time tracking
        self.time = 0.0
        self.dt = 0.0
        
        # Cell-centered quantities
        self.density = np.zeros((ny, nx), dtype=np.float32)
        self.velocity_x = np.zeros((ny, nx), dtype=np.float32)
        self.velocity_y = np.zeros((ny, nx), dtype=np.float32)
        self.temperature = np.zeros((ny, nx), dtype=np.float32)
        
        # CRITICAL: Pressure is ONLY calculated through velocity projection method
        # NEVER by integration, NEVER by hydrostatic approximation
        # ONLY through the incompressible flow projection in update_momentum()
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
        
        # Power density tracking for visualization (W/m³)
        self.power_density = np.zeros((ny, nx), dtype=np.float32)
        
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
        
    def update_temperature_for_new_materials(self, material_db):
        """Update temperature in cells where space has been created.
        
        When material leaves a cell and is replaced by space, we need to
        set the temperature to the CMB temperature (2.7K).
        """
        # Find cells that are mostly space (>90%)
        space_mask = self.vol_frac[0] > 0.9  # SPACE is index 0
        
        if np.any(space_mask):
            # Get space default temperature
            space_props = material_db.get_properties_by_index(0)
            self.temperature[space_mask] = space_props.default_temperature
        
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
        
        # First ensure volume fractions are normalized
        self.normalize_volume_fractions()
        
        # Compute mixture properties
        for mat_idx in range(self.n_materials):
            props = material_db.get_properties_by_index(mat_idx)
            phi = self.vol_frac[mat_idx]
            
            # Volume-weighted properties
            self.density += phi * props.density
            self.specific_heat += phi * props.specific_heat
            self.viscosity += phi * props.viscosity
            self.emissivity += phi * props.emissivity
            
        # Harmonic mean for thermal conductivity
        # k_mix = 1 / Σ(φᵢ/kᵢ)
        # IMPORTANT: Include space in calculation to properly reduce conductivity
        for mat_idx in range(self.n_materials):
            props = material_db.get_properties_by_index(mat_idx)
            phi = self.vol_frac[mat_idx]
            
            # For materials with very low conductivity, use a minimum value
            # to avoid numerical issues while still reducing heat transfer
            k_eff = max(props.thermal_conductivity, 1e-6)
            self.thermal_conductivity += phi / k_eff
            
        # Invert for harmonic mean (avoid division by zero)
        mask = self.thermal_conductivity > 1e-10
        self.thermal_conductivity[mask] = 1.0 / self.thermal_conductivity[mask]
        self.thermal_conductivity[~mask] = 1e-6  # Minimum conductivity
        
    def get_total_mass(self) -> float:
        """Calculate total mass in the system."""
        # Use float64 to avoid overflow with large volumes
        cell_volume = self.dx * self.dx * self.cell_depth
        return float(np.sum(self.density.astype(np.float64)) * cell_volume)
        
    def get_total_energy(self) -> float:
        """Calculate total thermal energy in the system."""
        # Use float64 for intermediate calculations to avoid overflow
        energy_density = self.density.astype(np.float64) * self.specific_heat.astype(np.float64) * self.temperature.astype(np.float64)
        cell_volume = self.dx * self.dx * self.cell_depth
        return float(np.sum(energy_density) * cell_volume)
        
    def get_material_inventory(self) -> Dict[int, float]:
        """Get total volume of each material type by index."""
        inventory = {}
        cell_volume = self.dx * self.dx * self.cell_depth
        
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
        # For beta calculation, use actual density everywhere
        # This allows materials to move through space
        rho_effective = np.copy(self.density)
        
        # Add small epsilon to avoid division by zero
        # Use space density as minimum to ensure physical behavior
        mat_db = MaterialDatabase()
        space_density = mat_db.get_properties(MaterialType.SPACE).density
        rho_effective = np.maximum(rho_effective, space_density)

        # β_x: shape (ny, nx+1)
        # Interior faces: harmonic mean of left & right cell densities
        self.beta_x[:, 1:-1] = 2.0 / (rho_effective[:, :-1] + rho_effective[:, 1:])
        self.beta_x[:, 0] = 1.0 / rho_effective[:, 0]
        self.beta_x[:, -1] = 1.0 / rho_effective[:, -1]

        # β_y: shape (ny+1, nx)
        self.beta_y[1:-1, :] = 2.0 / (rho_effective[:-1, :] + rho_effective[1:, :])
        self.beta_y[0, :] = 1.0 / rho_effective[0, :]
        self.beta_y[-1, :] = 1.0 / rho_effective[-1, :]

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
        # Enforce no-penetration boundary conditions (zero normal velocity)
        self.velocity_x_face[:, 0] = 0.0    # Left boundary
        self.velocity_x_face[:, -1] = 0.0   # Right boundary

        # vy faces (ny+1, nx)
        self.velocity_y_face[1:-1, :] = 0.5 * (
            self.velocity_y[:-1, :] + self.velocity_y[1:, :]
        )
        # Enforce no-penetration boundary conditions (zero normal velocity)
        self.velocity_y_face[0, :] = 0.0    # Top boundary
        self.velocity_y_face[-1, :] = 0.0   # Bottom boundary

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