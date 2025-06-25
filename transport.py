"""
Fully vectorized material transport using numpy broadcasting.

Eliminates all loops over materials by processing them simultaneously.
"""

import numpy as np
from state import FluxState
from materials import MaterialType


class FluxTransport:
    """Vectorized flux-based material transport using upwind scheme."""
    
    def __init__(self, state: FluxState):
        """
        Initialize transport solver.
        
        Args:
            state: FluxState instance
        """
        self.state = state
        
        # Pre-allocate arrays for all materials at once
        # Shape: (n_materials, ny, nx+1) for x-faces
        self.flux_x_all = np.zeros((state.n_materials, state.ny, state.nx + 1), dtype=np.float32)
        # Shape: (n_materials, ny+1, nx) for y-faces  
        self.flux_y_all = np.zeros((state.n_materials, state.ny + 1, state.nx), dtype=np.float32)
        
        # Face velocities (computed from cell-centered values)
        self.vx_face = np.zeros((state.ny, state.nx + 1), dtype=np.float32)
        self.vy_face = np.zeros((state.ny + 1, state.nx), dtype=np.float32)
        
    def compute_face_velocities(self):
        """
        Interpolate cell-centered velocities to faces.
        
        Uses simple averaging for interior faces.
        """
        # X-faces: average neighboring cells
        self.vx_face[:, 1:-1] = 0.5 * (
            self.state.velocity_x[:, :-1] + self.state.velocity_x[:, 1:]
        )
        # Boundary faces use one-sided values
        self.vx_face[:, 0] = self.state.velocity_x[:, 0]
        self.vx_face[:, -1] = self.state.velocity_x[:, -1]
        
        # Y-faces: average neighboring cells
        self.vy_face[1:-1, :] = 0.5 * (
            self.state.velocity_y[:-1, :] + self.state.velocity_y[1:, :]
        )
        # Boundary faces
        self.vy_face[0, :] = self.state.velocity_y[0, :]
        self.vy_face[-1, :] = self.state.velocity_y[-1, :]
        
    def advect_materials_vectorized(self, dt: float):
        """
        Fully vectorized material advection - no loops!
        
        Processes all materials simultaneously using broadcasting.
        
        Args:
            dt: Time step
        """
        # Compute face velocities once
        self.compute_face_velocities()
        
        dx = self.state.dx
        dt_dx = dt / dx
        
        # Get all material volume fractions at once
        # Shape: (n_materials, ny, nx)
        all_phi = self.state.vol_frac
        
        # Reset flux arrays
        self.flux_x_all.fill(0.0)
        self.flux_y_all.fill(0.0)
        
        # X-direction fluxes for ALL materials at once
        # Create masks for upwind scheme
        vx_positive = self.vx_face[:, 1:-1] > 0  # Shape: (ny, nx)
        
        # Broadcast to all materials
        # When vx > 0, use left cell value; when vx < 0, use right cell value
        # Shape operations:
        # all_phi[:, :, :-1] is (n_mat, ny, nx-1) 
        # vx_positive is (ny, nx)
        # Need to align dimensions properly
        
        # Expand vx_positive to match material dimension
        vx_pos_expanded = vx_positive[np.newaxis, :, :]  # (1, ny, nx)
        vx_expanded = self.vx_face[np.newaxis, :, 1:-1]  # (1, ny, nx)
        
        # Compute fluxes for interior faces
        self.flux_x_all[:, :, 1:-1] = np.where(
            vx_pos_expanded,
            all_phi[:, :, :-1],  # Use left cell when vx > 0
            all_phi[:, :, 1:]    # Use right cell when vx < 0
        ) * vx_expanded * dt_dx
        
        # Y-direction fluxes for ALL materials at once
        vy_positive = self.vy_face[1:-1, :] > 0  # Shape: (ny, nx)
        vy_pos_expanded = vy_positive[np.newaxis, :, :]  # (1, ny, nx)
        vy_expanded = self.vy_face[np.newaxis, 1:-1, :]  # (1, ny, nx)
        
        self.flux_y_all[:, 1:-1, :] = np.where(
            vy_pos_expanded,
            all_phi[:, :-1, :],  # Use bottom cell when vy > 0
            all_phi[:, 1:, :]    # Use top cell when vy < 0
        ) * vy_expanded * dt_dx
        
        # Apply flux divergence to ALL materials at once
        # No loop needed!
        flux_div_x = self.flux_x_all[:, :, 1:] - self.flux_x_all[:, :, :-1]
        flux_div_y = self.flux_y_all[:, 1:, :] - self.flux_y_all[:, :-1, :]
        
        # Store water flux for visualization (if needed)
        water_idx = MaterialType.WATER.value
        if water_idx < self.state.n_materials:
            self.state.mass_flux_x[:] = self.flux_x_all[water_idx]
            self.state.mass_flux_y[:] = self.flux_y_all[water_idx]
        
        # CRITICAL: Advect temperature BEFORE updating volume fractions
        # We need the OLD volume fractions to compute initial energy correctly
        old_vol_frac = self.state.vol_frac.copy()
        
        # Update all volume fractions (including space so materials can move through it)
        self.state.vol_frac[:, :, :] -= (flux_div_x + flux_div_y)
        
        # Now advect temperature using the material fluxes
        self._advect_temperature_with_materials(dt, self.flux_x_all, self.flux_y_all, old_vol_frac)
        
        # Ensure constraints
        self.state.normalize_volume_fractions()
        
        # Update mixture properties after everything is advected
        from materials import MaterialDatabase
        mat_db = MaterialDatabase()
        self.state.update_mixture_properties(mat_db)
        
    def _advect_temperature_with_materials(self, dt: float, material_flux_x: np.ndarray, 
                                          material_flux_y: np.ndarray, old_vol_frac: np.ndarray):
        """
        Advect temperature using conservative multi-material energy transport.
        
        This uses the SAME fluxes as material advection to ensure consistency.
        Energy is advected as Eᵢ = ρᵢ * cpᵢ * T for each material.
        
        Args:
            dt: Time step
            material_flux_x: Material volume fluxes in x-direction (already includes dt/dx)
            material_flux_y: Material volume fluxes in y-direction (already includes dt/dx)
            old_vol_frac: Volume fractions BEFORE material advection
        """
        # Get material properties
        from materials import MaterialDatabase
        mat_db = MaterialDatabase()
        
        # Pre-compute thermal properties for each material
        rho_mat = np.zeros(self.state.n_materials)
        cp_mat = np.zeros(self.state.n_materials)
        for i in range(self.state.n_materials):
            props = mat_db.get_properties_by_index(i)
            rho_mat[i] = props.density
            cp_mat[i] = props.specific_heat
        
        # Compute energy flux using material flux and temperature
        # Energy flux = material_flux * ρ * cp * T
        # We use upwind temperature based on velocity direction
        
        # X-direction energy flux
        energy_flux_x = np.zeros((self.state.n_materials, self.state.ny, self.state.nx + 1))
        for i in range(self.state.n_materials):
            if rho_mat[i] * cp_mat[i] > 1.0:
                # Temperature at faces (upwind)
                T_face = np.zeros((self.state.ny, self.state.nx + 1))
                T_face[:, 1:-1] = np.where(
                    self.vx_face[:, 1:-1] > 0,
                    self.state.temperature[:, :-1],  # Left cell temp
                    self.state.temperature[:, 1:]     # Right cell temp
                )
                T_face[:, 0] = self.state.temperature[:, 0]
                T_face[:, -1] = self.state.temperature[:, -1]
                
                # Energy flux = volume flux * density * cp * T
                energy_flux_x[i] = material_flux_x[i] * rho_mat[i] * cp_mat[i] * T_face
        
        # Y-direction energy flux
        energy_flux_y = np.zeros((self.state.n_materials, self.state.ny + 1, self.state.nx))
        for i in range(self.state.n_materials):
            if rho_mat[i] * cp_mat[i] > 1.0:
                # Temperature at faces (upwind)
                T_face = np.zeros((self.state.ny + 1, self.state.nx))
                T_face[1:-1, :] = np.where(
                    self.vy_face[1:-1, :] > 0,
                    self.state.temperature[:-1, :],  # Bottom cell temp
                    self.state.temperature[1:, :]     # Top cell temp
                )
                T_face[0, :] = self.state.temperature[0, :]
                T_face[-1, :] = self.state.temperature[-1, :]
                
                # Energy flux = volume flux * density * cp * T
                energy_flux_y[i] = material_flux_y[i] * rho_mat[i] * cp_mat[i] * T_face
        
        # Compute initial energy in each cell (before advection)
        initial_energy = np.zeros_like(self.state.temperature)
        for i in range(self.state.n_materials):
            if rho_mat[i] * cp_mat[i] > 1.0:
                initial_energy += old_vol_frac[i] * rho_mat[i] * cp_mat[i] * self.state.temperature
        
        # Apply flux divergence to get final energy
        # Note: material_flux already includes dt/dx factor
        energy_flux_div_x = energy_flux_x[:, :, 1:] - energy_flux_x[:, :, :-1]
        energy_flux_div_y = energy_flux_y[:, 1:, :] - energy_flux_y[:, :-1, :]
        
        # Total energy change
        total_energy_change = -np.sum(energy_flux_div_x + energy_flux_div_y, axis=0)
        final_energy = initial_energy + total_energy_change
        
        # Now materials have been advected, compute new thermal mass
        # Note: self.state.vol_frac has been updated by material advection
        new_thermal_mass = np.zeros_like(self.state.temperature)
        for i in range(self.state.n_materials):
            if rho_mat[i] * cp_mat[i] > 1.0:
                new_thermal_mass += self.state.vol_frac[i] * rho_mat[i] * cp_mat[i]
        
        # Update temperature: T = E / (thermal mass)
        mask = new_thermal_mass > 1.0
        if np.any(mask):
            self.state.temperature[mask] = final_energy[mask] / new_thermal_mass[mask]
        
        # For cells with no thermal mass (pure space), keep existing temperature
        # This prevents division by zero and maintains physical behavior
        
        # Ensure temperature stays physical
        self.state.temperature = np.maximum(self.state.temperature, 0.0)
        
    def compute_courant_number(self) -> float:
        """
        Compute maximum Courant number for stability check.
        
        Returns:
            Maximum CFL number across all cells
        """
        self.compute_face_velocities()
        
        # Maximum velocities
        max_vx = np.max(np.abs(self.vx_face))
        max_vy = np.max(np.abs(self.vy_face))
        
        # CFL = max(|v|) * dt / dx
        # We return max velocity; caller multiplies by dt/dx
        return max(max_vx, max_vy)