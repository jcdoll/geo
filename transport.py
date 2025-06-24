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
        
        # Update all volume fractions (except space at index 0)
        self.state.vol_frac[1:, :, :] -= (flux_div_x[1:] + flux_div_y[1:])
        
        # Store water flux for visualization (if needed)
        water_idx = MaterialType.WATER.value
        if water_idx < self.state.n_materials:
            self.state.mass_flux_x[:] = self.flux_x_all[water_idx]
            self.state.mass_flux_y[:] = self.flux_y_all[water_idx]
        
        # Ensure constraints
        self.state.normalize_volume_fractions()
        
        # Update mixture properties BEFORE temperature advection
        # This is critical because thermal mass changes when materials move
        from materials import MaterialDatabase
        mat_db = MaterialDatabase()
        self.state.update_mixture_properties(mat_db)
        
        # CRITICAL: Advect temperature (energy conservation)
        # This implements the ∇·(ρcₚTv) term from the heat equation
        self._advect_temperature(dt)
        
    def _advect_temperature(self, dt: float):
        """
        Advect temperature using the same upwind scheme as material transport.
        
        This implements the advective term ∇·(ρcₚTv) from the heat equation.
        Temperature is advected as a conserved quantity weighted by thermal mass.
        """
        dx = self.state.dx
        
        # Use the actual mixed thermal mass from the state
        # This is already computed by update_mixture_properties()
        thermal_mass_field = self.state.density * self.state.specific_heat
        
        # Compute energy density E = ρ * cp * T using actual mixed properties
        energy_density = thermal_mass_field * self.state.temperature
        
        # Apply upwind advection to energy density
        # X-direction
        energy_flux_x = np.zeros((self.state.ny, self.state.nx + 1), dtype=np.float32)
        energy_flux_x[:, 1:-1] = np.where(
            self.vx_face[:, 1:-1] > 0,
            energy_density[:, :-1] * self.vx_face[:, 1:-1],
            energy_density[:, 1:] * self.vx_face[:, 1:-1]
        ) * dt / dx
        
        # Y-direction
        energy_flux_y = np.zeros((self.state.ny + 1, self.state.nx), dtype=np.float32)
        energy_flux_y[1:-1, :] = np.where(
            self.vy_face[1:-1, :] > 0,
            energy_density[:-1, :] * self.vy_face[1:-1, :],
            energy_density[1:, :] * self.vy_face[1:-1, :]
        ) * dt / dx
        
        # Update energy density
        energy_density -= (energy_flux_x[:, 1:] - energy_flux_x[:, :-1] + 
                          energy_flux_y[1:, :] - energy_flux_y[:-1, :])
        
        # After material advection, we need to update the mixed properties
        # This will be done by the main simulation loop
        # For now, use the current thermal mass (which may be slightly stale)
        # The key is to use the ACTUAL mixed thermal mass, not the sum of individual materials
        
        # Update temperature: T = E / (ρ * cp)
        # Only update cells with significant thermal mass
        mask = thermal_mass_field > 1.0
        if np.any(mask):
            self.state.temperature[mask] = energy_density[mask] / thermal_mass_field[mask]
        
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