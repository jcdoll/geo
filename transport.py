"""
Flux-based transport calculations for geological simulation.

This module handles material advection using a finite-volume approach
with vectorized operations for performance.
"""

import numpy as np
from typing import Tuple
from state import FluxState
from materials import MaterialType


class FluxTransport:
    """Handles material transport using finite-volume flux calculations."""
    
    def __init__(self, state: FluxState):
        """
        Initialize transport calculator.
        
        Args:
            state: FluxState instance to operate on
        """
        self.state = state
        
        # Pre-allocate work arrays for face velocities
        self.vx_face = np.zeros((state.ny, state.nx+1), dtype=np.float32)
        self.vy_face = np.zeros((state.ny+1, state.nx), dtype=np.float32)
        
        # Pre-allocate flux arrays
        self.flux_x = np.zeros((state.ny, state.nx+1), dtype=np.float32)
        self.flux_y = np.zeros((state.ny+1, state.nx), dtype=np.float32)
        
    def compute_face_velocities(self):
        """Pre-compute face-centered velocities (only once per timestep)."""
        # X-faces: average of adjacent cells
        self.vx_face[:, 1:-1] = 0.5 * (self.state.velocity_x[:, :-1] + self.state.velocity_x[:, 1:])
        # Boundary conditions
        self.vx_face[:, 0] = self.state.velocity_x[:, 0]
        self.vx_face[:, -1] = self.state.velocity_x[:, -1]
        
        # Y-faces: average of adjacent cells  
        self.vy_face[1:-1, :] = 0.5 * (self.state.velocity_y[:-1, :] + self.state.velocity_y[1:, :])
        # Boundary conditions
        self.vy_face[0, :] = self.state.velocity_y[0, :]
        self.vy_face[-1, :] = self.state.velocity_y[-1, :]
        
    def compute_material_flux_vectorized(self, mat: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute flux for a specific material using fully vectorized operations.
        
        Args:
            mat: Material type index
            dt: Time step
            
        Returns:
            (flux_x, flux_y): Material volume fraction fluxes
        """
        dx = self.state.dx
        phi = self.state.vol_frac[mat]
        
        # X-direction flux (vectorized)
        # Upwind scheme: use left cell if v>0, right cell if v<0
        self.flux_x.fill(0.0)
        
        # Interior faces only (boundaries have zero flux)
        self.flux_x[:, 1:-1] = np.where(
            self.vx_face[:, 1:-1] > 0,
            phi[:, :-1] * self.vx_face[:, 1:-1],  # Use left cell
            phi[:, 1:] * self.vx_face[:, 1:-1]    # Use right cell
        ) * dt / dx
        
        # Y-direction flux (vectorized)
        self.flux_y.fill(0.0)
        
        # Interior faces only
        self.flux_y[1:-1, :] = np.where(
            self.vy_face[1:-1, :] > 0,
            phi[:-1, :] * self.vy_face[1:-1, :],  # Use bottom cell
            phi[1:, :] * self.vy_face[1:-1, :]    # Use top cell
        ) * dt / dx
        
        return self.flux_x.copy(), self.flux_y.copy()
        
    def advect_materials_vectorized(self, dt: float):
        """
        Transport all materials using vectorized operations.
        
        This is the main optimization - compute face velocities once,
        then use vectorized operations for all materials.
        
        Args:
            dt: Time step
        """
        # Compute face velocities once for all materials
        self.compute_face_velocities()
        
        # Process all materials at once using vectorized operations
        dx = self.state.dx
        
        # For each material (skip space at index 0)
        for mat in range(1, self.state.n_materials):
            phi = self.state.vol_frac[mat]
            
            # X-direction flux
            self.flux_x.fill(0.0)
            # Interior faces only (skip boundaries)
            self.flux_x[:, 1:-1] = np.where(
                self.vx_face[:, 1:-1] > 0,
                phi[:, :-1] * self.vx_face[:, 1:-1],
                phi[:, 1:] * self.vx_face[:, 1:-1]
            ) * dt / dx
            
            # Y-direction flux
            self.flux_y.fill(0.0)
            # Interior faces only (skip boundaries)
            self.flux_y[1:-1, :] = np.where(
                self.vy_face[1:-1, :] > 0,
                phi[:-1, :] * self.vy_face[1:-1, :],
                phi[1:, :] * self.vy_face[1:-1, :]
            ) * dt / dx
            
            # Apply flux divergence directly
            self.state.vol_frac[mat, :, :] -= (
                self.flux_x[:, 1:] - self.flux_x[:, :-1] +
                self.flux_y[1:, :] - self.flux_y[:-1, :]
            )
            
            # Store water flux for visualization
            if mat == MaterialType.WATER.value:
                self.state.mass_flux_x[:] = self.flux_x
                self.state.mass_flux_y[:] = self.flux_y
                
        # Ensure constraints
        self.state.normalize_volume_fractions()
        
    def advect_all_materials_single_pass(self, dt: float):
        """
        Ultra-optimized: Process all materials in a single pass.
        
        This computes fluxes for all materials simultaneously using
        broadcasting and advanced indexing.
        
        Args:
            dt: Time step
        """
        # Compute face velocities once
        self.compute_face_velocities()
        
        nx, ny = self.state.nx, self.state.ny
        dx = self.state.dx
        dt_dx = dt / dx
        
        # Get all material volume fractions (excluding space)
        all_phi = self.state.vol_frac[1:, :, :]  # Shape: (n_mat-1, ny, nx)
        
        # X-direction: Compute all fluxes at once
        # Shape: (n_mat-1, ny, nx+1)
        flux_x_all = np.zeros((self.state.n_materials-1, ny, nx+1), dtype=np.float32)
        
        # Vectorized upwind for all materials
        vx_positive = self.vx_face[:, 1:] > 0  # Shape: (ny, nx)
        
        for i, mat in enumerate(range(1, self.state.n_materials)):
            flux_x_all[i, :, 1:-1] = np.where(
                vx_positive[:, :-1],
                all_phi[i, :, :-1] * self.vx_face[:, 1:-1],
                all_phi[i, :, 1:] * self.vx_face[:, 1:-1]
            ) * dt_dx
            
        # Y-direction: Compute all fluxes at once
        flux_y_all = np.zeros((self.state.n_materials-1, ny+1, nx), dtype=np.float32)
        
        vy_positive = self.vy_face[1:, :] > 0  # Shape: (ny, nx)
        
        for i, mat in enumerate(range(1, self.state.n_materials)):
            flux_y_all[i, 1:-1, :] = np.where(
                vy_positive[:-1, :],
                all_phi[i, :-1, :] * self.vy_face[1:-1, :],
                all_phi[i, 1:, :] * self.vy_face[1:-1, :]
            ) * dt_dx
            
        # Apply all flux divergences at once
        for i, mat in enumerate(range(1, self.state.n_materials)):
            self.state.vol_frac[mat, :, :] -= (
                flux_x_all[i, :, 1:] - flux_x_all[i, :, :-1] +
                flux_y_all[i, 1:, :] - flux_y_all[i, :-1, :]
            )
            
            # Store water flux for visualization
            if mat == MaterialType.WATER.value:
                self.state.mass_flux_x[:] = flux_x_all[i]
                self.state.mass_flux_y[:] = flux_y_all[i]
                
        # Ensure constraints
        self.state.normalize_volume_fractions()
    def diffuse_heat(self, dt: float):
        """
        Apply thermal diffusion (delegated to base class for now).
        
        Args:
            dt: Time step
        """
        # For now, just use the base implementation
        # Could be optimized later if needed
        base_transport = FluxTransport(self.state)
        base_transport.diffuse_heat(dt)
