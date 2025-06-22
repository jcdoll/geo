"""
Flux-based transport calculations.

This module handles all flux computations for mass, momentum, and energy transport
using a finite volume approach with upwind schemes for stability.
"""

import numpy as np
from typing import Tuple
from state import FluxState
from materials import MaterialType


class FluxTransport:
    """Handles flux-based transport calculations."""
    
    def __init__(self, state: FluxState):
        """
        Initialize transport calculator.
        
        Args:
            state: FluxState instance to operate on
        """
        self.state = state
        
    def compute_mass_flux(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mass flux through cell faces using upwind scheme.
        
        Args:
            dt: Time step
            
        Returns:
            (flux_x, flux_y): Mass fluxes at cell faces
        """
        nx, ny = self.state.nx, self.state.ny
        dx = self.state.dx
        
        # X-direction flux at face (i+1/2, j)
        flux_x = np.zeros((ny, nx+1), dtype=np.float32)
        
        # Face velocities (average of adjacent cells)
        vx_face = np.zeros((ny, nx+1), dtype=np.float32)
        vx_face[:, 1:-1] = 0.5 * (self.state.velocity_x[:, :-1] + self.state.velocity_x[:, 1:])
        
        # Upwind scheme
        for i in range(1, nx):
            # Positive velocity: flow from left cell
            mask_pos = vx_face[:, i] > 0
            flux_x[mask_pos, i] = (
                self.state.density[mask_pos, i-1] * vx_face[mask_pos, i] * dt / dx
            )
            
            # Negative velocity: flow from right cell
            mask_neg = vx_face[:, i] <= 0
            flux_x[mask_neg, i] = (
                self.state.density[mask_neg, i] * vx_face[mask_neg, i] * dt / dx
            )
        
        # Y-direction flux at face (i, j+1/2)
        flux_y = np.zeros((ny+1, nx), dtype=np.float32)
        
        # Face velocities
        vy_face = np.zeros((ny+1, nx), dtype=np.float32)
        vy_face[1:-1, :] = 0.5 * (self.state.velocity_y[:-1, :] + self.state.velocity_y[1:, :])
        
        # Upwind scheme
        for j in range(1, ny):
            # Positive velocity: flow from bottom cell
            mask_pos = vy_face[j, :] > 0
            flux_y[j, mask_pos] = (
                self.state.density[j-1, mask_pos] * vy_face[j, mask_pos] * dt / dx
            )
            
            # Negative velocity: flow from top cell
            mask_neg = vy_face[j, :] <= 0
            flux_y[j, mask_neg] = (
                self.state.density[j, mask_neg] * vy_face[j, mask_neg] * dt / dx
            )
            
        return flux_x, flux_y
        
    def compute_material_flux(self, mat: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute flux for a specific material.
        
        Args:
            mat: Material type index
            dt: Time step
            
        Returns:
            (flux_x, flux_y): Material volume fraction fluxes
        """
        nx, ny = self.state.nx, self.state.ny
        dx = self.state.dx
        phi = self.state.vol_frac[mat]
        
        # X-direction flux
        flux_x = np.zeros((ny, nx+1), dtype=np.float32)
        vx_face = np.zeros((ny, nx+1), dtype=np.float32)
        vx_face[:, 1:-1] = 0.5 * (self.state.velocity_x[:, :-1] + self.state.velocity_x[:, 1:])
        
        for i in range(1, nx):
            mask_pos = vx_face[:, i] > 0
            flux_x[mask_pos, i] = phi[mask_pos, i-1] * vx_face[mask_pos, i] * dt / dx
            
            mask_neg = vx_face[:, i] <= 0
            flux_x[mask_neg, i] = phi[mask_neg, i] * vx_face[mask_neg, i] * dt / dx
        
        # Y-direction flux
        flux_y = np.zeros((ny+1, nx), dtype=np.float32)
        vy_face = np.zeros((ny+1, nx), dtype=np.float32)
        vy_face[1:-1, :] = 0.5 * (self.state.velocity_y[:-1, :] + self.state.velocity_y[1:, :])
        
        for j in range(1, ny):
            mask_pos = vy_face[j, :] > 0
            flux_y[j, mask_pos] = phi[j-1, mask_pos] * vy_face[j, mask_pos] * dt / dx
            
            mask_neg = vy_face[j, :] <= 0
            flux_y[j, mask_neg] = phi[j, mask_neg] * vy_face[j, mask_neg] * dt / dx
            
        return flux_x, flux_y
        
    def compute_heat_flux(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute heat flux through diffusion and advection.
        
        Args:
            dt: Time step
            
        Returns:
            (flux_x, flux_y): Heat fluxes at cell faces
        """
        nx, ny = self.state.nx, self.state.ny
        dx = self.state.dx
        T = self.state.temperature
        k = self.state.thermal_conductivity
        
        # Diffusive flux: -k * dT/dx
        flux_x = np.zeros((ny, nx+1), dtype=np.float32)
        flux_y = np.zeros((ny+1, nx), dtype=np.float32)
        
        # X-direction diffusion
        for i in range(1, nx):
            # Harmonic mean of conductivity at face
            k_face = 2.0 * k[:, i-1] * k[:, i] / (k[:, i-1] + k[:, i] + 1e-10)
            flux_x[:, i] = -k_face * (T[:, i] - T[:, i-1]) / dx * dt
            
        # Y-direction diffusion
        for j in range(1, ny):
            # Harmonic mean of conductivity at face
            k_face = 2.0 * k[j-1, :] * k[j, :] / (k[j-1, :] + k[j, :] + 1e-10)
            flux_y[j, :] = -k_face * (T[j, :] - T[j-1, :]) / dx * dt
            
        # Add advective heat flux (could be added here)
        # For now, focusing on diffusion
        
        return flux_x, flux_y
        
    def apply_flux_divergence(self, flux_x: np.ndarray, flux_y: np.ndarray, 
                            quantity: np.ndarray) -> np.ndarray:
        """
        Apply flux divergence to update a quantity.
        
        Args:
            flux_x: X-direction fluxes at faces
            flux_y: Y-direction fluxes at faces
            quantity: The quantity to update
            
        Returns:
            Updated quantity
        """
        # Compute flux divergence
        div_flux = np.zeros_like(quantity)
        
        # X-direction: flux_out - flux_in
        div_flux += flux_x[:, 1:] - flux_x[:, :-1]
        
        # Y-direction: flux_out - flux_in
        div_flux += flux_y[1:, :] - flux_y[:-1, :]
        
        # Update quantity
        return quantity - div_flux
        
    def advect_materials(self, dt: float):
        """
        Transport all materials using flux form.
        
        Args:
            dt: Time step
        """
        # Skip space material (index 0)
        for mat in range(1, self.state.n_materials):
            # Compute fluxes for this material
            flux_x, flux_y = self.compute_material_flux(mat, dt)
            
            # Store in state for visualization/debugging
            if mat == MaterialType.WATER:  # Track water flux for visualization
                self.state.mass_flux_x = flux_x
                self.state.mass_flux_y = flux_y
            
            # Apply flux divergence
            self.state.vol_frac[mat] = self.apply_flux_divergence(
                flux_x, flux_y, self.state.vol_frac[mat]
            )
            
        # Ensure constraints
        self.state.normalize_volume_fractions()
        
    def diffuse_heat(self, dt: float):
        """
        Apply thermal diffusion.
        
        Args:
            dt: Time step
        """
        # Compute heat fluxes
        flux_x, flux_y = self.compute_heat_flux(dt)
        
        # Store for visualization
        self.state.heat_flux_x = flux_x
        self.state.heat_flux_y = flux_y
        
        # Apply to temperature field
        # Need to account for heat capacity
        heat_div = np.zeros_like(self.state.temperature)
        heat_div += flux_x[:, 1:] - flux_x[:, :-1]
        heat_div += flux_y[1:, :] - flux_y[:-1, :]
        
        # dT/dt = -div(heat_flux) / (rho * cp)
        denom = self.state.density * self.state.specific_heat + 1e-10
        self.state.temperature -= heat_div / denom