"""
Pressure solver for flux-based geological simulation.

This implements the pressure-projection method for incompressible flow:
1. Given intermediate velocity v* (after applying forces)
2. Find pressure φ such that ∇·(β∇φ) = ∇·v*/Δt where β = 1/ρ
3. Update velocity: v_new = v* - Δt β∇φ
4. Result is divergence-free: ∇·v_new = 0

Key features:
- Variable density handled via face-centered β = 1/ρ coefficients
- MAC staggered grid for stability
- Harmonic averaging at material interfaces
"""

import numpy as np
from typing import Tuple, Optional
from state import FluxState
from materials import MaterialType

# Import MAC multigrid solver
from multigrid import solve_mac_poisson_vectorized as solve_mac_poisson, BoundaryCondition


class PressureSolver:
    """Multigrid pressure solver for projection method."""
    
    def __init__(self, state: FluxState):
        """
        Initialize pressure solver.
        
        Args:
            state: FluxState instance
        """
        self.state = state
        self.phi_prev = None  # Store previous solution for warm start
        
    def project_velocity(self, dt: float, gx: np.ndarray = None, gy: np.ndarray = None, 
                        bc_type: str = "neumann") -> np.ndarray:
        """Project velocity field to make it divergence-free.
        
        Solves: ∇·(β∇φ) = ∇·v*/Δt + ∇·g where β = 1/ρ
        Then updates: v = v* - Δt β∇φ + Δt g
        
        Args:
            dt: Time step
            gx, gy: Gravity field (optional)
            bc_type: Boundary condition type ('neumann' or 'dirichlet')
            
        Returns:
            phi: Pressure correction field
        """
        st = self.state
        
        # Ensure face coefficients are up to date
        st.update_face_coefficients()
        
        # Apply gravity to face velocities first
        if gx is not None and gy is not None:
            # Add gravity contribution to face velocities
            # For x-faces: average gx from adjacent cells
            st.velocity_x_face[:, 1:-1] += dt * 0.5 * (gx[:, :-1] + gx[:, 1:])
            st.velocity_x_face[:, 0] += dt * gx[:, 0]
            st.velocity_x_face[:, -1] += dt * gx[:, -1]
            
            # For y-faces: average gy from adjacent cells
            st.velocity_y_face[1:-1, :] += dt * 0.5 * (gy[:-1, :] + gy[1:, :])
            st.velocity_y_face[0, :] += dt * gy[0, :]
            st.velocity_y_face[-1, :] += dt * gy[-1, :]
        
        # Compute divergence of face velocities (including gravity)
        div = self._compute_divergence()
        
        # RHS for Poisson equation
        rhs = div / dt
        
        # Zero RHS in space regions - they should maintain zero pressure
        space_mask = st.density < 0.1
        rhs[space_mask] = 0.0
        
        # Solve variable-coefficient Poisson equation
        # Use very relaxed settings for real-time performance
        grid_size = st.nx * st.ny
        if grid_size > 8192:  # 128x64 or larger
            max_iter = 20   # Very strict iteration limit
            tol = 5e-2      # Accept approximate solutions
        else:
            max_iter = 50
            tol = 1e-3
            
        # Solve using MAC multigrid
        bc = BoundaryCondition.NEUMANN if bc_type == "neumann" else BoundaryCondition.DIRICHLET
        phi = solve_mac_poisson(
            rhs,
            st.beta_x,
            st.beta_y,
            st.dx,
            bc_type=bc,
            tol=tol,
            max_cycles=max_iter,
            initial_guess=self.phi_prev,
        )
        
        # Store for next time
        if self.phi_prev is None:
            self.phi_prev = np.copy(phi)
        else:
            self.phi_prev[:] = phi
        
        # Update face velocities using pressure gradient
        self._update_face_velocities(phi, dt)
        
        # Update cell-centered velocities from face velocities
        st.update_cell_velocities_from_face()
        
        # Accumulate pressure correction
        st.pressure += phi
        
        # Force zero pressure in space regions
        space_mask = st.density < 0.1
        st.pressure[space_mask] = 0.0
        
        # Remove mean pressure to prevent drift (for Neumann BC)
        # Only consider non-space regions for the mean
        if bc_type == "neumann":
            non_space_pressure = st.pressure[~space_mask]
            if len(non_space_pressure) > 0:
                st.pressure[~space_mask] -= np.mean(non_space_pressure)
        
        return phi
        
    def _compute_divergence(self) -> np.ndarray:
        """Compute divergence of face-centered velocities.
        
        Returns:
            div: Cell-centered divergence field
        """
        st = self.state
        dx = st.dx
        
        # Divergence at cell centers
        div = np.zeros_like(st.density)
        
        # ∇·v = (vx_e - vx_w)/dx + (vy_n - vy_s)/dx
        div[:, :] = (
            (st.velocity_x_face[:, 1:] - st.velocity_x_face[:, :-1]) / dx +
            (st.velocity_y_face[1:, :] - st.velocity_y_face[:-1, :]) / dx
        )
        
        # Zero divergence in space regions (they don't participate in incompressibility)
        space_mask = st.density < 0.1  # Space has essentially zero density
        div[space_mask] = 0.0
        
        return div
        
    def _update_face_velocities(self, phi: np.ndarray, dt: float):
        """Update face velocities using pressure gradient.
        
        v_new = v* - dt * β * ∇φ
        
        Args:
            phi: Pressure correction
            dt: Time step
        """
        st = self.state
        dx = st.dx
        
        # X-face velocities
        grad_phi_x = (phi[:, 1:] - phi[:, :-1]) / dx
        st.velocity_x_face[:, 1:-1] -= dt * st.beta_x[:, 1:-1] * grad_phi_x
        
        # Y-face velocities  
        grad_phi_y = (phi[1:, :] - phi[:-1, :]) / dx
        st.velocity_y_face[1:-1, :] -= dt * st.beta_y[1:-1, :] * grad_phi_y
        
        # Boundary faces handled by BC


