"""
Proper fix for the velocity banding issue in the pressure solver.

The key insight is that for Neumann BC on pressure (∂φ/∂n = 0), we still need
to update face velocities at boundaries, but the pressure gradient should be
computed using one-sided differences that respect the BC.
"""

import numpy as np
from typing import Tuple, Optional
from state import FluxState
from materials import MaterialType

# Import MAC multigrid solver
from multigrid import solve_mac_poisson_vectorized as solve_mac_poisson, BoundaryCondition


class PressureSolver:
    """Fixed multigrid pressure solver for projection method."""
    
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
        
        Solves: ∇·(β∇φ) = ∇·v*/Δt where β = 1/ρ
        Then updates: v = v* - Δt β∇φ
        
        Note: Gravity should already be applied in the predictor step.
        
        Args:
            dt: Time step
            gx, gy: Not used (kept for compatibility)
            bc_type: Boundary condition type ('neumann' or 'dirichlet')
            
        Returns:
            phi: Pressure correction field
        """
        st = self.state
        
        # Ensure face coefficients are up to date
        st.update_face_coefficients()
        
        # Compute divergence of face velocities
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
            max_iter = 10   # Very strict iteration limit
            tol = 1e-1      # Accept approximate solutions
        else:
            max_iter = 15   # Reduced from 50
            tol = 5e-2      # Relaxed from 1e-3
            
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
        self._update_face_velocities_proper(phi, dt, bc_type)
        
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
        
    def _update_face_velocities_proper(self, phi: np.ndarray, dt: float, bc_type: str):
        """Update face velocities using pressure gradient.
        
        PROPER VERSION: Correctly handles ALL faces including boundaries.
        
        For Neumann BC (∂φ/∂n = 0), the key insight is:
        - The BC applies to the NORMAL derivative at boundaries
        - Face velocities still need pressure gradient corrections
        - We use ghost cell values consistent with the BC
        
        v_new = v* - dt * β * ∇φ
        
        Args:
            phi: Pressure correction
            dt: Time step
            bc_type: Boundary condition type
        """
        st = self.state
        dx = st.dx
        
        # Compute pressure gradients at ALL faces
        # For Neumann BC, we need to handle boundaries carefully
        
        if bc_type == "neumann":
            # X-face velocities (including boundaries)
            # Interior faces: standard centered difference
            grad_phi_x_int = (phi[:, 1:] - phi[:, :-1]) / dx
            st.velocity_x_face[:, 1:-1] -= dt * st.beta_x[:, 1:-1] * grad_phi_x_int
            
            # Boundary faces for Neumann BC
            # At left boundary (x=0): ghost cell has φ[-1,j] = φ[0,j] (from ∂φ/∂x = 0)
            # So gradient at face is: (φ[0,j] - φ[-1,j])/dx = 0
            grad_phi_x_left = 0.0
            st.velocity_x_face[:, 0] -= dt * st.beta_x[:, 0] * grad_phi_x_left
            
            # At right boundary (x=nx): ghost cell has φ[nx,j] = φ[nx-1,j]
            # So gradient at face is: (φ[nx,j] - φ[nx-1,j])/dx = 0
            grad_phi_x_right = 0.0
            st.velocity_x_face[:, -1] -= dt * st.beta_x[:, -1] * grad_phi_x_right
            
            # Y-face velocities (including boundaries)
            # Interior faces: standard centered difference
            grad_phi_y_int = (phi[1:, :] - phi[:-1, :]) / dx
            st.velocity_y_face[1:-1, :] -= dt * st.beta_y[1:-1, :] * grad_phi_y_int
            
            # Boundary faces for Neumann BC
            # At bottom boundary (y=0): ghost cell has φ[i,-1] = φ[i,0]
            # So gradient at face is: (φ[i,0] - φ[i,-1])/dy = 0
            grad_phi_y_bottom = 0.0
            st.velocity_y_face[0, :] -= dt * st.beta_y[0, :] * grad_phi_y_bottom
            
            # At top boundary (y=ny): ghost cell has φ[i,ny] = φ[i,ny-1]
            # So gradient at face is: (φ[i,ny] - φ[i,ny-1])/dy = 0
            grad_phi_y_top = 0.0
            st.velocity_y_face[-1, :] -= dt * st.beta_y[-1, :] * grad_phi_y_top
            
        else:  # Dirichlet BC
            # For Dirichlet BC (φ = 0 at boundaries), we can use one-sided differences
            
            # X-faces
            # Interior faces
            grad_phi_x_int = (phi[:, 1:] - phi[:, :-1]) / dx
            st.velocity_x_face[:, 1:-1] -= dt * st.beta_x[:, 1:-1] * grad_phi_x_int
            
            # Left boundary: φ[-1,j] = 0, so gradient = (φ[0,j] - 0) / dx
            grad_phi_x_left = phi[:, 0] / dx
            st.velocity_x_face[:, 0] -= dt * st.beta_x[:, 0] * grad_phi_x_left
            
            # Right boundary: φ[nx,j] = 0, so gradient = (0 - φ[nx-1,j]) / dx
            grad_phi_x_right = -phi[:, -1] / dx
            st.velocity_x_face[:, -1] -= dt * st.beta_x[:, -1] * grad_phi_x_right
            
            # Y-faces
            # Interior faces
            grad_phi_y_int = (phi[1:, :] - phi[:-1, :]) / dx
            st.velocity_y_face[1:-1, :] -= dt * st.beta_y[1:-1, :] * grad_phi_y_int
            
            # Bottom boundary: φ[i,-1] = 0, so gradient = (φ[i,0] - 0) / dy
            grad_phi_y_bottom = phi[0, :] / dx
            st.velocity_y_face[0, :] -= dt * st.beta_y[0, :] * grad_phi_y_bottom
            
            # Top boundary: φ[i,ny] = 0, so gradient = (0 - φ[i,ny-1]) / dy
            grad_phi_y_top = -phi[-1, :] / dx
            st.velocity_y_face[-1, :] -= dt * st.beta_y[-1, :] * grad_phi_y_top