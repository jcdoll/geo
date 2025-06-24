"""
Fixed pressure solver for flux-based geological simulation.

This version properly handles boundary face velocities to avoid banding artifacts.
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
        self._update_face_velocities_fixed(phi, dt)
        
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
        
    def _update_face_velocities_fixed(self, phi: np.ndarray, dt: float):
        """Update face velocities using pressure gradient.
        
        FIXED VERSION: Properly handles boundary faces for Neumann BC.
        
        v_new = v* - dt * β * ∇φ
        
        Args:
            phi: Pressure correction
            dt: Time step
        """
        st = self.state
        dx = st.dx
        
        # X-face velocities
        # Interior faces: use centered difference
        grad_phi_x = (phi[:, 1:] - phi[:, :-1]) / dx
        st.velocity_x_face[:, 1:-1] -= dt * st.beta_x[:, 1:-1] * grad_phi_x
        
        # Boundary faces for Neumann BC
        # For solid walls, enforce no-penetration condition
        if True:  # Assuming solid wall boundaries
            st.velocity_x_face[:, 0] = 0.0    # Left wall
            st.velocity_x_face[:, -1] = 0.0   # Right wall
        else:
            # Alternative: For open boundaries, could extrapolate gradient
            # But this is not typically used in geological simulations
            pass
        
        # Y-face velocities  
        # Interior faces: use centered difference
        grad_phi_y = (phi[1:, :] - phi[:-1, :]) / dx
        st.velocity_y_face[1:-1, :] -= dt * st.beta_y[1:-1, :] * grad_phi_y
        
        # Boundary faces for Neumann BC
        # For solid walls, enforce no-penetration condition
        if True:  # Assuming solid wall boundaries
            st.velocity_y_face[0, :] = 0.0    # Bottom wall
            st.velocity_y_face[-1, :] = 0.0   # Top wall
        else:
            # Alternative: For open boundaries, could extrapolate gradient
            pass