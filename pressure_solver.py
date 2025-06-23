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

# Import multigrid solver
from multigrid import solve_variable_poisson_2d, BoundaryCondition


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
        # Use adaptive tolerances based on grid size
        grid_size = st.nx * st.ny
        if grid_size > 8192:  # 128x64 or larger
            max_iter = 10   # Very limited iterations for multigrid
            tol = 1e-3      # Relax tolerance for large grids
        else:
            max_iter = 20
            tol = 1e-5
            
        phi = solve_pressure_variable_coeff(
            rhs,
            st.beta_x,
            st.beta_y,
            st.dx,
            tol=tol,
            max_iter=max_iter,
            bc_type=bc_type,
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


# -----------------------------------------------------------------------------
# Variable-coefficient multigrid solver
# -----------------------------------------------------------------------------

def _apply_bc(phi: np.ndarray, bc_type: str = "neumann"):
    """Apply boundary conditions."""
    if bc_type == "neumann":
        # Homogeneous Neumann: ∂φ/∂n = 0
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]
    else:
        # Homogeneous Dirichlet: φ = 0
        phi[0, :] = 0.0
        phi[-1, :] = 0.0
        phi[:, 0] = 0.0
        phi[:, -1] = 0.0


def solve_pressure_variable_coeff(
    rhs: np.ndarray, 
    beta_x: np.ndarray, 
    beta_y: np.ndarray, 
    dx: float,
    *, 
    tol: float = 1e-6, 
    max_iter: int = 10000, 
    bc_type: str = "neumann",
    initial_guess: np.ndarray = None
) -> np.ndarray:
    """Solve variable-coefficient Poisson equation: ∇·(β∇φ) = rhs
    
    Uses multigrid for large grids, Gauss-Seidel for small grids.
    
    Args:
        rhs: Right-hand side (cell-centered)
        beta_x: Face-centered coefficients in x (shape ny, nx+1)
        beta_y: Face-centered coefficients in y (shape ny+1, nx)
        dx: Grid spacing
        tol: Convergence tolerance
        max_iter: Maximum iterations
        bc_type: Boundary condition type
        
    Returns:
        phi: Solution to Poisson equation
    """
    ny, nx = rhs.shape
    
    # Use multigrid for medium/large grids
    if nx * ny > 256:  # 16x16 or larger
        bc = BoundaryCondition.NEUMANN if bc_type == "neumann" else BoundaryCondition.DIRICHLET
        return solve_variable_poisson_2d(
            rhs, beta_x, beta_y, dx,
            bc_type=bc,
            tol=tol,
            initial_guess=initial_guess
        )
    
    # Fall back to Gauss-Seidel for very small grids
    if initial_guess is not None and initial_guess.shape == rhs.shape:
        phi = np.copy(initial_guess)
    else:
        phi = np.zeros_like(rhs)
    
    dx2 = dx * dx
    omega = 1.7  # SOR relaxation parameter
    
    for it in range(max_iter):
        max_change = 0.0
        
        # Red-black Gauss-Seidel
        for color in (0, 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    if (i + j) % 2 != color:
                        continue
                    
                    # Face coefficients
                    bx_e = beta_x[j, i + 1]
                    bx_w = beta_x[j, i]
                    by_n = beta_y[j + 1, i]
                    by_s = beta_y[j, i]
                    
                    # Denominator
                    denom = bx_e + bx_w + by_n + by_s
                    if denom < 1e-12:
                        continue
                    
                    # Gauss-Seidel update
                    phi_new = (
                        bx_e * phi[j, i + 1] +
                        bx_w * phi[j, i - 1] +
                        by_n * phi[j + 1, i] +
                        by_s * phi[j - 1, i] -
                        dx2 * rhs[j, i]
                    ) / denom
                    
                    # SOR update
                    phi_new = phi[j, i] + omega * (phi_new - phi[j, i])
                    
                    # Track convergence
                    change = abs(phi_new - phi[j, i])
                    if change > max_change:
                        max_change = change
                    
                    phi[j, i] = phi_new
        
        # Apply boundary conditions
        _apply_bc(phi, bc_type)
        
        # Check convergence
        if max_change < tol:
            break
    
    # For Neumann BC, remove mean to fix null space
    if bc_type == "neumann":
        phi -= np.mean(phi)
    
    return phi