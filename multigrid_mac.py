"""
Optimized multigrid solver for MAC (Marker-and-Cell) grids.

This version is specifically designed for face-centered coefficients
as used in the flux-based simulation with proper vectorization.
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum


class BoundaryCondition(Enum):
    """Boundary condition types."""
    NEUMANN = "neumann"     # ∂φ/∂n = 0
    DIRICHLET = "dirichlet" # φ = 0


def apply_bc(phi: np.ndarray, bc_type: BoundaryCondition):
    """Apply boundary conditions."""
    if bc_type == BoundaryCondition.NEUMANN:
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]
    else:  # DIRICHLET
        phi[0, :] = 0.0
        phi[-1, :] = 0.0
        phi[:, 0] = 0.0
        phi[:, -1] = 0.0


def smooth_redblack_mac(
    phi: np.ndarray,
    rhs: np.ndarray, 
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    dx: float,
    n_iter: int = 3,
    bc_type: BoundaryCondition = BoundaryCondition.NEUMANN
):
    """
    Vectorized red-black Gauss-Seidel for MAC grid.
    
    Uses face-centered coefficients:
    - beta_x: shape (ny, nx+1) - x-face coefficients  
    - beta_y: shape (ny+1, nx) - y-face coefficients
    """
    ny, nx = phi.shape
    dx2 = dx * dx
    
    # Create red-black masks for interior cells
    j_idx, i_idx = np.ogrid[1:ny-1, 1:nx-1]
    red_mask = ((i_idx + j_idx) % 2 == 0)
    black_mask = ~red_mask
    
    for _ in range(n_iter):
        # Prepare sliced views for vectorized operations
        phi_interior = phi[1:-1, 1:-1]
        rhs_interior = rhs[1:-1, 1:-1]
        
        # Neighbor values
        phi_e = phi[1:-1, 2:]    # East neighbors
        phi_w = phi[1:-1, :-2]   # West neighbors
        phi_n = phi[2:, 1:-1]    # North neighbors
        phi_s = phi[:-2, 1:-1]   # South neighbors
        
        # Face coefficients for interior cells
        # Note: beta_x has nx+1 columns, beta_y has ny+1 rows
        bx_e = beta_x[1:-1, 2:nx]    # East face coeffs
        bx_w = beta_x[1:-1, 1:nx-1]  # West face coeffs
        by_n = beta_y[2:ny, 1:-1]    # North face coeffs
        by_s = beta_y[1:ny-1, 1:-1]  # South face coeffs
        
        # Compute denominator
        denom = bx_e + bx_w + by_n + by_s
        
        # Compute new values where denominator is non-zero
        phi_new = np.zeros_like(phi_interior)
        valid = denom > 1e-12
        phi_new[valid] = (
            bx_e[valid] * phi_e[valid] + 
            bx_w[valid] * phi_w[valid] +
            by_n[valid] * phi_n[valid] + 
            by_s[valid] * phi_s[valid] - 
            dx2 * rhs_interior[valid]
        ) / denom[valid]
        
        # Red sweep
        phi_interior[red_mask] = phi_new[red_mask]
        
        # Recompute for black (values have changed)
        phi_new[valid] = (
            bx_e[valid] * phi_e[valid] + 
            bx_w[valid] * phi_w[valid] +
            by_n[valid] * phi_n[valid] + 
            by_s[valid] * phi_s[valid] - 
            dx2 * rhs_interior[valid]
        ) / denom[valid]
        
        # Black sweep
        phi_interior[black_mask] = phi_new[black_mask]
        
        # Apply boundary conditions
        apply_bc(phi, bc_type)


def restrict_full_weighting(fine: np.ndarray) -> np.ndarray:
    """
    Full-weighting restriction using vectorized operations.
    
    Handles odd dimensions properly.
    """
    ny, nx = fine.shape
    nyc = (ny + 1) // 2
    nxc = (nx + 1) // 2
    
    coarse = np.zeros((nyc, nxc), dtype=fine.dtype)
    
    # Main 2x2 blocks - fully vectorized
    ny_blocks = ny // 2
    nx_blocks = nx // 2
    
    if ny_blocks > 0 and nx_blocks > 0:
        # Reshape for efficient averaging
        blocks = fine[:ny_blocks*2, :nx_blocks*2].reshape(ny_blocks, 2, nx_blocks, 2)
        coarse[:ny_blocks, :nx_blocks] = blocks.mean(axis=(1, 3))
    
    # Handle odd edges if needed
    if ny % 2 == 1 and nx_blocks > 0:
        # Last row - average pairs in x
        coarse[-1, :nx_blocks] = 0.5 * (fine[-1, :nx_blocks*2:2] + fine[-1, 1:nx_blocks*2:2])
    
    if nx % 2 == 1 and ny_blocks > 0:
        # Last column - average pairs in y
        coarse[:ny_blocks, -1] = 0.5 * (fine[:ny_blocks*2:2, -1] + fine[1:ny_blocks*2:2, -1])
    
    if ny % 2 == 1 and nx % 2 == 1:
        # Corner cell
        coarse[-1, -1] = fine[-1, -1]
    
    return coarse


def prolong_bilinear(coarse: np.ndarray, fine_shape: Tuple[int, int]) -> np.ndarray:
    """
    Bilinear prolongation using vectorized operations.
    """
    nyf, nxf = fine_shape
    nyc, nxc = coarse.shape
    fine = np.zeros(fine_shape, dtype=coarse.dtype)
    
    # Direct injection at coarse points
    # Handle dimensions carefully
    ny_inject = min(nyc, (nyf + 1) // 2)
    nx_inject = min(nxc, (nxf + 1) // 2)
    fine[:ny_inject*2:2, :nx_inject*2:2] = coarse[:ny_inject, :nx_inject]
    
    # Horizontal interpolation
    if nxf > 1:
        # Fill odd columns by averaging neighbors
        for i in range(1, nxf, 2):
            if i < nxf - 1:
                fine[::2, i] = 0.5 * (fine[::2, i-1] + fine[::2, i+1])
            else:
                # Last column if odd
                fine[::2, i] = fine[::2, i-1]
    
    # Vertical interpolation  
    if nyf > 1:
        # Fill odd rows by averaging neighbors
        for j in range(1, nyf, 2):
            if j < nyf - 1:
                fine[j, :] = 0.5 * (fine[j-1, :] + fine[j+1, :])
            else:
                # Last row if odd
                fine[j, :] = fine[j-1, :]
    
    return fine


def restrict_face_coeffs_mac(
    beta_x: np.ndarray, 
    beta_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restrict face-centered coefficients for MAC grid.
    
    Uses harmonic averaging for coefficients (1/ρ).
    """
    # X-faces: (ny, nx+1) -> (ny//2, nx//2+1)
    ny, nx_faces = beta_x.shape
    nyc = ny // 2
    nxc = (nx_faces - 1) // 2
    
    beta_x_coarse = np.zeros((nyc, nxc + 1))
    
    # Vectorized restriction for x-faces
    # Interior faces: average 2x2 stencil with harmonic mean
    for ic in range(nxc + 1):
        i_fine = min(2 * ic, nx_faces - 1)
        
        if i_fine < nx_faces - 1:
            # Average vertically (arithmetic mean of 1/ρ is fine)
            beta_x_coarse[:, ic] = 0.5 * (
                beta_x[::2, i_fine][:nyc] + 
                beta_x[1::2, i_fine][:nyc]
            )
        else:
            # Boundary
            beta_x_coarse[:, ic] = beta_x[::2, -1][:nyc]
    
    # Y-faces: (ny+1, nx) -> (ny//2+1, nx//2)
    ny_faces, nx = beta_y.shape
    nxc = nx // 2
    
    beta_y_coarse = np.zeros((nyc + 1, nxc))
    
    # Vectorized restriction for y-faces  
    for jc in range(nyc + 1):
        j_fine = min(2 * jc, ny_faces - 1)
        
        if j_fine < ny_faces - 1:
            # Average horizontally
            beta_y_coarse[jc, :] = 0.5 * (
                beta_y[j_fine, ::2][:nxc] + 
                beta_y[j_fine, 1::2][:nxc]
            )
        else:
            # Boundary
            beta_y_coarse[jc, :] = beta_y[-1, ::2][:nxc]
    
    return beta_x_coarse, beta_y_coarse


def compute_residual_mac(
    phi: np.ndarray,
    rhs: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    dx: float
) -> np.ndarray:
    """
    Compute residual for MAC grid - fully vectorized.
    
    r = rhs - A*phi where A*phi = -∇·(β∇φ)
    """
    ny, nx = phi.shape
    dx2 = dx * dx
    residual = np.zeros_like(phi)
    
    # Vectorized Laplacian for interior points
    # Extract interior slices
    phi_e = phi[1:-1, 2:]
    phi_w = phi[1:-1, :-2] 
    phi_n = phi[2:, 1:-1]
    phi_s = phi[:-2, 1:-1]
    phi_c = phi[1:-1, 1:-1]
    
    # Face coefficients
    bx_e = beta_x[1:-1, 2:nx]
    bx_w = beta_x[1:-1, 1:nx-1]
    by_n = beta_y[2:ny, 1:-1]
    by_s = beta_y[1:ny-1, 1:-1]
    
    # Compute Laplacian
    laplacian = (
        bx_e * phi_e + bx_w * phi_w + 
        by_n * phi_n + by_s * phi_s -
        (bx_e + bx_w + by_n + by_s) * phi_c
    ) / dx2
    
    # Residual
    residual[1:-1, 1:-1] = rhs[1:-1, 1:-1] - laplacian
    
    return residual


def v_cycle_mac(
    phi: np.ndarray,
    rhs: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    dx: float,
    level: int,
    max_level: int,
    bc_type: BoundaryCondition = BoundaryCondition.NEUMANN,
    pre_smooth: int = 3,
    post_smooth: int = 3
) -> np.ndarray:
    """V-cycle for MAC grid."""
    
    # Pre-smooth
    smooth_redblack_mac(phi, rhs, beta_x, beta_y, dx, pre_smooth, bc_type)
    
    ny, nx = phi.shape
    if level < max_level and min(ny, nx) > 3:
        # Compute residual
        residual = compute_residual_mac(phi, rhs, beta_x, beta_y, dx)
        
        # Restrict
        residual_coarse = restrict_full_weighting(residual)
        beta_x_coarse, beta_y_coarse = restrict_face_coeffs_mac(beta_x, beta_y)
        
        # Solve on coarse grid
        phi_coarse = np.zeros_like(residual_coarse)
        phi_coarse = v_cycle_mac(
            phi_coarse, residual_coarse,
            beta_x_coarse, beta_y_coarse,
            dx * 2, level + 1, max_level,
            bc_type, pre_smooth, post_smooth
        )
        
        # Prolong and correct
        correction = prolong_bilinear(phi_coarse, phi.shape)
        phi += correction
    
    # Post-smooth
    smooth_redblack_mac(phi, rhs, beta_x, beta_y, dx, post_smooth, bc_type)
    
    return phi


def solve_mac_poisson(
    rhs: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    dx: float,
    *,
    tol: float = 1e-6,
    max_cycles: int = 50,
    bc_type: BoundaryCondition = BoundaryCondition.NEUMANN,
    initial_guess: Optional[np.ndarray] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Solve variable-coefficient Poisson equation on MAC grid.
    
    Solves: ∇·(β∇φ) = rhs
    
    where β is specified at cell faces:
    - beta_x: shape (ny, nx+1) - x-face coefficients
    - beta_y: shape (ny+1, nx) - y-face coefficients
    
    Uses padding to even dimensions for clean multigrid coarsening.
    """
    ny, nx = rhs.shape
    
    # Pad to even dimensions
    ny_pad = ny + (ny & 1)
    nx_pad = nx + (nx & 1)
    
    # Pad RHS
    rhs_pad = np.zeros((ny_pad, nx_pad), dtype=rhs.dtype)
    rhs_pad[:ny, :nx] = rhs
    
    # Pad face coefficients - extend last values
    beta_x_pad = np.ones((ny_pad, nx_pad + 1), dtype=beta_x.dtype)
    beta_x_pad[:ny, :beta_x.shape[1]] = beta_x
    # Extend in y if needed
    if ny_pad > ny:
        beta_x_pad[ny:, :beta_x.shape[1]] = beta_x[-1:, :]
    # Extend in x if needed  
    if nx_pad > nx:
        beta_x_pad[:, nx+1:] = beta_x_pad[:, nx:nx+1]
    
    beta_y_pad = np.ones((ny_pad + 1, nx_pad), dtype=beta_y.dtype)
    beta_y_pad[:beta_y.shape[0], :nx] = beta_y
    # Extend in x if needed
    if nx_pad > nx:
        beta_y_pad[:beta_y.shape[0], nx:] = beta_y[:, -1:]
    # Extend in y if needed
    if ny_pad > ny:
        beta_y_pad[ny+1:, :] = beta_y_pad[ny:ny+1, :]
    
    # Initial guess
    if initial_guess is not None and initial_guess.shape == (ny, nx):
        phi_pad = np.zeros((ny_pad, nx_pad), dtype=rhs.dtype)
        phi_pad[:ny, :nx] = initial_guess
    else:
        phi_pad = np.zeros((ny_pad, nx_pad), dtype=rhs.dtype)
    
    # Compute max levels
    max_level = max(1, int(np.log2(min(ny_pad, nx_pad))) - 2)
    
    # Solve
    rhs_norm = np.linalg.norm(rhs)
    if rhs_norm < 1e-14:
        return phi_pad[:ny, :nx]
    
    for cycle in range(max_cycles):
        # V-cycle
        phi_pad = v_cycle_mac(
            phi_pad, rhs_pad, beta_x_pad, beta_y_pad,
            dx, 0, max_level, bc_type
        )
        
        # Check convergence on original grid only
        residual = compute_residual_mac(phi_pad, rhs_pad, beta_x_pad, beta_y_pad, dx)
        res_norm = np.linalg.norm(residual[:ny, :nx])
        
        if verbose:
            print(f"Cycle {cycle}: residual = {res_norm:.3e} (rel = {res_norm/rhs_norm:.3e})")
        
        if res_norm < tol * rhs_norm:
            if verbose:
                print(f"Converged in {cycle + 1} cycles")
            break
    
    # Extract solution
    phi = phi_pad[:ny, :nx]
    
    # Remove mean for Neumann BC
    if bc_type == BoundaryCondition.NEUMANN:
        phi -= np.mean(phi)
    
    return phi