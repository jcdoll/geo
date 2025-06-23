"""
Fully vectorized multigrid solver for MAC (Marker-and-Cell) grids.

This version eliminates ALL loops in performance-critical sections.
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


def smooth_redblack_mac_vectorized(
    phi: np.ndarray,
    rhs: np.ndarray, 
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    dx: float,
    n_iter: int = 3,
    bc_type: BoundaryCondition = BoundaryCondition.NEUMANN
):
    """
    Fully vectorized red-black Gauss-Seidel for MAC grid.
    
    Uses face-centered coefficients:
    - beta_x: shape (ny, nx+1) - x-face coefficients  
    - beta_y: shape (ny+1, nx) - y-face coefficients
    """
    ny, nx = phi.shape
    dx2 = dx * dx
    
    # Create red-black masks using broadcasting
    j = np.arange(1, ny-1)[:, np.newaxis]
    i = np.arange(1, nx-1)[np.newaxis, :]
    red_mask = ((i + j) % 2 == 0)
    black_mask = ~red_mask
    
    for _ in range(n_iter):
        # Get interior slices
        phi_c = phi[1:-1, 1:-1]
        rhs_c = rhs[1:-1, 1:-1]
        
        # Neighbor values
        phi_e = phi[1:-1, 2:]
        phi_w = phi[1:-1, :-2]
        phi_n = phi[2:, 1:-1]
        phi_s = phi[:-2, 1:-1]
        
        # Face coefficients - proper slicing for MAC grid
        # Get interior dimensions
        ny_int = phi_c.shape[0]  # Should be ny-2
        nx_int = phi_c.shape[1]  # Should be nx-2
        
        # Extract face coefficients for interior cells
        # beta_x has shape (ny, nx+1), extract for interior cells
        bx_e = beta_x[1:1+ny_int, 2:2+nx_int]      # East faces
        bx_w = beta_x[1:1+ny_int, 1:1+nx_int]      # West faces
        
        # beta_y has shape (ny+1, nx), extract for interior cells
        by_n = beta_y[2:2+ny_int, 1:1+nx_int]      # North faces
        by_s = beta_y[1:1+ny_int, 1:1+nx_int]      # South faces
        
        # Compute update for all interior points
        denom = bx_e + bx_w + by_n + by_s
        denom = np.where(denom > 1e-12, denom, 1.0)  # Avoid division by zero
        
        phi_new = (
            bx_e * phi_e + bx_w * phi_w +
            by_n * phi_n + by_s * phi_s - dx2 * rhs_c
        ) / denom
        
        # Red sweep - update red points only
        phi_c[red_mask] = phi_new[red_mask]
        
        # Black sweep - recompute with updated red values
        phi_new = (
            bx_e * phi_e + bx_w * phi_w +
            by_n * phi_n + by_s * phi_s - dx2 * rhs_c
        ) / denom
        
        # Update black points
        phi_c[black_mask] = phi_new[black_mask]
        
        # Apply boundary conditions
        apply_bc(phi, bc_type)


def restrict_full_weighting_vectorized(fine: np.ndarray) -> np.ndarray:
    """
    Fully vectorized full-weighting restriction.
    
    No loops - uses reshape and mean operations.
    """
    ny, nx = fine.shape
    nyc = (ny + 1) // 2
    nxc = (nx + 1) // 2
    
    coarse = np.zeros((nyc, nxc), dtype=fine.dtype)
    
    # Handle the main grid - fully vectorized using reshape
    ny_even = (ny // 2) * 2
    nx_even = (nx // 2) * 2
    
    if ny_even > 0 and nx_even > 0:
        # Reshape to expose 2x2 blocks, then average
        fine_blocks = fine[:ny_even, :nx_even].reshape(ny_even//2, 2, nx_even//2, 2)
        coarse[:ny_even//2, :nx_even//2] = fine_blocks.mean(axis=(1, 3))
    
    # Handle edges for odd dimensions - vectorized
    if ny % 2 == 1:
        # Last row - average pairs
        if nx_even > 0:
            coarse[-1, :nx_even//2] = 0.5 * (
                fine[-1, 0:nx_even:2] + fine[-1, 1:nx_even:2]
            )
        if nx % 2 == 1:
            coarse[-1, -1] = fine[-1, -1]
    
    if nx % 2 == 1:
        # Last column - average pairs
        if ny_even > 0:
            coarse[:ny_even//2, -1] = 0.5 * (
                fine[0:ny_even:2, -1] + fine[1:ny_even:2, -1]
            )
    
    return coarse


def prolong_bilinear_vectorized(coarse: np.ndarray, fine_shape: Tuple[int, int]) -> np.ndarray:
    """
    Fully vectorized bilinear prolongation.
    
    Uses advanced indexing instead of loops.
    """
    nyf, nxf = fine_shape
    nyc, nxc = coarse.shape
    fine = np.zeros(fine_shape, dtype=coarse.dtype)
    
    # Direct injection - vectorized
    ny_inject = min(nyc, (nyf + 1) // 2)
    nx_inject = min(nxc, (nxf + 1) // 2)
    fine[:ny_inject*2:2, :nx_inject*2:2] = coarse[:ny_inject, :nx_inject]
    
    # Horizontal interpolation - fully vectorized
    # Create index arrays for odd columns
    odd_cols = np.arange(1, nxf, 2)
    even_cols_left = np.maximum(odd_cols - 1, 0)
    even_cols_right = np.minimum(odd_cols + 1, nxf - 1)
    
    # Interpolate all odd columns at once
    if len(odd_cols) > 0:
        fine[::2, odd_cols] = 0.5 * (
            fine[::2, even_cols_left] + fine[::2, even_cols_right]
        )
    
    # Vertical interpolation - fully vectorized
    # Create index arrays for odd rows
    odd_rows = np.arange(1, nyf, 2)
    even_rows_up = np.maximum(odd_rows - 1, 0)
    even_rows_down = np.minimum(odd_rows + 1, nyf - 1)
    
    # Interpolate all odd rows at once
    if len(odd_rows) > 0:
        fine[odd_rows, :] = 0.5 * (
            fine[even_rows_up, :] + fine[even_rows_down, :]
        )
    
    return fine


def restrict_face_coeffs_mac_vectorized(
    beta_x: np.ndarray, 
    beta_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fully vectorized restriction of face-centered coefficients.
    
    This is now just an alias for the ultra-vectorized version.
    """
    return restrict_face_coeffs_mac_ultra_vectorized(beta_x, beta_y)


def compute_residual_mac_vectorized(
    phi: np.ndarray,
    rhs: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    dx: float
) -> np.ndarray:
    """
    Fully vectorized residual computation for MAC grid.
    
    r = rhs - A*phi where A*phi = -∇·(β∇φ)
    """
    dx2 = dx * dx
    residual = np.zeros_like(phi)
    
    # Compute Laplacian for interior points - fully vectorized
    ny, nx = phi.shape
    ny_int = ny - 2
    nx_int = nx - 2
    
    # Extract coefficients
    bx_e = beta_x[1:1+ny_int, 2:2+nx_int]
    bx_w = beta_x[1:1+ny_int, 1:1+nx_int]
    by_n = beta_y[2:2+ny_int, 1:1+nx_int]
    by_s = beta_y[1:1+ny_int, 1:1+nx_int]
    
    laplacian = (
        bx_e * phi[1:-1, 2:] +       # East flux
        bx_w * phi[1:-1, :-2] +      # West flux
        by_n * phi[2:, 1:-1] +       # North flux
        by_s * phi[:-2, 1:-1] -      # South flux
        (bx_e + bx_w + by_n + by_s) * phi[1:-1, 1:-1]  # Center
    ) / dx2
    
    # Residual
    residual[1:-1, 1:-1] = rhs[1:-1, 1:-1] - laplacian
    
    return residual


# Alternative: Even more vectorized face coefficient restriction
def restrict_face_coeffs_mac_ultra_vectorized(
    beta_x: np.ndarray, 
    beta_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ultra-vectorized version - completely loop-free.
    
    For MAC grid coefficient restriction, we need to be careful about
    which faces map to which coarse faces.
    """
    # X-faces: (ny, nx+1) -> (ny//2, nx//2+1)
    ny, nx_faces = beta_x.shape
    nx = nx_faces - 1
    nyc = ny // 2
    nxc = nx // 2
    
    # For x-faces, we average in y-direction and sample in x
    beta_x_coarse = np.zeros((nyc, nxc + 1))
    
    # Average adjacent rows (y-direction)
    if ny >= 2:
        beta_x_y_avg = 0.5 * (beta_x[::2, :][:nyc, :] + beta_x[1::2, :][:nyc, :])
        
        # For x-faces, coarse face ic corresponds to fine face 2*ic
        # Sample every other face, but handle the last face specially
        if nxc > 0:
            beta_x_coarse[:, :nxc] = beta_x_y_avg[:, ::2][:, :nxc]
            beta_x_coarse[:, nxc] = beta_x_y_avg[:, min(2*nxc, nx)]
    
    # Y-faces: (ny+1, nx) -> (ny//2+1, nx//2)
    ny_faces, nx = beta_y.shape
    ny = ny_faces - 1
    nyc = ny // 2
    nxc = nx // 2
    
    beta_y_coarse = np.zeros((nyc + 1, nxc))
    
    # Average adjacent columns (x-direction)
    if nx >= 2:
        beta_y_x_avg = 0.5 * (beta_y[:, ::2][:, :nxc] + beta_y[:, 1::2][:, :nxc])
        
        # For y-faces, coarse face jc corresponds to fine face 2*jc
        if nyc > 0:
            beta_y_coarse[:nyc, :] = beta_y_x_avg[::2, :][:nyc, :]
            beta_y_coarse[nyc, :] = beta_y_x_avg[min(2*nyc, ny), :]
    
    return beta_x_coarse, beta_y_coarse


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
    """V-cycle for MAC grid with fully vectorized operations."""
    
    # Pre-smooth
    smooth_redblack_mac_vectorized(phi, rhs, beta_x, beta_y, dx, pre_smooth, bc_type)
    
    ny, nx = phi.shape
    if level < max_level and min(ny, nx) > 3:
        # Compute residual
        residual = compute_residual_mac_vectorized(phi, rhs, beta_x, beta_y, dx)
        
        # Restrict
        residual_coarse = restrict_full_weighting_vectorized(residual)
        beta_x_coarse, beta_y_coarse = restrict_face_coeffs_mac_ultra_vectorized(beta_x, beta_y)
        
        # Solve on coarse grid
        phi_coarse = np.zeros_like(residual_coarse)
        phi_coarse = v_cycle_mac(
            phi_coarse, residual_coarse,
            beta_x_coarse, beta_y_coarse,
            dx * 2, level + 1, max_level,
            bc_type, pre_smooth, post_smooth
        )
        
        # Prolong and correct
        correction = prolong_bilinear_vectorized(phi_coarse, phi.shape)
        phi += correction
    
    # Post-smooth
    smooth_redblack_mac_vectorized(phi, rhs, beta_x, beta_y, dx, post_smooth, bc_type)
    
    return phi


def solve_mac_poisson_vectorized(
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
    
    Fully vectorized implementation.
    """
    ny, nx = rhs.shape
    
    # Pad to even dimensions
    ny_pad = ny + (ny & 1)
    nx_pad = nx + (nx & 1)
    
    # Pad arrays - vectorized
    rhs_pad = np.pad(rhs, ((0, ny_pad-ny), (0, nx_pad-nx)), mode='constant')
    
    # Pad face coefficients - vectorized
    beta_x_pad = np.ones((ny_pad, nx_pad + 1), dtype=beta_x.dtype)
    beta_x_pad[:ny, :beta_x.shape[1]] = beta_x
    if ny_pad > ny:
        beta_x_pad[ny:, :beta_x.shape[1]] = beta_x[-1:, :]
    if nx_pad > nx:
        beta_x_pad[:, nx+1:] = beta_x_pad[:, nx:nx+1]
    
    beta_y_pad = np.ones((ny_pad + 1, nx_pad), dtype=beta_y.dtype)
    beta_y_pad[:beta_y.shape[0], :nx] = beta_y
    if nx_pad > nx:
        beta_y_pad[:beta_y.shape[0], nx:] = beta_y[:, -1:]
    if ny_pad > ny:
        beta_y_pad[ny+1:, :] = beta_y_pad[ny:ny+1, :]
    
    # Initial guess
    if initial_guess is not None and initial_guess.shape == (ny, nx):
        phi_pad = np.pad(initial_guess, ((0, ny_pad-ny), (0, nx_pad-nx)), mode='edge')
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
        
        # Check convergence
        residual = compute_residual_mac_vectorized(phi_pad, rhs_pad, beta_x_pad, beta_y_pad, dx)
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