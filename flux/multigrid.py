"""
Fully vectorized multigrid solver for MAC (Marker-and-Cell) grids.

This version eliminates ALL loops in performance-critical sections.
Includes multiple smoothers (red-black, Jacobi) and configurable interface.
FIXED: Correct Neumann BC implementation using ghost cell method.
"""

import numpy as np
from typing import Optional, Tuple, Callable
from enum import Enum


class BoundaryCondition(Enum):
    """Boundary condition types."""
    NEUMANN = "neumann"     # ∂φ/∂n = 0
    DIRICHLET = "dirichlet" # φ = 0


class SmootherType(Enum):
    """Available smoother types."""
    RED_BLACK = "red_black"
    JACOBI = "jacobi"


def apply_bc_dirichlet(phi: np.ndarray):
    """Apply Dirichlet boundary conditions."""
    phi[0, :] = 0.0
    phi[-1, :] = 0.0
    phi[:, 0] = 0.0
    phi[:, -1] = 0.0


def _apply_neumann_bc_stencils(phi, rhs, beta_x, beta_y, dx2, ny, nx):
    """Apply Neumann BC through modified stencils at boundaries."""
    # Left boundary (j=0, i=1:ny-1)
    if ny > 2:
        be = beta_x[1:-1, 1]
        bn = beta_y[2:ny, 0]
        bs = beta_y[1:ny-1, 0]
        denom = 2*be + bn + bs
        mask = denom > 1e-12
        phi[1:-1, 0] = np.where(mask,
            (2*be * phi[1:-1, 1] + bn * phi[2:, 0] + bs * phi[:-2, 0] - dx2 * rhs[1:-1, 0]) / denom,
            phi[1:-1, 0])
    
    # Right boundary (j=nx-1, i=1:ny-1)
    if ny > 2:
        bw = beta_x[1:-1, nx-1]
        bn = beta_y[2:ny, nx-1]
        bs = beta_y[1:ny-1, nx-1]
        denom = 2*bw + bn + bs
        mask = denom > 1e-12
        phi[1:-1, -1] = np.where(mask,
            (2*bw * phi[1:-1, -2] + bn * phi[2:, -1] + bs * phi[:-2, -1] - dx2 * rhs[1:-1, -1]) / denom,
            phi[1:-1, -1])
    
    # Top boundary (i=0, j=1:nx-1)
    if nx > 2:
        be = beta_x[0, 2:nx]
        bw = beta_x[0, 1:nx-1]
        bn = beta_y[1, 1:-1]
        denom = be + bw + 2*bn
        mask = denom > 1e-12
        phi[0, 1:-1] = np.where(mask,
            (be * phi[0, 2:] + bw * phi[0, :-2] + 2*bn * phi[1, 1:-1] - dx2 * rhs[0, 1:-1]) / denom,
            phi[0, 1:-1])
    
    # Bottom boundary (i=ny-1, j=1:nx-1)
    if nx > 2:
        be = beta_x[ny-1, 2:nx]
        bw = beta_x[ny-1, 1:nx-1]
        bs = beta_y[ny-1, 1:-1]
        denom = be + bw + 2*bs
        mask = denom > 1e-12
        phi[-1, 1:-1] = np.where(mask,
            (be * phi[-1, 2:] + bw * phi[-1, :-2] + 2*bs * phi[-2, 1:-1] - dx2 * rhs[-1, 1:-1]) / denom,
            phi[-1, 1:-1])
    
    # Corners (only 4 points, minimal performance impact)
    # Top-left (0,0)
    be = beta_x[0, 1]
    bn = beta_y[1, 0]
    denom = 2*be + 2*bn
    if denom > 1e-12:
        phi[0, 0] = (2*be * phi[0, 1] + 2*bn * phi[1, 0] - dx2 * rhs[0, 0]) / denom
    
    # Top-right (0, nx-1)
    bw = beta_x[0, nx-1]
    bn = beta_y[1, nx-1]
    denom = 2*bw + 2*bn
    if denom > 1e-12:
        phi[0, -1] = (2*bw * phi[0, -2] + 2*bn * phi[1, -1] - dx2 * rhs[0, -1]) / denom
    
    # Bottom-left (ny-1, 0)
    be = beta_x[ny-1, 1]
    bs = beta_y[ny-1, 0]
    denom = 2*be + 2*bs
    if denom > 1e-12:
        phi[-1, 0] = (2*be * phi[-1, 1] + 2*bs * phi[-2, 0] - dx2 * rhs[-1, 0]) / denom
    
    # Bottom-right (ny-1, nx-1)
    bw = beta_x[ny-1, nx-1]
    bs = beta_y[ny-1, nx-1]
    denom = 2*bw + 2*bs
    if denom > 1e-12:
        phi[-1, -1] = (2*bw * phi[-1, -2] + 2*bs * phi[-2, -1] - dx2 * rhs[-1, -1]) / denom


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
    
    For Neumann BC: Uses correct ghost cell stencils at boundaries
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
        if bc_type == BoundaryCondition.NEUMANN:
            # Modified stencils at boundaries
            _apply_neumann_bc_stencils(phi, rhs, beta_x, beta_y, dx2, ny, nx)
        else:
            # Dirichlet: set boundaries to zero
            apply_bc_dirichlet(phi)


def smooth_jacobi_mac_vectorized(
    phi: np.ndarray,
    rhs: np.ndarray, 
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    dx: float,
    n_iter: int = 5,
    omega: float = 0.8,  # Relaxation parameter
    bc_type: BoundaryCondition = BoundaryCondition.NEUMANN
):
    """
    Weighted Jacobi smoother for MAC grid.
    
    This avoids the checkerboard patterns of red-black Gauss-Seidel.
    
    Uses face-centered coefficients:
    - beta_x: shape (ny, nx+1) - x-face coefficients  
    - beta_y: shape (ny+1, nx) - y-face coefficients
    """
    ny, nx = phi.shape
    dx2 = dx * dx
    
    for _ in range(n_iter):
        # Create a copy for Jacobi update
        phi_old = phi.copy()
        
        # Get interior slices
        phi_c = phi_old[1:-1, 1:-1]
        rhs_c = rhs[1:-1, 1:-1]
        
        # Neighbor values
        phi_e = phi_old[1:-1, 2:]
        phi_w = phi_old[1:-1, :-2]
        phi_n = phi_old[2:, 1:-1]
        phi_s = phi_old[:-2, 1:-1]
        
        # Face coefficients
        ny_int = phi_c.shape[0]
        nx_int = phi_c.shape[1]
        
        # Extract face coefficients for interior cells
        bx_e = beta_x[1:1+ny_int, 2:2+nx_int]  # East faces
        bx_w = beta_x[1:1+ny_int, 1:1+nx_int]  # West faces
        by_n = beta_y[2:2+ny_int, 1:1+nx_int]  # North faces
        by_s = beta_y[1:1+ny_int, 1:1+nx_int]  # South faces
        
        # Compute update for all interior points
        denom = bx_e + bx_w + by_n + by_s
        denom = np.where(denom > 1e-12, denom, 1.0)  # Avoid division by zero
        
        phi_new = (
            bx_e * phi_e + bx_w * phi_w +
            by_n * phi_n + by_s * phi_s - dx2 * rhs_c
        ) / denom
        
        # Weighted update (under-relaxation for stability)
        phi[1:-1, 1:-1] = (1 - omega) * phi_c + omega * phi_new
        
        # Apply boundary conditions
        if bc_type == BoundaryCondition.NEUMANN:
            # Modified stencils at boundaries
            _apply_neumann_bc_stencils(phi, rhs, beta_x, beta_y, dx2, ny, nx)
        else:
            # Dirichlet: set boundaries to zero
            apply_bc_dirichlet(phi)


def _compute_boundary_residuals_neumann(residual, phi, rhs, beta_x, beta_y, dx2, ny, nx):
    """Compute residuals at boundaries for Neumann BC."""
    # Left boundary
    if ny > 2:
        be = beta_x[1:-1, 1]
        bn = beta_y[2:ny, 0]
        bs = beta_y[1:ny-1, 0]
        laplacian = (
            2*be * phi[1:-1, 1] + bn * phi[2:, 0] + bs * phi[:-2, 0] -
            (2*be + bn + bs) * phi[1:-1, 0]
        ) / dx2
        residual[1:-1, 0] = rhs[1:-1, 0] - laplacian
    
    # Right boundary
    if ny > 2:
        bw = beta_x[1:-1, nx-1]
        bn = beta_y[2:ny, nx-1]
        bs = beta_y[1:ny-1, nx-1]
        laplacian = (
            2*bw * phi[1:-1, -2] + bn * phi[2:, -1] + bs * phi[:-2, -1] -
            (2*bw + bn + bs) * phi[1:-1, -1]
        ) / dx2
        residual[1:-1, -1] = rhs[1:-1, -1] - laplacian
    
    # Top boundary
    if nx > 2:
        be = beta_x[0, 2:nx]
        bw = beta_x[0, 1:nx-1]
        bn = beta_y[1, 1:-1]
        laplacian = (
            be * phi[0, 2:] + bw * phi[0, :-2] + 2*bn * phi[1, 1:-1] -
            (be + bw + 2*bn) * phi[0, 1:-1]
        ) / dx2
        residual[0, 1:-1] = rhs[0, 1:-1] - laplacian
    
    # Bottom boundary
    if nx > 2:
        be = beta_x[ny-1, 2:nx]
        bw = beta_x[ny-1, 1:nx-1]
        bs = beta_y[ny-1, 1:-1]
        laplacian = (
            be * phi[-1, 2:] + bw * phi[-1, :-2] + 2*bs * phi[-2, 1:-1] -
            (be + bw + 2*bs) * phi[-1, 1:-1]
        ) / dx2
        residual[-1, 1:-1] = rhs[-1, 1:-1] - laplacian
    
    # Corners
    # Top-left
    be = beta_x[0, 1]
    bn = beta_y[1, 0]
    laplacian = (2*be * phi[0, 1] + 2*bn * phi[1, 0] - (2*be + 2*bn) * phi[0, 0]) / dx2
    residual[0, 0] = rhs[0, 0] - laplacian
    
    # Top-right
    bw = beta_x[0, nx-1]
    bn = beta_y[1, nx-1]
    laplacian = (2*bw * phi[0, -2] + 2*bn * phi[1, -1] - (2*bw + 2*bn) * phi[0, -1]) / dx2
    residual[0, -1] = rhs[0, -1] - laplacian
    
    # Bottom-left
    be = beta_x[ny-1, 1]
    bs = beta_y[ny-1, 0]
    laplacian = (2*be * phi[-1, 1] + 2*bs * phi[-2, 0] - (2*be + 2*bs) * phi[-1, 0]) / dx2
    residual[-1, 0] = rhs[-1, 0] - laplacian
    
    # Bottom-right
    bw = beta_x[ny-1, nx-1]
    bs = beta_y[ny-1, nx-1]
    laplacian = (2*bw * phi[-1, -2] + 2*bs * phi[-2, -1] - (2*bw + 2*bs) * phi[-1, -1]) / dx2
    residual[-1, -1] = rhs[-1, -1] - laplacian


def compute_residual_mac_vectorized(
    phi: np.ndarray,
    rhs: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    dx: float,
    bc_type: BoundaryCondition = BoundaryCondition.NEUMANN
) -> np.ndarray:
    """
    Fully vectorized residual computation for MAC grid.
    
    r = rhs - A*phi where A*phi = -∇·(β∇φ)
    
    For Neumann BC: Uses modified stencils at boundaries
    For Dirichlet BC: Assumes phi=0 at boundaries
    """
    dx2 = dx * dx
    ny, nx = phi.shape
    residual = np.zeros_like(phi)
    
    # Compute Laplacian for interior points - fully vectorized
    ny_int = ny - 2
    nx_int = nx - 2
    
    if ny_int > 0 and nx_int > 0:
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
    
    if bc_type == BoundaryCondition.NEUMANN:
        # Compute residuals at boundaries with modified stencils
        _compute_boundary_residuals_neumann(residual, phi, rhs, beta_x, beta_y, dx2, ny, nx)
    # For Dirichlet, boundary residuals remain zero
    
    return residual


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
        residual = compute_residual_mac_vectorized(phi, rhs, beta_x, beta_y, dx, bc_type)
        
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
    
    Now with correct Neumann BC implementation!
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
        residual = compute_residual_mac_vectorized(phi_pad, rhs_pad, beta_x_pad, beta_y_pad, dx, bc_type)
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


def get_smoother(smoother_type: SmootherType) -> Callable:
    """Get the smoother function based on type."""
    if smoother_type == SmootherType.RED_BLACK:
        return smooth_redblack_mac_vectorized
    elif smoother_type == SmootherType.JACOBI:
        return smooth_jacobi_mac_vectorized
    else:
        raise ValueError(f"Unknown smoother type: {smoother_type}")


def mg_cycle_mac_vectorized_configurable(
    levels: list,
    phi: np.ndarray,
    bc_type: BoundaryCondition,
    smoother_type: SmootherType = SmootherType.RED_BLACK,
    pre_smooth: int = 3,
    post_smooth: int = 3
):
    """V-cycle with configurable smoother."""
    # Get the smoother function
    smoother = get_smoother(smoother_type)
    
    # Copy phi to finest level
    levels[0]['phi'][:] = phi
    
    # Downward sweep
    for i in range(len(levels) - 1):
        level = levels[i]
        
        # Pre-smooth
        smoother(
            level['phi'], level['rhs'],
            level['beta_x'], level['beta_y'],
            level['dx'], pre_smooth, bc_type=bc_type
        )
        
        # Compute residual
        level['res'] = compute_residual_mac_vectorized(
            level['phi'], level['rhs'],
            level['beta_x'], level['beta_y'],
            level['dx'], bc_type
        )
        
        # Restrict residual
        levels[i+1]['rhs'] = restrict_full_weighting_vectorized(level['res'])
        levels[i+1]['phi'].fill(0)
    
    # Coarsest level - more iterations
    coarse = levels[-1]
    smoother(
        coarse['phi'], coarse['rhs'],
        coarse['beta_x'], coarse['beta_y'],
        coarse['dx'], 20, bc_type=bc_type
    )
    
    # Upward sweep
    for i in range(len(levels) - 2, -1, -1):
        level = levels[i]
        
        # Prolong correction
        correction = prolong_bilinear_vectorized(
            levels[i+1]['phi'],
            (level['ny'], level['nx'])
        )
        level['phi'] += correction
        
        # Post-smooth
        smoother(
            level['phi'], level['rhs'],
            level['beta_x'], level['beta_y'],
            level['dx'], post_smooth, bc_type=bc_type
        )
    
    # Copy result back
    phi[:] = levels[0]['phi']


def solve_mac_poisson_configurable(
    rhs: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    dx: float,
    *,
    tol: float = 1e-6,
    max_cycles: int = 50,
    bc_type: BoundaryCondition = BoundaryCondition.NEUMANN,
    initial_guess: Optional[np.ndarray] = None,
    verbose: bool = False,
    smoother_type: SmootherType = SmootherType.RED_BLACK,
    pre_smooth: int = 3,
    post_smooth: int = 3
) -> np.ndarray:
    """
    Solve variable-coefficient Poisson equation with configurable smoother.
    
    Args:
        rhs: Right-hand side
        beta_x, beta_y: Face-centered coefficients
        dx: Grid spacing
        tol: Convergence tolerance
        max_cycles: Maximum V-cycles
        bc_type: Boundary condition type
        initial_guess: Initial guess for solution
        verbose: Print convergence info
        smoother_type: Which smoother to use
        pre_smooth: Pre-smoothing iterations
        post_smooth: Post-smoothing iterations
        
    Returns:
        phi: Solution
    """
    ny, nx = rhs.shape
    
    # Pad to even dimensions
    ny_pad = ny + (ny & 1)
    nx_pad = nx + (nx & 1)
    
    # Pad arrays if needed
    if ny_pad != ny or nx_pad != nx:
        rhs_pad = np.zeros((ny_pad, nx_pad))
        rhs_pad[:ny, :nx] = rhs
        
        beta_x_pad = np.zeros((ny_pad, nx_pad + 1))
        beta_x_pad[:ny, :beta_x.shape[1]] = beta_x
        if ny_pad > ny:
            beta_x_pad[ny, :beta_x.shape[1]] = beta_x[-1, :]
        if nx_pad > nx:
            beta_x_pad[:, -1] = beta_x_pad[:, -2]
            
        beta_y_pad = np.zeros((ny_pad + 1, nx_pad))
        beta_y_pad[:beta_y.shape[0], :nx] = beta_y
        if ny_pad > ny:
            beta_y_pad[-1, :] = beta_y_pad[-2, :]
        if nx_pad > nx:
            beta_y_pad[:beta_y.shape[0], nx] = beta_y[:, -1]
    else:
        rhs_pad = rhs
        beta_x_pad = beta_x
        beta_y_pad = beta_y
    
    # Initial guess
    if initial_guess is not None and initial_guess.shape == (ny, nx):
        phi = np.zeros((ny_pad, nx_pad))
        phi[:ny, :nx] = initial_guess
    else:
        phi = np.zeros((ny_pad, nx_pad))
    
    # Build multigrid hierarchy
    levels = []
    rhs_level = rhs_pad
    bx_level = beta_x_pad
    by_level = beta_y_pad
    
    ny_level, nx_level = rhs_level.shape
    level = 0
    
    while ny_level > 3 and nx_level > 3:
        levels.append({
            'ny': ny_level,
            'nx': nx_level,
            'dx': dx * (2 ** level),
            'rhs': rhs_level if level == 0 else np.zeros((ny_level, nx_level)),
            'beta_x': bx_level,
            'beta_y': by_level,
            'phi': np.zeros((ny_level, nx_level)),
            'res': np.zeros((ny_level, nx_level))
        })
        
        # Restrict to coarser level
        ny_level = (ny_level + 1) // 2
        nx_level = (nx_level + 1) // 2
        
        bx_level, by_level = restrict_face_coeffs_mac_ultra_vectorized(bx_level, by_level)
        level += 1
    
    # Main V-cycle iteration
    for cycle in range(max_cycles):
        # V-cycle with selected smoother
        mg_cycle_mac_vectorized_configurable(
            levels, phi, bc_type,
            smoother_type=smoother_type,
            pre_smooth=pre_smooth,
            post_smooth=post_smooth
        )
        
        # Check convergence
        res = compute_residual_mac_vectorized(
            phi, rhs_pad, beta_x_pad, beta_y_pad, dx, bc_type
        )
        res_norm = np.linalg.norm(res)
        
        if verbose and cycle % 10 == 0:
            print(f"[{smoother_type.value}] Cycle {cycle}: residual = {res_norm:.6e}")
        
        if res_norm < tol:
            if verbose:
                print(f"[{smoother_type.value}] Converged in {cycle + 1} cycles")
            break
    
    # Extract original size
    return phi[:ny, :nx]