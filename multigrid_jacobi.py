"""
Modified multigrid solver with Jacobi smoother to avoid checkerboard patterns.
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
        apply_bc(phi, bc_type)


# Import other functions from original multigrid
from multigrid import (
    restrict_full_weighting_vectorized,
    prolong_bilinear_vectorized,
    compute_residual_mac_vectorized,
    restrict_face_coeffs_mac_ultra_vectorized
)


def solve_mac_poisson_jacobi(
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
    smoother: str = "jacobi"  # New parameter
) -> np.ndarray:
    """
    Solve variable-coefficient Poisson equation on MAC grid using Jacobi smoother.
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
    
    # Build multigrid hierarchy with Jacobi smoother
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
    
    # Main V-cycle iteration with Jacobi smoother
    for cycle in range(max_cycles):
        # V-cycle
        mg_cycle_mac_vectorized_jacobi(levels, phi, bc_type)
        
        # Check convergence
        res = compute_residual_mac_vectorized(
            phi, rhs_pad, beta_x_pad, beta_y_pad, dx
        )
        res_norm = np.linalg.norm(res)
        
        if verbose and cycle % 10 == 0:
            print(f"Cycle {cycle}: residual = {res_norm:.6e}")
        
        if res_norm < tol:
            if verbose:
                print(f"Converged in {cycle + 1} cycles")
            break
    
    # Extract original size
    return phi[:ny, :nx]


def mg_cycle_mac_vectorized_jacobi(
    levels: list,
    phi: np.ndarray,
    bc_type: BoundaryCondition,
    pre_smooth: int = 3,
    post_smooth: int = 3
):
    """V-cycle using Jacobi smoother instead of red-black."""
    # Copy phi to finest level
    levels[0]['phi'][:] = phi
    
    # Downward sweep
    for i in range(len(levels) - 1):
        level = levels[i]
        
        # Pre-smooth with Jacobi
        smooth_jacobi_mac_vectorized(
            level['phi'], level['rhs'],
            level['beta_x'], level['beta_y'],
            level['dx'], pre_smooth, bc_type=bc_type
        )
        
        # Compute residual
        level['res'] = compute_residual_mac_vectorized(
            level['phi'], level['rhs'],
            level['beta_x'], level['beta_y'],
            level['dx']
        )
        
        # Restrict residual
        levels[i+1]['rhs'] = restrict_full_weighting_vectorized(level['res'])
        levels[i+1]['phi'].fill(0)
    
    # Coarsest level - more iterations
    coarse = levels[-1]
    smooth_jacobi_mac_vectorized(
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
        
        # Post-smooth with Jacobi
        smooth_jacobi_mac_vectorized(
            level['phi'], level['rhs'],
            level['beta_x'], level['beta_y'],
            level['dx'], post_smooth, bc_type=bc_type
        )
    
    # Copy result back
    phi[:] = levels[0]['phi']