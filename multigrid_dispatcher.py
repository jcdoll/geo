"""
Multigrid solver with selectable smoothers to compare performance and accuracy.
"""

import numpy as np
from typing import Optional, Tuple, Callable
from enum import Enum

# Import smoothers
from multigrid import (
    BoundaryCondition,
    apply_bc,
    smooth_redblack_mac_vectorized,
    restrict_full_weighting_vectorized,
    prolong_bilinear_vectorized,
    compute_residual_mac_vectorized,
    restrict_face_coeffs_mac_ultra_vectorized
)
from multigrid_jacobi import smooth_jacobi_mac_vectorized


class SmootherType(Enum):
    """Available smoother types."""
    RED_BLACK = "red_black"
    JACOBI = "jacobi"
    GAUSS_SEIDEL = "gauss_seidel"  # Future implementation


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
            level['dx']
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
            phi, rhs_pad, beta_x_pad, beta_y_pad, dx
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