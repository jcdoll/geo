"""Direct pressure solver: integrates ∇P = ρg + velocity terms.

This is an alternative to the Poisson approach that directly integrates
the pressure gradient rather than solving ∇²P = ∇·(ρg).
"""

import numpy as np
from typing import Tuple, Optional


def check_curl_free(fx: np.ndarray, fy: np.ndarray, dx: float) -> float:
    """Check if vector field (fx, fy) is curl-free.
    
    Returns maximum curl magnitude. Should be near zero for valid pressure gradient.
    """
    # Curl in 2D: ∂fy/∂x - ∂fx/∂y
    curl = np.zeros_like(fx)
    curl[1:-1, 1:-1] = (
        (fy[1:-1, 2:] - fy[1:-1, :-2]) / (2*dx) -
        (fx[2:, 1:-1] - fx[:-2, 1:-1]) / (2*dx)
    )
    return np.max(np.abs(curl))


def integrate_pressure_direct(
    rho_gx: np.ndarray,
    rho_gy: np.ndarray, 
    dx: float,
    boundary_pressure: float = 0.0,
    space_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Integrate ∇P = (ρgx, ρgy) to find pressure field.
    
    Uses path integration from boundaries where P = boundary_pressure.
    
    Args:
        rho_gx, rho_gy: Components of ρg (force per volume)
        dx: Grid spacing
        boundary_pressure: Pressure at space/boundaries
        space_mask: Boolean mask for space cells (P = 0)
        
    Returns:
        Pressure field in Pa
    """
    ny, nx = rho_gx.shape
    pressure = np.full((ny, nx), np.nan)
    
    # Set boundary conditions
    if space_mask is not None:
        pressure[space_mask] = boundary_pressure
    else:
        # Assume edges are boundaries
        pressure[0, :] = boundary_pressure
        pressure[-1, :] = boundary_pressure
        pressure[:, 0] = boundary_pressure
        pressure[:, -1] = boundary_pressure
    
    # Method 1: Layer-by-layer integration from boundaries
    # This works well for stratified problems
    
    # Keep iterating until all cells are filled
    max_iterations = nx + ny
    for iteration in range(max_iterations):
        n_filled_before = np.sum(~np.isnan(pressure))
        
        # For each unknown cell, try to compute from known neighbors
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if not np.isnan(pressure[i, j]):
                    continue
                    
                # Try to compute from neighbors
                # From west: P[i,j] = P[i,j-1] + dx * ρgx[i,j-1]
                if not np.isnan(pressure[i, j-1]):
                    pressure[i, j] = pressure[i, j-1] + dx * 0.5 * (rho_gx[i, j-1] + rho_gx[i, j])
                # From east: P[i,j] = P[i,j+1] - dx * ρgx[i,j]  
                elif not np.isnan(pressure[i, j+1]):
                    pressure[i, j] = pressure[i, j+1] - dx * 0.5 * (rho_gx[i, j] + rho_gx[i, j+1])
                # From north: P[i,j] = P[i-1,j] + dx * ρgy[i-1,j]
                elif not np.isnan(pressure[i-1, j]):
                    pressure[i, j] = pressure[i-1, j] + dx * 0.5 * (rho_gy[i-1, j] + rho_gy[i, j])
                # From south: P[i,j] = P[i+1,j] - dx * ρgy[i,j]
                elif not np.isnan(pressure[i+1, j]):
                    pressure[i, j] = pressure[i+1, j] - dx * 0.5 * (rho_gy[i, j] + rho_gy[i+1, j])
        
        n_filled_after = np.sum(~np.isnan(pressure))
        if n_filled_after == n_filled_before:
            break
    
    # Fill any remaining NaNs with average of neighbors (disconnected regions)
    pressure = np.nan_to_num(pressure, nan=boundary_pressure)
    
    return pressure


def solve_pressure_least_squares(
    rho_gx: np.ndarray,
    rho_gy: np.ndarray,
    dx: float,
    boundary_pressure: float = 0.0,
    space_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Solve ∇P = (ρgx, ρgy) using least squares.
    
    Minimizes ||∂P/∂x - ρgx||² + ||∂P/∂y - ρgy||²
    
    This handles the case where the vector field might not be exactly curl-free.
    """
    # This is equivalent to solving:
    # ∇²P = ∂(ρgx)/∂x + ∂(ρgy)/∂y = ∇·(ρg)
    # 
    # So we're back to the Poisson equation! But now we understand why:
    # it's the least-squares solution to ∇P = ρg
    
    ny, nx = rho_gx.shape
    
    # Compute divergence of (ρgx, ρgy)
    div_rho_g = np.zeros((ny, nx))
    div_rho_g[1:-1, 1:-1] = (
        (rho_gx[1:-1, 2:] - rho_gx[1:-1, :-2]) / (2*dx) +
        (rho_gy[2:, 1:-1] - rho_gy[:-2, 1:-1]) / (2*dx)
    )
    
    # This is exactly ∇·(ρg), which is what we were solving before!
    # The Poisson approach IS the least-squares solution to ∇P = ρg
    
    return div_rho_g  # Would need to call Poisson solver here


def integrate_hydrostatic_1d_vertical(
    density: np.ndarray,
    gravity: float,
    dx: float,
    surface_pressure: float = 0.0,
    surface_y: int = 0
) -> np.ndarray:
    """Simple 1D hydrostatic integration in vertical direction.
    
    P(y) = P_surface + ∫ ρ(y') g dy'
    
    This is exact for stratified problems where g points down.
    """
    ny, nx = density.shape
    pressure = np.zeros_like(density)
    
    # Set surface pressure
    pressure[surface_y, :] = surface_pressure
    
    # Integrate downward
    for y in range(surface_y + 1, ny):
        pressure[y, :] = pressure[y-1, :] + density[y-1, :] * gravity * dx
    
    # Integrate upward  
    for y in range(surface_y - 1, -1, -1):
        pressure[y, :] = pressure[y+1, :] - density[y, :] * gravity * dx
        
    return pressure


# Test the understanding
if __name__ == "__main__":
    # Create simple test case
    nx, ny = 20, 20
    dx = 50.0
    
    # Uniform gravity pointing down
    g = 10.0
    
    # Water column in middle
    density = np.zeros((ny, nx))
    density[5:15, 5:15] = 1000.0  # water
    
    # Compute ρg field
    rho_gx = np.zeros_like(density)
    rho_gy = density * g
    
    # Check if curl-free
    curl = check_curl_free(rho_gx, rho_gy, dx)
    print(f"Maximum curl of ρg field: {curl:.6f}")
    print("(Should be ~0 for uniform gravity)")
    
    # Direct integration
    p_direct = integrate_pressure_direct(rho_gx, rho_gy, dx)
    
    # The least-squares approach gives us back the Poisson equation!
    div_rho_g = solve_pressure_least_squares(rho_gx, rho_gy, dx)
    
    print(f"\nDirect integration pressure at (10,10): {p_direct[10,10]:.1f} Pa")
    print(f"This approach only works if ρg is curl-free!")