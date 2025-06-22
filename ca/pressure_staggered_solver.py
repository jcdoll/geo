"""Pressure solver using staggered grid operators for correct hydrostatic equilibrium.

This solver uses the MAC (Marker-And-Cell) staggered grid layout:
- Pressure: cell centers
- Forces/velocities: face centers

The key is using adjoint-consistent gradient and divergence operators.
"""

import numpy as np
from staggered_grid_operators import gradient_staggered, divergence_staggered


def laplacian_staggered(p, dx):
    """Staggered grid Laplacian: div(grad(p)).
    
    This is the composition of gradient_staggered and divergence_staggered,
    ensuring consistency between the operators.
    """
    grad_x, grad_y = gradient_staggered(p, dx)
    return divergence_staggered(grad_x, grad_y, dx)


def solve_pressure_staggered(rho_g_x, rho_g_y, dx, tol=1e-6, max_iter=10000, 
                             bc_type='neumann'):
    """Solve pressure equation using staggered grid operators.
    
    Solves: ∇²P = ∇·(ρg)
    
    Args:
        rho_g_x: x-component of ρg at cell centers (typically 0)
        rho_g_y: y-component of ρg at cell centers
        dx: Grid spacing
        tol: Convergence tolerance
        max_iter: Maximum iterations
        bc_type: Boundary condition type ('dirichlet' or 'neumann')
        
    Returns:
        pressure: Pressure field at cell centers
    """
    ny, nx = rho_g_x.shape
    
    # Build RHS: ∇·(ρg)
    # First interpolate ρg to faces
    rho_g_x_face = np.zeros((ny, nx+1))
    rho_g_y_face = np.zeros((ny+1, nx))
    
    # Average to get values at faces
    rho_g_x_face[:, 1:-1] = 0.5 * (rho_g_x[:, :-1] + rho_g_x[:, 1:])
    rho_g_y_face[1:-1, :] = 0.5 * (rho_g_y[:-1, :] + rho_g_y[1:, :])
    
    # Apply divergence to get RHS
    rhs = divergence_staggered(rho_g_x_face, rho_g_y_face, dx)
    
    # Initialize pressure
    pressure = np.zeros((ny, nx))
    
    # Use direct solve for small problems, iterative for large
    if ny * nx < 10000:
        # Build matrix explicitly for small problems
        return _solve_direct_staggered(rhs, dx, bc_type)
    
    # Gauss-Seidel iteration for larger problems
    for iteration in range(max_iter):
        pressure_old = pressure.copy()
        
        # Gauss-Seidel sweep
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                # Staggered Laplacian stencil
                pressure[i, j] = 0.25 * (
                    pressure[i-1, j] + pressure[i+1, j] +
                    pressure[i, j-1] + pressure[i, j+1] -
                    dx * dx * rhs[i, j]
                )
        
        # Apply boundary conditions
        if bc_type == 'neumann':
            # Zero gradient at boundaries
            pressure[0, :] = pressure[1, :]
            pressure[-1, :] = pressure[-2, :]
            pressure[:, 0] = pressure[:, 1]
            pressure[:, -1] = pressure[:, -2]
        elif bc_type == 'dirichlet':
            # Zero pressure at boundaries
            pressure[0, :] = 0
            pressure[-1, :] = 0
            pressure[:, 0] = 0
            pressure[:, -1] = 0
        
        # Check convergence
        max_change = np.max(np.abs(pressure - pressure_old))
        if max_change < tol:
            break
    
    if iteration == max_iter - 1:
        print(f"Warning: Pressure solver did not converge (max change: {max_change:.2e})")
    
    return pressure


def _solve_direct_staggered(rhs, dx, bc_type):
    """Direct solve using scipy sparse matrices."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    
    ny, nx = rhs.shape
    n = ny * nx
    
    # Build Laplacian matrix
    main_diag = -4.0 * np.ones(n) / (dx * dx)
    off_diag = 1.0 * np.ones(n-1) / (dx * dx)
    off_diag_y = 1.0 * np.ones(n-nx) / (dx * dx)
    
    # Handle boundaries in off-diagonals
    for i in range(1, ny):
        off_diag[i*nx - 1] = 0  # No connection across rows
    
    diagonals = [main_diag, off_diag, off_diag, off_diag_y, off_diag_y]
    offsets = [0, -1, 1, -nx, nx]
    
    A = diags(diagonals, offsets, shape=(n, n), format='csr')
    
    # Flatten RHS
    b = rhs.flatten()
    
    # Modify for boundary conditions
    if bc_type == 'neumann':
        # Modify matrix for Neumann BC
        for i in range(nx):  # Top row
            A[i, i] += 1.0 / (dx * dx)  # Add back the boundary term
        for i in range((ny-1)*nx, n):  # Bottom row
            A[i, i] += 1.0 / (dx * dx)
        for i in range(0, n, nx):  # Left column
            A[i, i] += 1.0 / (dx * dx)
        for i in range(nx-1, n, nx):  # Right column
            A[i, i] += 1.0 / (dx * dx)
    
    # Solve
    p_flat = spsolve(A, b)
    
    return p_flat.reshape(ny, nx)


def compute_pressure_forces_staggered(pressure, dx):
    """Compute pressure gradient forces using staggered grid.
    
    Returns forces at cell centers by averaging face values.
    """
    # Get pressure gradients at faces
    grad_x, grad_y = gradient_staggered(pressure, dx)
    
    # Pressure force is negative gradient
    force_x_face = -grad_x
    force_y_face = -grad_y
    
    # Average back to cell centers for use in velocity update
    ny, nx = pressure.shape
    force_x = np.zeros((ny, nx))
    force_y = np.zeros((ny, nx))
    
    # Average x-forces from faces to centers
    force_x[:, :] = 0.5 * (force_x_face[:, :-1] + force_x_face[:, 1:])
    
    # Average y-forces from faces to centers  
    force_y[:, :] = 0.5 * (force_y_face[:-1, :] + force_y_face[1:, :])
    
    return force_x, force_y


def test_hydrostatic_equilibrium():
    """Test that water column achieves hydrostatic equilibrium."""
    # Setup
    ny, nx = 30, 10
    dx = 50.0  # 50m cells
    g = 9.81
    
    # Density field: air above, water below
    density = np.ones((ny, nx)) * 1.2  # air density
    density[10:, :] = 1000.0  # water density
    
    # Gravity forces
    rho_g_x = np.zeros_like(density)
    rho_g_y = density * g
    
    # Solve for pressure
    pressure = solve_pressure_staggered(rho_g_x, rho_g_y, dx)
    
    # Get pressure forces
    fx, fy = compute_pressure_forces_staggered(pressure, dx)
    
    # Check force balance: pressure gradient should balance gravity
    print("Hydrostatic Equilibrium Test")
    print("-" * 50)
    print("Location | ρg (gravity) | -∇P (pressure) | Net Force")
    print("-" * 50)
    
    x = nx // 2
    for y in [5, 10, 15, 20, 25]:
        gravity = rho_g_y[y, x]
        pressure_force = fy[y, x]
        net = gravity + pressure_force
        mat = "AIR" if y < 10 else "WATER"
        print(f"y={y:2d} ({mat:5s}) | {gravity:12.1f} | {pressure_force:12.1f} | {net:10.6f}")
    
    # Check that net forces are near zero
    max_net_force = np.max(np.abs(rho_g_y + fy))
    print(f"\nMax net force: {max_net_force:.6f} N/m³")
    print(f"Success: {'YES' if max_net_force < 1.0 else 'NO'}")
    
    return pressure, fx, fy


if __name__ == "__main__":
    test_hydrostatic_equilibrium()