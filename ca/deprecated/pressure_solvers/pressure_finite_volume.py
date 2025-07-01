"""Finite volume approach for pressure equation with proper handling of density discontinuities.

The key insight: for discontinuous density fields, we need to compute fluxes at
cell faces using averaged values. This naturally handles material interfaces.
"""

import numpy as np
from pressure_solver import solve_pressure


def compute_face_averaged_divergence_rho_g(rho, gx, gy, dx):
    """Compute ∇·(ρg) using face-averaged density values.
    
    This handles discontinuous density fields correctly by computing
    ρg at cell faces through averaging adjacent cells.
    """
    ny, nx = rho.shape
    
    # Compute ρg at cell centers
    rho_gx = rho * gx
    rho_gy = rho * gy
    
    # Average to get (ρg)_x at vertical faces (i, j±½)
    rho_gx_faces_x = np.zeros((ny, nx + 1))
    rho_gx_faces_x[:, 1:-1] = 0.5 * (rho_gx[:, :-1] + rho_gx[:, 1:])
    rho_gx_faces_x[:, 0] = rho_gx[:, 0]      # Left boundary
    rho_gx_faces_x[:, -1] = rho_gx[:, -1]    # Right boundary
    
    # Average to get (ρg)_y at horizontal faces (i±½, j)
    rho_gy_faces_y = np.zeros((ny + 1, nx))
    rho_gy_faces_y[1:-1, :] = 0.5 * (rho_gy[:-1, :] + rho_gy[1:, :])
    rho_gy_faces_y[0, :] = rho_gy[0, :]      # Top boundary
    rho_gy_faces_y[-1, :] = rho_gy[-1, :]    # Bottom boundary
    
    # Compute divergence at cell centers
    div_rho_g = np.zeros((ny, nx))
    
    # ∂(ρg_x)/∂x using face values
    div_rho_g[:, :] += (rho_gx_faces_x[:, 1:] - rho_gx_faces_x[:, :-1]) / dx
    
    # ∂(ρg_y)/∂y using face values
    div_rho_g[:, :] += (rho_gy_faces_y[1:, :] - rho_gy_faces_y[:-1, :]) / dx
    
    return div_rho_g


def compute_pressure_gradient_finite_volume(pressure, dx):
    """Compute pressure gradient using finite volume approach.
    
    Returns pressure gradient at cell centers, computed from
    face-centered pressure differences.
    """
    ny, nx = pressure.shape
    
    # Pressure at vertical faces (average adjacent cells)
    p_faces_x = np.zeros((ny, nx + 1))
    p_faces_x[:, 1:-1] = 0.5 * (pressure[:, :-1] + pressure[:, 1:])
    p_faces_x[:, 0] = pressure[:, 0]      # Left boundary
    p_faces_x[:, -1] = pressure[:, -1]    # Right boundary
    
    # Pressure at horizontal faces
    p_faces_y = np.zeros((ny + 1, nx))
    p_faces_y[1:-1, :] = 0.5 * (pressure[:-1, :] + pressure[1:, :])
    p_faces_y[0, :] = pressure[0, :]      # Top boundary
    p_faces_y[-1, :] = pressure[-1, :]    # Bottom boundary
    
    # Compute gradients at cell centers from face values
    grad_p_x = np.zeros((ny, nx))
    grad_p_y = np.zeros((ny, nx))
    
    # For each cell, gradient is difference of face values
    grad_p_x[:, 1:-1] = (p_faces_x[:, 2:-1] - p_faces_x[:, 1:-2]) / dx
    grad_p_y[1:-1, :] = (p_faces_y[2:-1, :] - p_faces_y[1:-2, :]) / dx
    
    # Boundaries: one-sided differences
    grad_p_x[:, 0] = (pressure[:, 1] - pressure[:, 0]) / dx
    grad_p_x[:, -1] = (pressure[:, -1] - pressure[:, -2]) / dx
    grad_p_y[0, :] = (pressure[1, :] - pressure[0, :]) / dx
    grad_p_y[-1, :] = (pressure[-1, :] - pressure[-2, :]) / dx
    
    return grad_p_x, grad_p_y


def solve_pressure_finite_volume(rho, gx, gy, dx, **kwargs):
    """Solve pressure equation using finite volume approach.
    
    Solves: ∇²P = ∇·(ρg)
    
    With proper handling of discontinuous density fields.
    """
    # Compute RHS using face-averaged fluxes
    rhs = compute_face_averaged_divergence_rho_g(rho, gx, gy, dx)
    
    # Solve using existing fast multigrid solver
    pressure = solve_pressure(rhs, dx, **kwargs)
    
    return pressure


def compute_forces_finite_volume(rho, gx, gy, pressure, dx):
    """Compute total forces using finite volume approach."""
    # Gravity forces at cell centers
    fx_gravity = rho * gx
    fy_gravity = rho * gy
    
    # Pressure gradient forces
    grad_p_x, grad_p_y = compute_pressure_gradient_finite_volume(pressure, dx)
    
    # Total forces
    fx_total = fx_gravity - grad_p_x
    fy_total = fy_gravity - grad_p_y
    
    return fx_total, fy_total


def test_finite_volume_equilibrium():
    """Test hydrostatic equilibrium with finite volume approach."""
    # Setup
    ny, nx = 30, 10
    dx = 50.0
    g = 9.81
    
    # Sharp density transition
    rho = np.ones((ny, nx)) * 1.2  # air
    rho[10:, :] = 1000.0  # water
    
    # Uniform downward gravity
    gx = np.zeros((ny, nx))
    gy = np.full((ny, nx), g)
    
    print("FINITE VOLUME APPROACH TEST")
    print("=" * 60)
    
    # Method 1: Original approach (for comparison)
    print("\n1. Original approach (centered differences):")
    print("-" * 50)
    
    # Original RHS
    grad_rho_y = np.zeros_like(rho)
    grad_rho_y[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2 * dx)
    rhs_orig = g * grad_rho_y
    
    # Solve
    p_orig = solve_pressure(rhs_orig, dx, bc_type='neumann')
    
    # Original gradient
    grad_p_orig = np.zeros_like(p_orig)
    grad_p_orig[1:-1, :] = (p_orig[2:, :] - p_orig[:-2, :]) / (2 * dx)
    
    # Method 2: Finite volume approach
    print("\n2. Finite volume approach:")
    print("-" * 50)
    
    # Solve with finite volume
    p_fv = solve_pressure_finite_volume(rho, gx, gy, dx, bc_type='neumann')
    
    # Compute forces
    fx_fv, fy_fv = compute_forces_finite_volume(rho, gx, gy, p_fv, dx)
    
    # Compare results
    print("\nForce comparison at x=5:")
    print("   y | Material |  ρg (N/m³) | Original Net | FV Net Force")
    print("-" * 60)
    
    x = 5
    for y in [5, 9, 10, 11, 15, 20, 25]:
        rho_g = rho[y, x] * g
        net_orig = rho_g - grad_p_orig[y, x]
        net_fv = fy_fv[y, x]
        mat = "AIR" if y < 10 else "WATER"
        print(f"{y:4d} | {mat:8s} | {rho_g:10.1f} | {net_orig:12.3f} | {net_fv:12.3f}")
    
    # Check equilibrium in bulk regions
    print("\n\nEquilibrium in bulk regions:")
    
    # Air region
    air_max = np.max(np.abs(fy_fv[3:8, :]))
    print(f"Air (y=3-8): max |force| = {air_max:.3f} N/m³")
    
    # Water region
    water_max = np.max(np.abs(fy_fv[15:25, :]))
    print(f"Water (y=15-25): max |force| = {water_max:.3f} N/m³")
    
    # Overall statistics
    overall_max = np.max(np.abs(fy_fv))
    overall_avg = np.mean(np.abs(fy_fv))
    
    print(f"\nOverall max |force|: {overall_max:.1f} N/m³")
    print(f"Overall avg |force|: {overall_avg:.1f} N/m³")
    print(f"Reference ρ_water * g: {1000 * g:.1f} N/m³")
    
    # Success criteria
    bulk_success = max(air_max, water_max) < 100  # Small compared to ρg
    print(f"\n{'SUCCESS' if bulk_success else 'FAILURE'}: "
          f"Bulk regions {'are' if bulk_success else 'are NOT'} in equilibrium")


if __name__ == "__main__":
    test_finite_volume_equilibrium()