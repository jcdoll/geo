"""Direct test of hydrostatic equilibrium using staggered grid operators."""

import numpy as np
from staggered_grid_operators import gradient_staggered, divergence_staggered
from pressure_solver import solve_pressure


def test_hydrostatic_with_staggered_rhs():
    """Test hydrostatic equilibrium with properly constructed RHS."""
    # Setup
    ny, nx = 30, 10
    dx = 50.0  # 50m cells
    g = 9.81
    
    # Density field: air above, water below
    density = np.ones((ny, nx)) * 1.2  # air density
    density[10:, :] = 1000.0  # water density
    
    # Method 1: Current approach (centered differences for RHS)
    print("Method 1: Centered difference RHS")
    print("-" * 50)
    
    # Standard RHS: g * ∂ρ/∂y
    grad_rho_y = np.zeros_like(density)
    grad_rho_y[1:-1, :] = (density[2:, :] - density[:-2, :]) / (2 * dx)
    rhs_centered = g * grad_rho_y
    
    # Solve
    pressure_centered = solve_pressure(rhs_centered, dx, bc_type='neumann')
    
    # Check gradient
    grad_p_y_centered = np.zeros_like(pressure_centered)
    grad_p_y_centered[1:-1, :] = (pressure_centered[2:, :] - pressure_centered[:-2, :]) / (2 * dx)
    
    x = 5
    for y in [8, 10, 12, 15, 20]:
        actual = grad_p_y_centered[y, x]
        target = density[y, x] * g
        error = abs(actual - target)
        mat = "AIR" if y < 10 else "WATER"
        print(f"y={y:2d} ({mat:5s}) | ∇P={actual:8.1f} | ρg={target:8.1f} | error={error:8.1f}")
    
    # Method 2: Staggered grid RHS
    print("\nMethod 2: Staggered grid RHS")
    print("-" * 50)
    
    # Build RHS using staggered divergence of ρg
    rho_g_x = np.zeros_like(density)  # No x-gravity
    rho_g_y = density * g
    
    # Interpolate to faces
    rho_g_x_face = np.zeros((ny, nx+1))
    rho_g_y_face = np.zeros((ny+1, nx))
    
    rho_g_x_face[:, 1:-1] = 0.5 * (rho_g_x[:, :-1] + rho_g_x[:, 1:])
    rho_g_y_face[1:-1, :] = 0.5 * (rho_g_y[:-1, :] + rho_g_y[1:, :])
    
    # Boundaries: use one-sided values
    rho_g_y_face[0, :] = rho_g_y[0, :]
    rho_g_y_face[-1, :] = rho_g_y[-1, :]
    
    # Apply divergence
    rhs_staggered = divergence_staggered(rho_g_x_face, rho_g_y_face, dx)
    
    # Solve with staggered RHS
    pressure_staggered = solve_pressure(rhs_staggered, dx, bc_type='neumann')
    
    # Get staggered gradient
    grad_x_stag, grad_y_stag = gradient_staggered(pressure_staggered, dx)
    
    # Check forces at cell centers (average from faces)
    for y in [8, 10, 12, 15, 20]:
        # Average y-gradient from faces to center
        grad_p = 0.5 * (grad_y_stag[y, x] + grad_y_stag[y+1, x])
        target = density[y, x] * g
        error = abs(grad_p - target)
        mat = "AIR" if y < 10 else "WATER"
        print(f"y={y:2d} ({mat:5s}) | ∇P={grad_p:8.1f} | ρg={target:8.1f} | error={error:8.1f}")
    
    # Method 3: Direct construction of hydrostatic pressure
    print("\nMethod 3: Direct hydrostatic pressure")
    print("-" * 50)
    
    # Build pressure by integration
    pressure_direct = np.zeros_like(density)
    
    # Integrate from top: P[i+1] = P[i] + ρ[i] * g * dx
    for i in range(ny-1):
        pressure_direct[i+1, :] = pressure_direct[i, :] + density[i, :] * g * dx
    
    # Check gradient  
    grad_p_y_direct = np.zeros_like(pressure_direct)
    grad_p_y_direct[1:-1, :] = (pressure_direct[2:, :] - pressure_direct[:-2, :]) / (2 * dx)
    
    for y in [8, 10, 12, 15, 20]:
        actual = grad_p_y_direct[y, x]
        target = density[y, x] * g
        error = abs(actual - target)
        mat = "AIR" if y < 10 else "WATER"
        print(f"y={y:2d} ({mat:5s}) | ∇P={actual:8.1f} | ρg={target:8.1f} | error={error:8.1f}")
    
    print("\nConclusion:")
    print("- Centered differences give ∇P ≈ 0 in bulk (wrong!)")
    print("- Staggered grid RHS still doesn't fix it with current solver")
    print("- Direct integration gives correct hydrostatic pressure")
    print("\nThe issue: standard Laplacian doesn't match staggered divergence!")


if __name__ == "__main__":
    test_hydrostatic_with_staggered_rhs()