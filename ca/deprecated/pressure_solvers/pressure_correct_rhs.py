"""Correct RHS formulation for pressure equation to achieve hydrostatic equilibrium.

The key insight: for hydrostatic equilibrium ∇P = ρg, we need to solve:
∇²P = ∇·(ρg)

But we must compute the FULL divergence, not just g·∇ρ.
"""

import numpy as np
from pressure_solver import solve_pressure


def compute_hydrostatic_rhs(rho, gx, gy, dx):
    """Compute RHS for pressure equation: ∇²P = ∇·(ρg).
    
    This computes the FULL divergence ∇·(ρg), not just g·∇ρ.
    
    For discrete divergence:
    ∇·(ρg) = ∂(ρgx)/∂x + ∂(ρgy)/∂y
    
    Using centered differences that match the 5-point Laplacian.
    """
    ny, nx = rho.shape
    
    # Compute ρg at cell centers
    rho_gx = rho * gx
    rho_gy = rho * gy
    
    # Initialize RHS
    rhs = np.zeros_like(rho)
    
    # Centered differences for divergence (matching Laplacian discretization)
    # ∂(ρgx)/∂x
    rhs[1:-1, 1:-1] += (rho_gx[1:-1, 2:] - rho_gx[1:-1, :-2]) / (2 * dx)
    
    # ∂(ρgy)/∂y  
    rhs[1:-1, 1:-1] += (rho_gy[2:, 1:-1] - rho_gy[:-2, 1:-1]) / (2 * dx)
    
    # Boundary handling for Neumann BC
    # Top/bottom boundaries
    rhs[0, 1:-1] = ((rho_gx[0, 2:] - rho_gx[0, :-2]) / (2 * dx) +
                    (rho_gy[1, 1:-1] - rho_gy[0, 1:-1]) / dx)
    rhs[-1, 1:-1] = ((rho_gx[-1, 2:] - rho_gx[-1, :-2]) / (2 * dx) +
                     (rho_gy[-1, 1:-1] - rho_gy[-2, 1:-1]) / dx)
    
    # Left/right boundaries
    rhs[1:-1, 0] = ((rho_gx[1:-1, 1] - rho_gx[1:-1, 0]) / dx +
                    (rho_gy[2:, 0] - rho_gy[:-2, 0]) / (2 * dx))
    rhs[1:-1, -1] = ((rho_gx[1:-1, -1] - rho_gx[1:-1, -2]) / dx +
                     (rho_gy[2:, -1] - rho_gy[:-2, -1]) / (2 * dx))
    
    # Corners
    rhs[0, 0] = ((rho_gx[0, 1] - rho_gx[0, 0]) / dx +
                 (rho_gy[1, 0] - rho_gy[0, 0]) / dx)
    rhs[0, -1] = ((rho_gx[0, -1] - rho_gx[0, -2]) / dx +
                  (rho_gy[1, -1] - rho_gy[0, -1]) / dx)
    rhs[-1, 0] = ((rho_gx[-1, 1] - rho_gx[-1, 0]) / dx +
                  (rho_gy[-1, 0] - rho_gy[-2, 0]) / dx)
    rhs[-1, -1] = ((rho_gx[-1, -1] - rho_gx[-1, -2]) / dx +
                   (rho_gy[-1, -1] - rho_gy[-2, -1]) / dx)
    
    return rhs


def test_correct_rhs():
    """Test the corrected RHS formulation."""
    # Setup
    ny, nx = 30, 10
    dx = 50.0
    g = 9.81
    
    # Uniform density field (pure water)
    rho = np.full((ny, nx), 1000.0)
    
    # Uniform gravity
    gx = np.zeros((ny, nx))
    gy = np.full((ny, nx), g)
    
    print("TEST: Uniform water with correct RHS")
    print("=" * 50)
    
    # Method 1: Wrong RHS (g·∇ρ) - what's currently used
    grad_rho_y = np.zeros_like(rho)
    grad_rho_y[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2 * dx)
    rhs_wrong = g * grad_rho_y
    
    print(f"Wrong RHS (g·∇ρ): max = {np.max(np.abs(rhs_wrong)):.3e}")
    print("(Should be ~0 for uniform density)")
    
    # Method 2: Correct RHS (∇·(ρg))
    rhs_correct = compute_hydrostatic_rhs(rho, gx, gy, dx)
    
    print(f"\nCorrect RHS (∇·(ρg)): max = {np.max(np.abs(rhs_correct)):.3e}")
    print("(Should be ~0 for uniform density AND uniform gravity)")
    
    # The issue: BOTH give zero RHS for uniform fields!
    # This is why we get ∇P ≈ 0 instead of ∇P = ρg
    
    print("\n" + "="*50)
    print("CONCLUSION: The fundamental issue is that for uniform ρ and g:")
    print("- ∇·(ρg) = 0 (no sources/sinks)")
    print("- So ∇²P = 0 (Laplace equation)")
    print("- P is harmonic, but ∇P is underdetermined!")
    print("- Boundary conditions alone don't force ∇P = ρg")
    
    print("\nTHE REAL SOLUTION:")
    print("We need a different formulation that directly enforces ∇P = ρg")
    print("One approach: Solve for pressure INCREMENT from a reference state")
    
    # Test with non-uniform density
    print("\n" + "="*50)
    print("TEST: Water column with air above")
    
    rho2 = np.ones((ny, nx)) * 1.2  # air
    rho2[10:, :] = 1000.0  # water
    
    rhs2 = compute_hydrostatic_rhs(rho2, gx, gy, dx)
    
    print(f"\nRHS at interface (y=9-11, x=5):")
    for y in range(9, 12):
        print(f"y={y}: rhs = {rhs2[y, 5]:.3f}")
    
    print(f"\nRHS in bulk water (y=20, x=5): {rhs2[20, 5]:.3e}")
    print("(Still ~0 in bulk regions!)")


if __name__ == "__main__":
    test_correct_rhs()