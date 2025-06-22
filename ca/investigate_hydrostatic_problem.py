"""Investigate why solving ∇²P = ∇·(ρg) doesn't give ∇P = ρg in bulk regions.

This is a fundamental investigation into the mathematical and numerical issues.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def analyze_the_mathematics():
    """Analyze the mathematical formulation."""
    
    print("MATHEMATICAL ANALYSIS")
    print("=" * 60)
    
    print("\n1. The Physical Requirement:")
    print("   For hydrostatic equilibrium: ∇P = ρg")
    
    print("\n2. The Standard Approach:")
    print("   Take divergence: ∇·(∇P) = ∇·(ρg)")
    print("   This gives: ∇²P = ∇·(ρg)")
    
    print("\n3. Expanding ∇·(ρg):")
    print("   ∇·(ρg) = ρ(∇·g) + g·(∇ρ)")
    
    print("\n4. For constant density regions:")
    print("   If ∇ρ = 0 and ∇·g = 0 (uniform gravity)")
    print("   Then ∇·(ρg) = 0")
    print("   So we solve ∇²P = 0 (Laplace equation)")
    
    print("\n5. THE KEY INSIGHT:")
    print("   Laplace equation ∇²P = 0 has infinitely many solutions!")
    print("   Examples that all satisfy ∇²P = 0:")
    print("   - P = constant (gives ∇P = 0)")
    print("   - P = ax + by (gives ∇P = (a,b))")
    print("   - P = x² - y² (gives ∇P = (2x, -2y))")
    
    print("\n6. The Problem:")
    print("   We want ∇P = ρg, but the equation ∇²P = 0")
    print("   doesn't uniquely determine this!")
    print("   Boundary conditions alone can't force the interior gradient.")
    
    print("\n7. Why This Happens:")
    print("   The Poisson equation is a SECOND-order PDE")
    print("   But ∇P = ρg is a FIRST-order constraint")
    print("   We've lost information by taking the divergence!")


def test_simple_1d_case():
    """Test a simple 1D case to understand the issue."""
    
    print("\n\nSIMPLE 1D TEST")
    print("=" * 60)
    
    # 1D domain
    n = 50
    dx = 1.0
    x = np.arange(n) * dx
    
    # Constant density and gravity
    rho = 1000.0  # kg/m³
    g = 9.81      # m/s²
    
    print(f"\nSetup: n={n} cells, ρ={rho} kg/m³, g={g} m/s²")
    
    # Build 1D Laplacian matrix
    # d²P/dx² = (P[i-1] - 2*P[i] + P[i+1]) / dx²
    main_diag = -2.0 * np.ones(n) / (dx * dx)
    off_diag = 1.0 * np.ones(n-1) / (dx * dx)
    
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n, n), format='csr')
    
    # RHS for ∇²P = ∇·(ρg)
    # In 1D with constant ρ and g: d/dx(ρg) = 0
    rhs = np.zeros(n)
    
    # Boundary conditions
    # Let's try P=0 at top (x=0) and bottom (x=L)
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1
    rhs[0] = 0
    rhs[-1] = 0
    A = A.tocsr()
    
    # Solve
    P = spsolve(A, rhs)
    
    # Compute gradient
    dP_dx = np.gradient(P, dx)
    
    # What we want
    target_gradient = rho * g * np.ones(n)
    
    print(f"\nResults at x=n/2:")
    print(f"  Computed dP/dx = {dP_dx[n//2]:.3f}")
    print(f"  Target ρg = {target_gradient[n//2]:.3f}")
    print(f"  Pressure P = {P[n//2]:.3f}")
    
    print("\nConclusion: With ∇²P = 0 and P=0 at boundaries,")
    print("we get P=0 everywhere, so dP/dx = 0, NOT ρg!")


def test_2d_water_column():
    """Test 2D water column - the classic case."""
    
    print("\n\n2D WATER COLUMN TEST")
    print("=" * 60)
    
    # 2D domain
    nx, ny = 20, 30
    dx = 1.0
    
    # Water below y=10, air above
    rho = np.ones((ny, nx)) * 1.2  # air
    rho[10:, :] = 1000.0  # water
    
    # Uniform gravity
    gx = 0.0
    gy = 9.81
    
    print(f"\nSetup: {nx}×{ny} grid, water below y=10")
    
    # Method 1: Standard RHS = g·∇ρ
    print("\nMethod 1: RHS = g·∇ρ")
    grad_rho_y = np.zeros_like(rho)
    grad_rho_y[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2 * dx)
    rhs1 = gy * grad_rho_y
    
    print(f"  RHS at interface (y=10): {rhs1[10, 10]:.1f}")
    print(f"  RHS in bulk water (y=20): {rhs1[20, 10]:.1f}")
    
    # Method 2: Full divergence RHS = ∇·(ρg)
    print("\nMethod 2: RHS = ∇·(ρg)")
    rho_gy = rho * gy
    rhs2 = np.zeros_like(rho)
    rhs2[1:-1, :] = (rho_gy[2:, :] - rho_gy[:-2, :]) / (2 * dx)
    
    print(f"  RHS at interface (y=10): {rhs2[10, 10]:.1f}")
    print(f"  RHS in bulk water (y=20): {rhs2[20, 10]:.1f}")
    
    print("\nKey observation: Both methods give RHS ≈ 0 in bulk regions!")
    print("This is why we get ∇P ≈ 0 instead of ∇P = ρg")
    
    # Solve with Method 2
    n = nx * ny
    
    # Build 2D Laplacian (5-point stencil)
    main_diag = -4.0 * np.ones(n) / (dx * dx)
    off_diag_x = np.ones(n-1) / (dx * dx)
    off_diag_y = np.ones(n-nx) / (dx * dx)
    
    # Zero out connections across row boundaries
    for i in range(1, ny):
        off_diag_x[i*nx - 1] = 0
    
    A = diags([main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y],
              [0, -1, 1, -nx, nx], shape=(n, n), format='csr')
    
    # Flatten RHS
    b = rhs2.flatten()
    
    # Neumann BC (modify matrix diagonal)
    A = A.tolil()
    # Top row
    for i in range(nx):
        A[i, i] += 1.0 / (dx * dx)
    # Bottom row  
    for i in range((ny-1)*nx, n):
        A[i, i] += 1.0 / (dx * dx)
    # Left/right columns
    for i in range(0, n, nx):
        A[i, i] += 1.0 / (dx * dx)
    for i in range(nx-1, n, nx):
        A[i, i] += 1.0 / (dx * dx)
    A = A.tocsr()
    
    # Solve
    P_flat = spsolve(A, b)
    P = P_flat.reshape(ny, nx)
    
    # Add constant to make average pressure reasonable
    P += 101325  # 1 atm
    
    # Compute pressure gradient
    grad_P_y = np.zeros_like(P)
    grad_P_y[1:-1, :] = (P[2:, :] - P[:-2, :]) / (2 * dx)
    
    # Compare to target
    target = rho * gy
    
    print("\n\nResults at x=10:")
    print("   y | ρg (target) | ∇P (computed) | Error")
    print("-" * 45)
    for y in [5, 10, 15, 20, 25]:
        error = grad_P_y[y, 10] - target[y, 10]
        print(f"{y:4d} | {target[y, 10]:11.1f} | {grad_P_y[y, 10]:13.1f} | {error:6.1f}")


def propose_solution():
    """Propose potential solutions."""
    
    print("\n\nPOTENTIAL SOLUTIONS")
    print("=" * 60)
    
    print("\n1. Modified Formulation:")
    print("   Instead of solving for P directly, solve for P - P_ref")
    print("   where P_ref is a reference pressure that already satisfies ∇P_ref = ρg")
    print("   But this just moves the problem - how do we get P_ref?")
    
    print("\n2. Augmented System:")
    print("   Solve the coupled system:")
    print("   ∇²P = ∇·(ρg)  (Poisson equation)")
    print("   ∇P = ρg        (Direct constraint)")
    print("   This is overdetermined but could use least squares")
    
    print("\n3. Projection Method (Current approach):")
    print("   Don't solve for static pressure")
    print("   Let it evolve dynamically through velocity projection")
    print("   This is what CFD codes actually do!")
    
    print("\n4. Stokes-like System:")
    print("   Solve for velocity and pressure together:")
    print("   -∇P + ρg = 0  (momentum)")
    print("   ∇·v = 0       (continuity)")
    print("   With v=0 for static case")
    
    print("\n5. Different Discretization:")
    print("   Use exactly compatible discrete operators")
    print("   Ensure discrete ∇· and ∇ are adjoints")
    print("   This might help but won't fix the fundamental issue")


if __name__ == "__main__":
    analyze_the_mathematics()
    test_simple_1d_case()
    test_2d_water_column()
    propose_solution()