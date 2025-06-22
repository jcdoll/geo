"""Analysis of different discretization schemes for pressure equation.

The key challenge: we need consistent discretization between:
1. The Laplacian operator in the pressure solver
2. The gradient operator for computing forces
3. The divergence operator for building the RHS

For hydrostatic equilibrium, we need: ∇P = ρg
Taking divergence: ∇²P = ∇·(ρg)
"""

import numpy as np


def analyze_discretizations():
    """Compare different discretization approaches."""
    
    print("DISCRETIZATION ANALYSIS FOR PRESSURE EQUATION")
    print("=" * 60)
    
    print("\n1. CURRENT APPROACH (INCONSISTENT):")
    print("-" * 40)
    print("Laplacian (5-point stencil):")
    print("  L[P] = (P[i+1,j] + P[i-1,j] + P[i,j+1] + P[i,j-1] - 4*P[i,j])/dx²")
    print("\nGradient (centered differences):")
    print("  ∂P/∂x = (P[i,j+1] - P[i,j-1])/(2*dx)")
    print("  ∂P/∂y = (P[i+1,j] - P[i-1,j])/(2*dx)")
    print("\nProblem: These operators are NOT adjoint!")
    print("Result: ∇P ≠ ρg in bulk regions")
    
    print("\n\n2. STAGGERED MAC GRID (ADJOINT-CONSISTENT):")
    print("-" * 40)
    print("Layout:")
    print("  - Pressure P at cell centers (i,j)")
    print("  - u-velocity at vertical faces (i,j+½)")
    print("  - v-velocity at horizontal faces (i+½,j)")
    print("\nGradient (centers → faces):")
    print("  (∇P)_x[i,j+½] = (P[i,j+1] - P[i,j])/dx")
    print("  (∇P)_y[i+½,j] = (P[i+1,j] - P[i,j])/dx")
    print("\nDivergence (faces → centers):")
    print("  (∇·u)[i,j] = (u[i,j+½] - u[i,j-½])/dx + (v[i+½,j] - v[i-½,j])/dx")
    print("\nLaplacian = ∇·∇:")
    print("  L[P] = ((P[i,j+1] - P[i,j]) - (P[i,j] - P[i,j-1]))/dx²")
    print("       + ((P[i+1,j] - P[i,j]) - (P[i,j] - P[i-1,j]))/dx²")
    print("       = (P[i+1,j] + P[i-1,j] + P[i,j+1] + P[i,j-1] - 4*P[i,j])/dx²")
    print("\nAdvantages:")
    print("  ✓ Operators are exactly adjoint: <∇·u, p> = -<u, ∇p>")
    print("  ✓ Ensures correct hydrostatic equilibrium")
    print("  ✓ Conservative discretization")
    print("Disadvantages:")
    print("  - Requires interpolation between faces and centers")
    print("  - More complex implementation")
    
    print("\n\n3. CONSISTENT CELL-CENTERED (MODIFIED GRADIENT):")
    print("-" * 40)
    print("Key insight: Use gradient operator consistent with 5-point Laplacian")
    print("\nModified gradient at cell centers:")
    print("  ∂P/∂x ≈ (P[i,j+1] - P[i,j-1])/(2*dx)  [standard centered]")
    print("  BUT for RHS, use:")
    print("  ∇·(ρg) computed to match what Laplacian inverts")
    print("\nSpecifically, for RHS = ∇·(ρg), use:")
    print("  RHS[i,j] = (ρg_x[i,j+1] - ρg_x[i,j-1])/(2*dx)")
    print("           + (ρg_y[i+1,j] - ρg_y[i-1,j])/(2*dx)")
    print("\nThis is what we're already doing! The issue is elsewhere...")
    
    print("\n\n4. THE REAL PROBLEM:")
    print("-" * 40)
    print("The issue isn't the choice of discretization per se.")
    print("The problem is that for a DISCONTINUOUS density field:")
    print("  - ∇ρ has delta functions at interfaces")
    print("  - Standard finite differences can't represent this well")
    print("  - We get large errors near material interfaces")
    print("\nFor smooth density variations, our current approach works fine!")
    
    print("\n\n5. RECOMMENDED SOLUTION:")
    print("-" * 40)
    print("Option A: Smoothed density field")
    print("  - Apply small smoothing to density before computing ∇·(ρg)")
    print("  - Removes delta functions at interfaces")
    print("  - Maintains fast multigrid solver")
    print("\nOption B: Finite Volume formulation")
    print("  - Interpret cells as control volumes")
    print("  - Compute fluxes at faces using averaged densities")
    print("  - More accurate at discontinuities")
    print("\nOption C: Direct pressure integration (1D only)")
    print("  - Only works for purely vertical gravity")
    print("  - Not suitable for planetary gravity")


def test_smoothing_approach():
    """Test the effect of density smoothing on hydrostatic equilibrium."""
    print("\n\nTESTING DENSITY SMOOTHING APPROACH")
    print("=" * 60)
    
    # Setup
    ny, nx = 30, 10
    dx = 50.0
    g = 9.81
    
    # Sharp density transition
    rho_sharp = np.ones((ny, nx)) * 1.2  # air
    rho_sharp[10:, :] = 1000.0  # water
    
    # Smoothed density (Gaussian filter)
    from scipy.ndimage import gaussian_filter
    rho_smooth = gaussian_filter(rho_sharp, sigma=0.5)
    
    # Compare gradients
    grad_rho_sharp = np.zeros_like(rho_sharp)
    grad_rho_smooth = np.zeros_like(rho_smooth)
    
    grad_rho_sharp[1:-1, :] = (rho_sharp[2:, :] - rho_sharp[:-2, :]) / (2 * dx)
    grad_rho_smooth[1:-1, :] = (rho_smooth[2:, :] - rho_smooth[:-2, :]) / (2 * dx)
    
    print("\nDensity gradient at interface (x=5):")
    print("  y  | ρ_sharp | ∇ρ_sharp | ρ_smooth | ∇ρ_smooth")
    print("-" * 50)
    for y in range(8, 13):
        print(f"{y:3d} | {rho_sharp[y,5]:8.1f} | {grad_rho_sharp[y,5]:9.1f} |"
              f" {rho_smooth[y,5]:8.1f} | {grad_rho_smooth[y,5]:10.1f}")
    
    print("\nKey observation:")
    print("- Sharp: Huge gradient spike at y=10")
    print("- Smooth: Gradient spread over several cells")
    print("- This explains why pressure solver struggles with sharp interfaces!")


if __name__ == "__main__":
    analyze_discretizations()
    test_smoothing_approach()