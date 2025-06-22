"""Correct pressure calculation using Poisson solver with consistent discretization.

The key fix: ensure the RHS computation matches what the multigrid solver expects,
and use consistent gradient operators.
"""

import numpy as np
from pressure_solver import solve_pressure


def calculate_pressure_poisson(sim):
    """Calculate pressure using Poisson equation with correct formulation.
    
    For hydrostatic equilibrium: ∇P = ρg
    Taking divergence: ∇²P = ∇·(ρg)
    
    The key is computing ∇·(ρg) correctly for the 5-point Laplacian.
    """
    # Get total gravity field
    gx_total = np.zeros_like(sim.density)
    gy_total = np.zeros_like(sim.density)
    
    # Add self-gravity component
    if sim.enable_self_gravity and hasattr(sim, 'calculate_self_gravity'):
        sim.calculate_self_gravity()
        gx_total += sim.gravity_x
        gy_total += sim.gravity_y
        
    # Add external gravity
    if hasattr(sim, 'external_gravity'):
        g_ext_x, g_ext_y = sim.external_gravity
        if g_ext_x != 0.0 or g_ext_y != 0.0:
            gx_total += g_ext_x
            gy_total += g_ext_y
    
    # Build RHS for pressure equation
    dx = sim.cell_size
    rho = sim.density
    
    # Method 1: Standard approach (what was likely used before)
    # RHS = g·∇ρ
    # This gives non-zero values only at material interfaces
    grad_rho_x = np.zeros_like(rho)
    grad_rho_y = np.zeros_like(rho)
    
    # Centered differences for density gradient
    grad_rho_x[1:-1, 1:-1] = (rho[1:-1, 2:] - rho[1:-1, :-2]) / (2 * dx)
    grad_rho_y[1:-1, 1:-1] = (rho[2:, 1:-1] - rho[:-2, 1:-1]) / (2 * dx)
    
    # Boundaries
    grad_rho_x[:, 0] = (rho[:, 1] - rho[:, 0]) / dx
    grad_rho_x[:, -1] = (rho[:, -1] - rho[:, -2]) / dx
    grad_rho_y[0, :] = (rho[1, :] - rho[0, :]) / dx
    grad_rho_y[-1, :] = (rho[-1, :] - rho[-2, :]) / dx
    
    # Standard RHS
    rhs_standard = gx_total * grad_rho_x + gy_total * grad_rho_y
    
    # Method 2: Full divergence (more accurate but still has issues)
    # RHS = ∇·(ρg) = ∂(ρgx)/∂x + ∂(ρgy)/∂y
    rho_gx = rho * gx_total
    rho_gy = rho * gy_total
    
    rhs_div = np.zeros_like(rho)
    
    # Interior points
    rhs_div[1:-1, 1:-1] = ((rho_gx[1:-1, 2:] - rho_gx[1:-1, :-2]) / (2 * dx) +
                           (rho_gy[2:, 1:-1] - rho_gy[:-2, 1:-1]) / (2 * dx))
    
    # Boundaries (one-sided differences)
    # Top
    rhs_div[0, 1:-1] = ((rho_gx[0, 2:] - rho_gx[0, :-2]) / (2 * dx) +
                        (rho_gy[1, 1:-1] - rho_gy[0, 1:-1]) / dx)
    # Bottom
    rhs_div[-1, 1:-1] = ((rho_gx[-1, 2:] - rho_gx[-1, :-2]) / (2 * dx) +
                         (rho_gy[-1, 1:-1] - rho_gy[-2, 1:-1]) / dx)
    # Left
    rhs_div[1:-1, 0] = ((rho_gx[1:-1, 1] - rho_gx[1:-1, 0]) / dx +
                        (rho_gy[2:, 0] - rho_gy[:-2, 0]) / (2 * dx))
    # Right
    rhs_div[1:-1, -1] = ((rho_gx[1:-1, -1] - rho_gx[1:-1, -2]) / dx +
                         (rho_gy[2:, -1] - rho_gy[:-2, -1]) / (2 * dx))
    
    # Use the full divergence RHS
    rhs = rhs_div
    
    # Solve for pressure
    pressure = solve_pressure(rhs, dx, bc_type='neumann', tol=1e-6)
    
    # Add reference pressure
    p_ref = 101325.0  # 1 atmosphere
    pressure += p_ref
    
    return pressure


def compute_forces_consistent(sim, pressure):
    """Compute forces with consistent gradient operator.
    
    Uses the same discretization as the Poisson solver.
    """
    rho = sim.density
    gx_total = sim.gravity_x
    gy_total = sim.gravity_y
    dx = sim.cell_size
    
    # Gravity forces
    fx = rho * gx_total
    fy = rho * gy_total
    
    # Pressure gradient (centered differences matching the solver)
    fx_pressure = np.zeros_like(fx)
    fy_pressure = np.zeros_like(fy)
    
    # Interior points
    fx_pressure[1:-1, 1:-1] = -(pressure[1:-1, 2:] - pressure[1:-1, :-2]) / (2 * dx)
    fy_pressure[1:-1, 1:-1] = -(pressure[2:, 1:-1] - pressure[:-2, 1:-1]) / (2 * dx)
    
    # Boundaries (one-sided)
    fx_pressure[:, 0] = -(pressure[:, 1] - pressure[:, 0]) / dx
    fx_pressure[:, -1] = -(pressure[:, -1] - pressure[:, -2]) / dx
    fy_pressure[0, :] = -(pressure[1, :] - pressure[0, :]) / dx
    fy_pressure[-1, :] = -(pressure[-1, :] - pressure[-2, :]) / dx
    
    # Total forces
    fx_total = fx + fx_pressure
    fy_total = fy + fy_pressure
    
    return fx_total, fy_total


def test_poisson_approach():
    """Test the Poisson solver approach."""
    from core_state import CoreState
    from materials import MaterialType
    
    # Create test setup
    sim = CoreState(width=10, height=30, cell_size=50.0)
    
    # Water column with air above
    sim.material_types[:10, :] = MaterialType.AIR
    sim.material_types[10:, :] = MaterialType.WATER
    sim._update_material_properties()
    
    # Simple uniform gravity
    sim.gravity_x = np.zeros_like(sim.temperature)
    sim.gravity_y = np.full_like(sim.temperature, 9.81)
    sim.enable_self_gravity = False
    
    # Calculate pressure
    pressure = calculate_pressure_poisson(sim)
    sim.pressure = pressure
    
    # Compute forces
    fx, fy = compute_forces_consistent(sim, pressure)
    
    print("Poisson Solver Approach Test")
    print("=" * 60)
    print("Location | Material | ρg (N/m³) | Net Force | Ratio")
    print("-" * 60)
    
    x = 5
    for y in [5, 9, 10, 11, 15, 20, 25]:
        rho_g = sim.density[y, x] * sim.gravity_y[y, x]
        net = fy[y, x]
        ratio = net / rho_g if rho_g > 0 else 0
        mat = "AIR" if y < 10 else "WATER"
        print(f"y={y:2d} ({mat:5s}) | {rho_g:10.1f} | {net:10.3f} | {ratio:6.3f}")
    
    # Check equilibrium
    air_max = np.max(np.abs(fy[3:8, :]))
    water_max = np.max(np.abs(fy[15:25, :]))
    
    print(f"\nBulk air (y=3-8): max |force| = {air_max:.1f} N/m³")
    print(f"Bulk water (y=15-25): max |force| = {water_max:.1f} N/m³")
    
    # The fundamental issue
    print("\nFUNDAMENTAL ISSUE:")
    print("In bulk regions where ∇ρ = 0, the RHS ≈ 0")
    print("So ∇²P ≈ 0, making P harmonic")
    print("This doesn't constrain ∇P = ρg!")
    print("\nThe problem requires a different formulation.")


if __name__ == "__main__":
    test_poisson_approach()