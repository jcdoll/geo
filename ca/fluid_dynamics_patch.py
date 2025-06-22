"""Patch for fluid_dynamics.py to add hydrostatic pressure option.

This module provides a modified compute_forces method that can use
either the original Poisson solver or direct hydrostatic integration.
"""

import numpy as np
from pressure_hydrostatic import compute_hydrostatic_pressure
from pressure_solver import solve_pressure


def compute_forces_with_hydrostatic(self, use_hydrostatic=True):
    """Modified compute_forces that optionally uses direct hydrostatic pressure.
    
    Args:
        use_hydrostatic: If True, use direct integration for pressure.
                        If False, use original Poisson solver.
    
    This method can be monkey-patched onto FluidDynamics instances.
    """
    # Validate gravity field
    if not hasattr(self.sim, 'gravity_x') or self.sim.gravity_x is None:
        raise RuntimeError("[fluid_dynamics] gravity field not initialized")
    
    # Grab references to gravity field
    gx_total = self.sim.gravity_x
    gy_total = self.sim.gravity_y
    rho = self.sim.density
    
    # Gravity force density (N/m³)
    fx = rho * gx_total
    fy = rho * gy_total
    
    if use_hydrostatic:
        # Direct hydrostatic pressure calculation
        P_pa = compute_hydrostatic_pressure(rho, gy_total, self.sim.cell_size)
        self.sim.pressure = P_pa
    else:
        # Original Poisson solver approach
        # Build RHS for pressure equation: ∇²P = g·∇ρ
        dx = self.sim.cell_size
        grad_rho_x = np.zeros_like(rho)
        grad_rho_y = np.zeros_like(rho)
        
        # Central differences for density gradient
        grad_rho_x[1:-1, 1:-1] = (rho[1:-1, 2:] - rho[1:-1, :-2]) / (2 * dx)
        grad_rho_y[1:-1, 1:-1] = (rho[2:, 1:-1] - rho[:-2, 1:-1]) / (2 * dx)
        
        # Handle boundaries with one-sided differences
        grad_rho_x[:, 0] = (rho[:, 1] - rho[:, 0]) / dx
        grad_rho_x[:, -1] = (rho[:, -1] - rho[:, -2]) / dx
        grad_rho_y[0, :] = (rho[1, :] - rho[0, :]) / dx
        grad_rho_y[-1, :] = (rho[-1, :] - rho[-2, :]) / dx
        
        # Pressure equation RHS: g·∇ρ
        rhs = gx_total * grad_rho_x + gy_total * grad_rho_y
        
        # Solve for pressure
        P_pa = solve_pressure(rhs, dx, bc_type='neumann')
        self.sim.pressure = P_pa
    
    # Pressure gradient force (same for both methods)
    dx = self.sim.cell_size
    fx_pressure = np.zeros_like(fx)
    fy_pressure = np.zeros_like(fy)
    
    # Centered differences for pressure gradients
    fx_pressure[1:-1, 1:-1] = -(P_pa[1:-1, 2:] - P_pa[1:-1, :-2]) / (2 * dx)
    fy_pressure[1:-1, 1:-1] = -(P_pa[2:, 1:-1] - P_pa[:-2, 1:-1]) / (2 * dx)
    
    # Boundaries
    fx_pressure[:, 0] = -(P_pa[:, 1] - P_pa[:, 0]) / dx
    fx_pressure[:, -1] = -(P_pa[:, -1] - P_pa[:, -2]) / dx
    fy_pressure[0, :] = -(P_pa[1, :] - P_pa[0, :]) / dx
    fy_pressure[-1, :] = -(P_pa[-1, :] - P_pa[-2, :]) / dx
    
    # Add pressure forces
    fx += fx_pressure
    fy += fy_pressure
    
    # Store forces
    self.sim.force_x = fx
    self.sim.force_y = fy
    return fx, fy


def apply_hydrostatic_patch(fluid_dynamics_instance):
    """Apply the hydrostatic pressure patch to a FluidDynamics instance.
    
    Usage:
        from fluid_dynamics_patch import apply_hydrostatic_patch
        apply_hydrostatic_patch(my_fluid_dynamics)
        # Now compute_forces will use hydrostatic pressure by default
    """
    import types
    fluid_dynamics_instance.compute_forces = types.MethodType(
        compute_forces_with_hydrostatic, 
        fluid_dynamics_instance
    )
    print("Hydrostatic pressure patch applied to FluidDynamics instance")


def test_patch():
    """Test the patched compute_forces method."""
    from core_state import CoreState
    from fluid_dynamics import FluidDynamics
    from materials import MaterialType
    
    # Create test simulation
    sim = CoreState(width=10, height=30, cell_size=50.0)
    
    # Setup water column
    sim.material_types[:10, :] = MaterialType.AIR
    sim.material_types[10:, :] = MaterialType.WATER
    sim._update_material_properties()
    
    # Initialize gravity
    sim.gravity_x = np.zeros_like(sim.temperature)
    sim.gravity_y = np.full_like(sim.temperature, 9.81)
    
    # Create fluid dynamics and apply patch
    fluid = FluidDynamics(sim)
    apply_hydrostatic_patch(fluid)
    
    # Test with hydrostatic pressure
    print("Testing with hydrostatic pressure:")
    fx_hydro, fy_hydro = fluid.compute_forces(use_hydrostatic=True)
    
    # Check forces at different depths
    x = 5
    print("\n   y | Material | Net Force (N/m³)")
    print("-" * 35)
    for y in [5, 10, 15, 20, 25]:
        mat = "AIR" if y < 10 else "WATER"
        print(f"{y:4d} | {mat:8s} | {fy_hydro[y, x]:15.6f}")
    
    max_force = np.max(np.abs(fy_hydro))
    print(f"\nMax |force| with hydrostatic: {max_force:.3f} N/m³")
    
    # Compare with original method
    print("\nTesting with original Poisson solver:")
    fx_orig, fy_orig = fluid.compute_forces(use_hydrostatic=False)
    
    print("\n   y | Material | Net Force (N/m³)")
    print("-" * 35)
    for y in [5, 10, 15, 20, 25]:
        mat = "AIR" if y < 10 else "WATER"
        print(f"{y:4d} | {mat:8s} | {fy_orig[y, x]:15.6f}")
    
    max_force_orig = np.max(np.abs(fy_orig))
    print(f"\nMax |force| with Poisson: {max_force_orig:.3f} N/m³")
    
    # Show improvement
    print(f"\nImprovement factor: {max_force_orig / max_force:.1f}x")


if __name__ == "__main__":
    test_patch()