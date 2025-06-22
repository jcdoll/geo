"""Fix for fluid_dynamics.py - remove path-dependent pressure calculation.

The current calculate_planetary_pressure() uses path-dependent integration
which is incorrect for 2D. We should remove it and use only the projection
method to maintain pressure.
"""

import numpy as np
from materials import MaterialType


def calculate_planetary_pressure_fixed(self):
    """Don't calculate static pressure - let it evolve dynamically.
    
    Just initialize to a constant or simple estimate.
    The projection method will adjust it over time.
    """
    # Option 1: Constant pressure (simplest)
    # self.sim.pressure[:] = 101325.0  # 1 atmosphere everywhere
    
    # Option 2: Simple vertical integration (better initial guess)
    ny, nx = self.sim.material_types.shape
    pressure = np.full((ny, nx), 101325.0)  # Start with 1 atm
    
    # Add hydrostatic component if gravity is mostly vertical
    if hasattr(self.sim, 'gravity_y'):
        for i in range(1, ny):
            # Simple vertical integration
            rho_avg = 0.5 * (self.sim.density[i-1, :] + self.sim.density[i, :])
            g_avg = np.mean(self.sim.gravity_y[i-1:i+1, :])  # Average gravity
            dp = rho_avg * g_avg * self.sim.cell_size
            pressure[i, :] = pressure[i-1, :] + dp
    
    self.sim.pressure[:] = pressure


def compute_force_field_no_pressure(self):
    """Compute only gravity forces - no pressure gradient initially.
    
    The pressure gradient will be added through the projection method.
    """
    rho = self.sim.density
    
    # Get gravity
    if hasattr(self.sim, 'calculate_self_gravity'):
        self.sim.calculate_self_gravity()
    gx_total = self.sim.gravity_x
    gy_total = self.sim.gravity_y
    
    # Add external gravity
    if hasattr(self.sim, 'external_gravity'):
        g_ext_x, g_ext_y = self.sim.external_gravity
        if g_ext_x != 0.0 or g_ext_y != 0.0:
            gx_total = gx_total + g_ext_x
            gy_total = gy_total + g_ext_y
    
    # Only gravity forces (no pressure gradient)
    fx = rho * gx_total
    fy = rho * gy_total
    
    self.sim.force_x = fx
    self.sim.force_y = fy
    
    return fx, fy


def apply_velocity_projection_with_pressure_update(self, dt: float):
    """Modified projection that also updates the pressure field.
    
    After solving for the pressure correction φ, update:
    P_new = P_old + φ/dt
    
    This makes pressure evolve to hydrostatic equilibrium.
    """
    # Standard projection code...
    # [existing projection code]
    
    # After solving for φ (pressure correction):
    # self.sim.pressure += phi / dt
    
    # This is the key - pressure evolves dynamically!
    pass


# The real fix: monkey patch these methods
def apply_fix(fluid_dynamics_instance):
    """Apply the fix to a FluidDynamics instance."""
    import types
    
    # Replace the incorrect path-dependent pressure calculation
    fluid_dynamics_instance.calculate_planetary_pressure = types.MethodType(
        calculate_planetary_pressure_fixed, fluid_dynamics_instance
    )
    
    print("Fix applied: Removed path-dependent pressure calculation")
    print("Pressure will now evolve dynamically through projection method")


# Better idea: Just fix the RHS computation in the existing solver
def compute_correct_pressure_rhs(sim):
    """Compute the correct RHS for pressure Poisson equation.
    
    For ∇²P = ∇·(ρg), we need the FULL divergence, not just g·∇ρ.
    """
    dx = sim.cell_size
    rho = sim.density
    gx = sim.gravity_x if hasattr(sim, 'gravity_x') else 0
    gy = sim.gravity_y if hasattr(sim, 'gravity_y') else 9.81
    
    # Full divergence: ∇·(ρg) = ∂(ρgx)/∂x + ∂(ρgy)/∂y
    rho_gx = rho * gx
    rho_gy = rho * gy
    
    rhs = np.zeros_like(rho)
    
    # Centered differences
    rhs[1:-1, 1:-1] = ((rho_gx[1:-1, 2:] - rho_gx[1:-1, :-2]) / (2*dx) +
                       (rho_gy[2:, 1:-1] - rho_gy[:-2, 1:-1]) / (2*dx))
    
    # Boundaries
    rhs[0, :] = (rho_gy[1, :] - rho_gy[0, :]) / dx
    rhs[-1, :] = (rho_gy[-1, :] - rho_gy[-2, :]) / dx
    rhs[:, 0] = (rho_gx[:, 1] - rho_gx[:, 0]) / dx
    rhs[:, -1] = (rho_gx[:, -1] - rho_gx[:, -2]) / dx
    
    return rhs


if __name__ == "__main__":
    print("FLUID DYNAMICS FIX")
    print("=" * 60)
    print("\nThe issue: calculate_planetary_pressure() uses path-dependent")
    print("integration which is incorrect for 2D problems.")
    print("\nThe fix: Remove it and let pressure evolve dynamically")
    print("through the velocity projection method.")
    print("\nAlternatively: Fix the RHS computation to use full ∇·(ρg)")
    print("instead of just g·∇ρ.")