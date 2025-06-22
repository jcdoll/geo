"""Fluid dynamics with correct hydrostatic equilibrium.

This version modifies the pressure equation to ensure hydrostatic equilibrium.
The key insight: for hydrostatic equilibrium, we need:

1. Pressure satisfies: ∇P = ρg (force balance)
2. Taking divergence: ∇²P = ∇·(ρg)

But we must ensure consistency between:
- How we compute ∇·(ρg) for the RHS
- How we compute ∇P for forces
- The Laplacian ∇² used in the solver
"""

import numpy as np
from typing import Tuple
from materials import MaterialType
from pressure_solver import solve_pressure

try:
    from .core_state import CoreState
except ImportError:
    from core_state import CoreState


class FluidDynamicsHydrostatic:
    """Fluid dynamics with corrected pressure for hydrostatic equilibrium."""
    
    def __init__(self, sim: CoreState):
        self.sim = sim
        self.velocity_x = np.zeros_like(sim.temperature)
        self.velocity_y = np.zeros_like(sim.temperature)
        self.displacement_x = np.zeros_like(sim.temperature)
        self.displacement_y = np.zeros_like(sim.temperature)
        
    def compute_hydrostatic_pressure(self) -> np.ndarray:
        """Compute pressure for hydrostatic equilibrium.
        
        For consistency with the 5-point Laplacian in the multigrid solver,
        we need to use the same discretization for the RHS.
        """
        rho = self.sim.density
        g_y = self.sim.gravity_y
        dx = self.sim.cell_size
        
        # For hydrostatic equilibrium: ∇²P = ∇·(ρg)
        # With our 5-point Laplacian, we need:
        # (P[i+1,j] + P[i-1,j] + P[i,j+1] + P[i,j-1] - 4*P[i,j])/dx² = RHS
        
        # For ρg pointing downward (positive y), we have:
        # ∇·(ρg) = ∂(ρg_y)/∂y
        
        # Using centered differences to match the solver:
        rhs = np.zeros_like(rho)
        rho_g_y = rho * g_y
        
        # Centered difference for divergence
        rhs[1:-1, :] = (rho_g_y[2:, :] - rho_g_y[:-2, :]) / (2 * dx)
        
        # Boundary conditions for RHS
        # For Neumann BC (∂P/∂n = 0), we need special handling
        # Top boundary: use one-sided difference
        rhs[0, :] = (rho_g_y[1, :] - rho_g_y[0, :]) / dx
        # Bottom boundary: use one-sided difference  
        rhs[-1, :] = (rho_g_y[-1, :] - rho_g_y[-2, :]) / dx
        
        # Solve for pressure
        pressure = solve_pressure(rhs, dx, bc_type='neumann', tol=1e-6)
        
        return pressure
    
    def compute_forces_hydrostatic(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces with hydrostatic pressure."""
        # Get density and gravity
        rho = self.sim.density
        gx_total = self.sim.gravity_x
        gy_total = self.sim.gravity_y
        
        # Gravity force density (N/m³)
        fx = rho * gx_total
        fy = rho * gy_total
        
        # Compute hydrostatic pressure
        pressure = self.compute_hydrostatic_pressure()
        self.sim.pressure = pressure
        
        # Pressure gradient using same centered differences as the solver
        dx = self.sim.cell_size
        fx_pressure = np.zeros_like(fx)
        fy_pressure = np.zeros_like(fy)
        
        # Centered differences for interior
        fx_pressure[1:-1, 1:-1] = -(pressure[1:-1, 2:] - pressure[1:-1, :-2]) / (2 * dx)
        fy_pressure[1:-1, 1:-1] = -(pressure[2:, 1:-1] - pressure[:-2, 1:-1]) / (2 * dx)
        
        # One-sided differences for boundaries
        fx_pressure[:, 0] = -(pressure[:, 1] - pressure[:, 0]) / dx
        fx_pressure[:, -1] = -(pressure[:, -1] - pressure[:, -2]) / dx
        fy_pressure[0, :] = -(pressure[1, :] - pressure[0, :]) / dx
        fy_pressure[-1, :] = -(pressure[-1, :] - pressure[-2, :]) / dx
        
        # Total forces
        fx += fx_pressure
        fy += fy_pressure
        
        # Store forces
        self.sim.force_x = fx
        self.sim.force_y = fy
        
        return fx, fy
    
    def update_velocities_with_viscosity(self, fx: np.ndarray, fy: np.ndarray, 
                                         rho: np.ndarray, dt: float):
        """Update velocities with forces and material viscosity damping."""
        # Calculate accelerations
        accel_x = np.where(rho > 0, fx / rho, 0.0)
        accel_y = np.where(rho > 0, fy / rho, 0.0)

        # Update velocities
        self.velocity_x += accel_x * dt
        self.velocity_y += accel_y * dt

        # Apply viscosity damping
        from materials import MaterialDatabase
        mat_db = MaterialDatabase()
        viscosity = mat_db.get_array_property("viscosity", self.sim.material_types)
        
        # Viscosity damping factor
        damping = 1.0 - viscosity
        self.velocity_x *= damping
        self.velocity_y *= damping

        # Zero out small velocities
        min_vel = 1e-8
        self.velocity_x[np.abs(self.velocity_x) < min_vel] = 0.0
        self.velocity_y[np.abs(self.velocity_y) < min_vel] = 0.0

    def step_forward(self, dt: float):
        """Main fluid dynamics step."""
        # Compute forces with hydrostatic pressure
        fx, fy = self.compute_forces_hydrostatic()
        
        # Update velocities
        self.update_velocities_with_viscosity(fx, fy, self.sim.density, dt)
        
        # Apply movement (simplified for testing)
        self.apply_simple_movement(dt)
        
    def apply_simple_movement(self, dt: float):
        """Simplified movement for testing."""
        # Just update positions based on velocity
        # Real implementation would handle material swapping
        pass


def test_hydrostatic_equilibrium():
    """Test that water achieves hydrostatic equilibrium."""
    from core_state import CoreState
    
    # Create test grid
    sim = CoreState(width=10, height=30, cell_size=50.0)
    
    # Setup: air above, water below
    sim.material_types[:10, :] = MaterialType.AIR
    sim.material_types[10:, :] = MaterialType.WATER
    sim._update_material_properties()
    
    # Initialize gravity
    sim.gravity_x = np.zeros_like(sim.temperature)
    sim.gravity_y = np.full_like(sim.temperature, 9.81)
    
    # Create fluid dynamics
    fluid = FluidDynamicsHydrostatic(sim)
    
    # Compute forces
    fx, fy = fluid.compute_forces_hydrostatic()
    
    print("Hydrostatic Equilibrium Test")
    print("-" * 60)
    print("Location | Material | ρg (N/m³) | -∇P (N/m³) | Net Force")
    print("-" * 60)
    
    x = 5
    for y in [5, 9, 10, 11, 15, 20, 25]:
        density = sim.density[y, x]
        gravity_force = density * sim.gravity_y[y, x]
        pressure_grad = fy[y, x] - gravity_force  # Just the pressure contribution
        net = fy[y, x]
        mat = "AIR" if y < 10 else "WATER"
        print(f"y={y:2d} ({mat:5s}) | {density:8.1f} | {gravity_force:10.1f} | {pressure_grad:11.1f} | {net:10.6f}")
    
    # Check maximum net force
    max_net = np.max(np.abs(fy))
    avg_net = np.mean(np.abs(fy))
    
    print(f"\nMax |net force|: {max_net:.2f} N/m³")
    print(f"Avg |net force|: {avg_net:.2f} N/m³") 
    print(f"Reference (ρ_water * g): {1000 * 9.81:.1f} N/m³")
    
    # The net force should be small compared to ρg
    relative_error = max_net / (1000 * 9.81)
    print(f"Relative error: {relative_error:.2%}")
    print(f"Success: {'YES' if relative_error < 0.01 else 'NO'}")
    
    # Also check pressure gradient directly
    print("\nPressure gradient check:")
    P = sim.pressure
    dx = sim.cell_size
    
    for y in [11, 15, 20]:
        dP_dy = (P[y+1, x] - P[y-1, x]) / (2 * dx)
        rho_g = sim.density[y, x] * sim.gravity_y[y, x]
        error = abs(dP_dy - rho_g) / rho_g
        print(f"y={y}: dP/dy = {dP_dy:.1f}, ρg = {rho_g:.1f}, error = {error:.2%}")


if __name__ == "__main__":
    test_hydrostatic_equilibrium()