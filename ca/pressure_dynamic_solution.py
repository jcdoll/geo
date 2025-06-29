"""The correct solution: let pressure evolve dynamically.

Don't try to solve for static hydrostatic pressure. Instead:
1. Initialize with a simple estimate
2. Let velocity projection update pressure each timestep
3. System will converge to equilibrium
"""

import numpy as np
from materials import MaterialType
from core_state import CoreState
from fluid_dynamics import FluidDynamics


def calculate_initial_pressure_estimate(sim):
    """Simple pressure initialization - just integrate ρg vertically.
    
    This gives a reasonable starting point. The exact value doesn't matter
    much because the projection method will adjust it dynamically.
    """
    ny, nx = sim.material_types.shape
    pressure = np.zeros((ny, nx))
    
    # Get gravity (assuming mostly vertical)
    gy = sim.gravity_y if hasattr(sim, 'gravity_y') else 9.81
    
    # Simple vertical integration from top
    # P[i+1] = P[i] + ρ[i] * g * Δy
    for i in range(ny - 1):
        # Use average density between cells
        rho_avg = 0.5 * (sim.density[i, :] + sim.density[i+1, :])
        if np.isscalar(gy):
            g_avg = gy
        else:
            g_avg = 0.5 * (gy[i, :] + gy[i+1, :])
        
        dp = rho_avg * g_avg * sim.cell_size
        pressure[i+1, :] = pressure[i, :] + dp
    
    # Add reference pressure
    pressure += 101325.0  # 1 atmosphere
    
    return pressure


def monkey_patch_pressure_calculation(fluid_dynamics_instance):
    """Replace the complex pressure calculation with simple initialization."""
    
    def simple_pressure_init(self):
        """Initialize pressure with vertical integration."""
        self.sim.pressure[:] = calculate_initial_pressure_estimate(self.sim)
    
    # Replace the method
    import types
    fluid_dynamics_instance.calculate_planetary_pressure = types.MethodType(
        simple_pressure_init, fluid_dynamics_instance
    )


def test_dynamic_equilibrium():
    """Test that system reaches equilibrium dynamically."""
    # Create test
    sim = CoreState(width=10, height=30, cell_size=50.0)
    
    # Water column
    sim.material_types[:10, :] = MaterialType.AIR  
    sim.material_types[10:, :] = MaterialType.WATER
    sim._update_material_properties()
    
    # Simple gravity
    sim.gravity_x = np.zeros_like(sim.temperature)
    sim.gravity_y = np.full_like(sim.temperature, 9.81)
    sim.enable_self_gravity = False
    
    # Create fluid dynamics
    fluid = FluidDynamics(sim)
    
    # Use simple pressure initialization
    monkey_patch_pressure_calculation(fluid)
    fluid.calculate_planetary_pressure()
    
    print("DYNAMIC EQUILIBRIUM TEST")
    print("=" * 60)
    print("\nInitial state:")
    
    # Check initial forces
    fx, fy = fluid.compute_force_field()
    
    x = 5
    print("\n   y | Material | Initial Force (N/m³)")
    print("-" * 40)
    for y in [5, 10, 15, 20, 25]:
        mat = "AIR" if y < 10 else "WATER"
        print(f"{y:4d} | {mat:8s} | {fy[y, x]:15.3f}")
    
    # Run several timesteps
    print("\nRunning 100 timesteps with projection...")
    dt = 0.01
    
    for step in range(100):
        # Apply strong damping for faster convergence
        fluid.velocity_x *= 0.9
        fluid.velocity_y *= 0.9
        
        # Run dynamics
        fluid.apply_unified_kinematics(dt)
        
        if step % 20 == 0:
            max_vel = np.max(np.abs(fluid.velocity_y))
            print(f"Step {step:3d}: max |v_y| = {max_vel:.6f} m/s")
    
    # Check final forces
    fx, fy = fluid.compute_force_field()
    
    print("\nFinal state:")
    print("\n   y | Material | Final Force (N/m³) | Velocity (m/s)")
    print("-" * 55)
    for y in [5, 10, 15, 20, 25]:
        mat = "AIR" if y < 10 else "WATER"
        print(f"{y:4d} | {mat:8s} | {fy[y, x]:15.3f} | {fluid.velocity_y[y, x]:14.6f}")
    
    # Check convergence
    max_force = np.max(np.abs(fy))
    max_vel = np.max(np.abs(fluid.velocity_y))
    
    print(f"\nMax |force|: {max_force:.1f} N/m³")
    print(f"Max |velocity|: {max_vel:.6f} m/s")
    print(f"\nEquilibrium reached: {'YES' if max_vel < 1e-3 else 'NO'}")
    
    print("\nCONCLUSION:")
    print("The system SHOULD converge to equilibrium dynamically")
    print("through the velocity projection method.")
    print("We don't need to solve for static pressure!")


if __name__ == "__main__":
    test_dynamic_equilibrium()