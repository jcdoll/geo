"""Modified fluid dynamics using staggered grid for correct hydrostatic equilibrium.

This version uses adjoint-consistent operators to ensure water columns remain
stationary under gravity.
"""

import numpy as np
from typing import Tuple, Optional
from materials import MaterialType
from pressure_solver import solve_pressure
from staggered_grid_operators import gradient_staggered, divergence_staggered

try:
    from .core_state import CoreState
except ImportError:
    from core_state import CoreState


class FluidDynamicsStaggered:
    def __init__(self, sim: CoreState):
        self.sim = sim
        self.velocity_x = np.zeros_like(sim.temperature)
        self.velocity_y = np.zeros_like(sim.temperature)
        self.displacement_x = np.zeros_like(sim.temperature)
        self.displacement_y = np.zeros_like(sim.temperature)
        
    def compute_forces_staggered(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces using staggered grid for pressure gradient."""
        # Get density and gravity
        rho = self.sim.density
        gx_total = self.sim.gravity_x
        gy_total = self.sim.gravity_y
        
        # Gravity force density (N/m³) at cell centers
        fx = rho * gx_total
        fy = rho * gy_total
        
        # For hydrostatic equilibrium, we need to solve:
        # ∇²P = ∇·(ρg)
        # Then ∇P will balance ρg exactly
        
        # Build RHS for pressure equation
        # First, we need ρg at faces for divergence
        ny, nx = rho.shape
        dx = self.sim.cell_size
        
        # Interpolate ρg to faces
        rho_g_x_face = np.zeros((ny, nx+1))
        rho_g_y_face = np.zeros((ny+1, nx))
        
        # Average ρg_x to vertical faces
        rho_g_x = rho * gx_total
        rho_g_x_face[:, 1:-1] = 0.5 * (rho_g_x[:, :-1] + rho_g_x[:, 1:])
        # Boundary faces use one-sided values
        rho_g_x_face[:, 0] = rho_g_x[:, 0]
        rho_g_x_face[:, -1] = rho_g_x[:, -1]
        
        # Average ρg_y to horizontal faces  
        rho_g_y = rho * gy_total
        rho_g_y_face[1:-1, :] = 0.5 * (rho_g_y[:-1, :] + rho_g_y[1:, :])
        # Boundary faces
        rho_g_y_face[0, :] = rho_g_y[0, :]
        rho_g_y_face[-1, :] = rho_g_y[-1, :]
        
        # Compute divergence for RHS
        rhs = divergence_staggered(rho_g_x_face, rho_g_y_face, dx)
        
        # Solve for pressure
        pressure = solve_pressure(rhs, dx, bc_type='neumann', tol=1e-6)
        self.sim.pressure = pressure
        
        # Get pressure gradient forces at faces
        grad_p_x, grad_p_y = gradient_staggered(pressure, dx)
        
        # Pressure forces at faces (negative gradient)
        fx_pressure_face = -grad_p_x
        fy_pressure_face = -grad_p_y
        
        # Interpolate pressure forces back to cell centers
        fx_pressure = np.zeros_like(fx)
        fy_pressure = np.zeros_like(fy)
        
        # Average x-forces from vertical faces to centers
        fx_pressure[:, 1:-1] = 0.5 * (fx_pressure_face[:, 1:-2] + fx_pressure_face[:, 2:-1])
        fx_pressure[:, 0] = fx_pressure_face[:, 1]  # Use adjacent face value
        fx_pressure[:, -1] = fx_pressure_face[:, -2]
        
        # Average y-forces from horizontal faces to centers
        fy_pressure[1:-1, :] = 0.5 * (fy_pressure_face[1:-2, :] + fy_pressure_face[2:-1, :])
        fy_pressure[0, :] = fy_pressure_face[1, :]
        fy_pressure[-1, :] = fy_pressure_face[-2, :]
        
        # Total forces: gravity + pressure
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
        viscosity = self.sim.material_properties.get_array_property("viscosity", self.sim.material_types)
        
        # Viscosity damping factor
        damping = 1.0 - viscosity
        self.velocity_x *= damping
        self.velocity_y *= damping

        # Zero out small velocities
        min_vel = 1e-8
        self.velocity_x[np.abs(self.velocity_x) < min_vel] = 0.0
        self.velocity_y[np.abs(self.velocity_y) < min_vel] = 0.0

    def step_forward(self, dt: float):
        """Main fluid dynamics step with staggered grid pressure."""
        # Compute forces with staggered grid
        fx, fy = self.compute_forces_staggered()
        
        # Update velocities
        self.update_velocities_with_viscosity(fx, fy, self.sim.density, dt)
        
        # Apply velocity movement (same as original)
        self.apply_velocity_movement_vectorized(dt)
        
    def apply_velocity_movement_vectorized(self, dt: float):
        """Same movement code as original - this part works fine."""
        # Update displacements
        self.displacement_x += self.velocity_x * dt / self.sim.cell_size
        self.displacement_y += self.velocity_y * dt / self.sim.cell_size

        # Find cells that need to move
        move_mask = (np.abs(self.displacement_x) >= 1.0) | (np.abs(self.displacement_y) >= 1.0)
        
        if not np.any(move_mask):
            return

        # Get movement directions
        move_x = np.where(move_mask, np.sign(self.displacement_x).astype(int), 0)
        move_y = np.where(move_mask, np.sign(self.displacement_y).astype(int), 0)

        # Separate by direction for vectorized processing
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        for dx, dy in directions:
            direction_mask = move_mask & (move_x == dx) & (move_y == dy)
            if not np.any(direction_mask):
                continue
                
            self._process_direction_batch(direction_mask, dx, dy)

        # Update displacements for cells that moved
        self.displacement_x = np.where(move_mask, 
                                       self.displacement_x - move_x,
                                       self.displacement_x)
        self.displacement_y = np.where(move_mask,
                                       self.displacement_y - move_y, 
                                       self.displacement_y)

    def _process_direction_batch(self, mask, dx, dy):
        """Process all cells moving in the same direction."""
        ny, nx = mask.shape
        
        # Valid source and target ranges
        src_y_min = max(0, -dy)
        src_y_max = min(ny, ny - dy)
        src_x_min = max(0, -dx)
        src_x_max = min(nx, nx - dx)
        
        tgt_y_min = max(0, dy)
        tgt_y_max = min(ny, ny + dy)
        tgt_x_min = max(0, dx)
        tgt_x_max = min(nx, nx + dx)
        
        # Get slices
        src_slice = (slice(src_y_min, src_y_max), slice(src_x_min, src_x_max))
        tgt_slice = (slice(tgt_y_min, tgt_y_max), slice(tgt_x_min, tgt_x_max))
        
        # Movement mask for valid positions
        valid_mask = np.zeros_like(mask)
        valid_mask[src_slice] = mask[src_slice]
        
        # Check target availability
        can_move = valid_mask[src_slice] & (self.sim.material_types[tgt_slice] == MaterialType.AIR)
        
        # Perform swaps where possible
        if np.any(can_move):
            # Create temporary copies for materials
            temp_mat = self.sim.material_types.copy()
            temp_mat[tgt_slice] = np.where(can_move, 
                                           self.sim.material_types[src_slice],
                                           temp_mat[tgt_slice])
            temp_mat[src_slice] = np.where(can_move, MaterialType.AIR, temp_mat[src_slice])
            self.sim.material_types[:] = temp_mat
            
            # Swap temperatures
            temp_T = self.sim.temperature.copy()
            temp_T[tgt_slice] = np.where(can_move,
                                         self.sim.temperature[src_slice], 
                                         temp_T[tgt_slice])
            temp_T[src_slice] = np.where(can_move, 293.15, temp_T[src_slice])
            self.sim.temperature[:] = temp_T
            
            # Move velocities
            temp_vx = self.velocity_x.copy()
            temp_vx[tgt_slice] = np.where(can_move,
                                          self.velocity_x[src_slice],
                                          temp_vx[tgt_slice])
            temp_vx[src_slice] = np.where(can_move, 0.0, temp_vx[src_slice])
            self.velocity_x[:] = temp_vx
            
            temp_vy = self.velocity_y.copy()
            temp_vy[tgt_slice] = np.where(can_move,
                                          self.velocity_y[src_slice],
                                          temp_vy[tgt_slice])
            temp_vy[src_slice] = np.where(can_move, 0.0, temp_vy[src_slice])
            self.velocity_y[:] = temp_vy
            
            # Update properties after movement
            self.sim._update_material_properties()


def test_hydrostatic_equilibrium():
    """Test that a water column remains stationary."""
    # Create small test grid
    sim = CoreState(width=10, height=30, cell_size=50.0)
    
    # Setup: air above, water below
    sim.material_types[:10, :] = MaterialType.AIR
    sim.material_types[10:, :] = MaterialType.WATER
    sim._update_material_properties()
    
    # Initialize gravity (simple uniform downward gravity)
    sim.gravity_x = np.zeros_like(sim.temperature)
    sim.gravity_y = np.full_like(sim.temperature, 9.81)  # m/s²
    
    # Create fluid dynamics with staggered grid
    fluid = FluidDynamicsStaggered(sim)
    
    # Compute initial forces
    fx, fy = fluid.compute_forces_staggered()
    
    print("Hydrostatic Equilibrium Test (Staggered Grid)")
    print("-" * 60)
    print("Location | Density | Gravity | Pressure | Net Force")
    print("-" * 60)
    
    x = 5
    for y in [5, 10, 15, 20, 25]:
        density = sim.density[y, x]
        gravity = sim.gravity_y[y, x] * density
        pressure_force = fy[y, x] - gravity  # Net pressure force
        net = fy[y, x]
        mat = "AIR" if y < 10 else "WATER"
        print(f"y={y:2d} ({mat:5s}) | {density:7.1f} | {gravity:8.1f} | {pressure_force:9.1f} | {net:10.6f}")
    
    # Check maximum net force
    max_net = np.max(np.abs(fy))
    print(f"\nMax net force: {max_net:.6f} N/m³")
    print(f"Success: {'YES' if max_net < 1.0 else 'NO'}")
    
    # Run a few steps to verify stability
    print("\nRunning 10 timesteps...")
    initial_vy = fluid.velocity_y.copy()
    for i in range(10):
        fluid.step_forward(0.01)
    
    max_vel_change = np.max(np.abs(fluid.velocity_y - initial_vy))
    print(f"Max velocity change: {max_vel_change:.6e} m/s")
    print(f"Stable: {'YES' if max_vel_change < 1e-6 else 'NO'}")


if __name__ == "__main__":
    test_hydrostatic_equilibrium()