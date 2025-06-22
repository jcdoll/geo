"""
Optimized fluid dynamics module with vectorized movement.
"""

import numpy as np
from typing import Tuple
from scipy import ndimage
import time as _time
try:
    from .materials import MaterialType, MaterialDatabase
    from .pressure_solver import solve_pressure, solve_poisson_variable_multigrid
except ImportError:  # standalone script execution
    from materials import MaterialType, MaterialDatabase  # type: ignore
    from pressure_solver import solve_pressure, solve_poisson_variable_multigrid  # type: ignore


class FluidDynamicsOptimized:
    """Optimized fluid dynamics with vectorized movement calculation"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
        
        # Expose velocity fields directly for UI compatibility (shared memory)
        self.velocity_x = self.sim.velocity_x
        self.velocity_y = self.sim.velocity_y
        
        # Material viscosity values (0 = no resistance, 1 = no flow)
        self.material_viscosity = {
            MaterialType.SPACE: 0.0,
            MaterialType.AIR: 0.01,
            MaterialType.WATER_VAPOR: 0.01,
            MaterialType.WATER: 0.05,      # Flows easily
            MaterialType.MAGMA: 0.3,       # Flows but slowly
            MaterialType.ICE: 0.7,         # Very slow flow
            # Rock types - still flow but very slowly
            MaterialType.GRANITE: 0.9,     # Very resistant but can move
            MaterialType.BASALT: 0.9,      
            MaterialType.OBSIDIAN: 0.9,
            MaterialType.SANDSTONE: 0.85,  # Slightly less resistant
            MaterialType.LIMESTONE: 0.85,
            MaterialType.QUARTZITE: 0.9,
            MaterialType.MARBLE: 0.9,
            MaterialType.SLATE: 0.85,
            MaterialType.URANIUM: 0.9,
        }
        
        # Pre-allocate work arrays for movement
        h, w = self.sim.material_types.shape
        self.movement_buffer = {
            'materials': np.zeros((h, w), dtype=self.sim.material_types.dtype),
            'temperature': np.zeros((h, w), dtype=np.float32),
            'age': np.zeros((h, w), dtype=np.float32),
            'velocity_x': np.zeros((h, w), dtype=np.float32),
            'velocity_y': np.zeros((h, w), dtype=np.float32),
            'density': np.zeros((h, w), dtype=np.float32),
        }
    
    def calculate_planetary_pressure(self):
        """Multigrid solve for pressure using both self-gravity and external gravity fields."""
        
        # On first call, preserve initial pressure as offset
        if not hasattr(self, '_pressure_initialized'):
            self.sim.pressure_offset[:] = self.sim.pressure
            self._pressure_initialized = True

        # Build total gravity field (self-gravity + external gravity)
        gx_total = np.zeros_like(self.sim.density, dtype=np.float64)
        gy_total = np.zeros_like(self.sim.density, dtype=np.float64)

        # Add self-gravity component (if enabled)
        if self.sim.enable_self_gravity and hasattr(self.sim, 'calculate_self_gravity'):
            self.sim.calculate_self_gravity()
            gx_total += self.sim.gravity_x
            gy_total += self.sim.gravity_y

        # Add external gravity component
        if hasattr(self.sim, 'external_gravity'):
            g_ext_x, g_ext_y = self.sim.external_gravity
            if g_ext_x != 0.0 or g_ext_y != 0.0:
                gx_total += g_ext_x
                gy_total += g_ext_y

        # Calculate divergence and gradients for pressure equation
        div_g = np.zeros_like(gx_total)
        dx = self.sim.cell_size
        div_g[1:-1, 1:-1] = (
            (gx_total[1:-1, 2:] - gx_total[1:-1, :-2]) + (gy_total[2:, 1:-1] - gy_total[:-2, 1:-1])
        ) / (2 * dx)

        # Build RHS for Poisson equation
        rhs = np.zeros_like(self.sim.density)
        
        # Calculate density gradient
        grad_rho_x = np.zeros_like(self.sim.density)
        grad_rho_y = np.zeros_like(self.sim.density)
        
        grad_rho_x[1:-1, 1:-1] = (self.sim.density[1:-1, 2:] - self.sim.density[1:-1, :-2]) / (2 * dx)
        grad_rho_y[1:-1, 1:-1] = (self.sim.density[2:, 1:-1] - self.sim.density[:-2, 1:-1]) / (2 * dx)
        
        # Self-gravity contribution
        if self.sim.enable_self_gravity:
            rhs += (self.sim.density * div_g) / 1e6   # Pa → MPa
            g_dot_grad_rho = gx_total * grad_rho_x + gy_total * grad_rho_y
            rhs += g_dot_grad_rho / 1e6
        
        # External gravity contribution
        if hasattr(self.sim, 'external_gravity'):
            g_ext_x, g_ext_y = self.sim.external_gravity
            if abs(g_ext_x) > 1e-10 or abs(g_ext_y) > 1e-10:
                div_rho_g = g_ext_x * grad_rho_x + g_ext_y * grad_rho_y
                rhs += div_rho_g / 1e6

        # Solve Poisson equation for pressure
        pressure = solve_pressure(rhs, dx, bc_type='neumann')
        
        # Force space/vacuum cells to zero pressure
        space_mask = self.sim.material_types == MaterialType.SPACE
        pressure[space_mask] = 0.0
        
        vacuum_threshold = 0.1  # kg/m³
        vacuum_mask = self.sim.density < vacuum_threshold
        pressure[vacuum_mask] = 0.0
        
        # Clip negative pressures
        pressure = np.maximum(pressure, 0.0)

        # Store pressure
        self.sim.pressure[:] = pressure * 1e6 + self.sim.pressure_offset
    
    def apply_unified_kinematics(self, dt: float) -> None:
        """Simplified kinematics - all materials flow based on viscosity."""
        if not hasattr(self.sim, "_perf_times"):
            self.sim._perf_times = {}
        _t0 = _time.perf_counter()

        rho = self.sim.density

        # 1. Assemble force field
        _t_force_start = _time.perf_counter()
        fx, fy = self.compute_force_field()
        self.sim._perf_times["uk_force_assembly"] = _time.perf_counter() - _t_force_start

        # 2. Update velocities with viscosity damping
        _t_vel_start = _time.perf_counter()
        self.update_velocities_with_viscosity(fx, fy, rho, dt)
        self.sim._perf_times["uk_velocity_update"] = _time.perf_counter() - _t_vel_start

        # 3. Velocity projection for incompressibility (fluids only)
        _t_proj_start = _time.perf_counter()
        self.apply_velocity_projection(dt)
        self.sim._perf_times["uk_projection"] = _time.perf_counter() - _t_proj_start

        # 4. Apply velocity-based movement (OPTIMIZED)
        _t_move_start = _time.perf_counter()
        self.apply_velocity_movement_vectorized(dt)
        self.sim._perf_times["uk_movement"] = _time.perf_counter() - _t_move_start

    def compute_force_field(self):
        """Calculate net force field: gravity + pressure + buoyancy"""
        rho = self.sim.density

        # Ensure gravity is up-to-date
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

        # Gravity force density (N/m³)
        fx = rho * gx_total
        fy = rho * gy_total

        # Pressure gradient force
        P_pa = self.sim.pressure
        dx = self.sim.cell_size
        
        # Centered differences for pressure gradients
        fx_pressure = np.zeros_like(fx)
        fy_pressure = np.zeros_like(fy)
        
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

    def update_velocities_with_viscosity(self, fx: np.ndarray, fy: np.ndarray, 
                                         rho: np.ndarray, dt: float):
        """Update velocities with forces and material viscosity damping."""
        # Calculate accelerations
        accel_x = np.where(rho > 0, fx / rho, 0.0)
        accel_y = np.where(rho > 0, fy / rho, 0.0)

        # Update velocities
        self.velocity_x += accel_x * dt
        self.velocity_y += accel_y * dt

        # Apply material viscosity damping
        viscosity = np.zeros_like(self.velocity_x)
        for mat_type, visc in self.material_viscosity.items():
            mask = self.sim.material_types == mat_type
            viscosity[mask] = visc
        
        # Damping factor (ensures stability)
        damping = np.maximum(0.0, 1.0 - viscosity * dt)
        self.velocity_x *= damping
        self.velocity_y *= damping

        # Velocity clamping for stability (always on for simplified system)
        max_u = 2.0 * self.sim.cell_size / dt  # Allow max 2 cells per timestep
        np.clip(self.velocity_x, -max_u, max_u, out=self.velocity_x)
        np.clip(self.velocity_y, -max_u, max_u, out=self.velocity_y)

    def apply_velocity_projection(self, dt: float):
        """Apply velocity projection to enforce incompressibility in fluids."""
        # Define fluid types
        fluid_types = {MaterialType.WATER, MaterialType.MAGMA, MaterialType.AIR, MaterialType.WATER_VAPOR}
        is_fluid = np.zeros(self.sim.material_types.shape, dtype=bool)
        for ftype in fluid_types:
            is_fluid |= (self.sim.material_types == ftype)
        
        fluid_mask = is_fluid & (self.sim.density > 0.0)
        
        # Calculate velocity divergence
        div_u = np.zeros_like(self.velocity_x)
        dx_m = self.sim.cell_size
        
        div_u[1:-1, 1:-1] = (
            (self.velocity_x[1:-1, 2:] - self.velocity_x[1:-1, :-2]) +
            (self.velocity_y[2:, 1:-1] - self.velocity_y[:-2, 1:-1])
        ) / (2 * dx_m)

        # Only enforce in fluid regions
        div_u = div_u * fluid_mask
        
        # Build RHS for projection
        rhs = div_u / dt
        
        # Use fluid density for projection
        inv_rho_fluid = np.where(fluid_mask, 1.0 / self.sim.density, 0.0)
        
        # Solve for pressure correction
        phi = solve_poisson_variable_multigrid(
            rhs,
            inv_rho_fluid,
            dx_m,
            tol=1e-3,
            max_cycles=10,
        )
        
        # Apply velocity correction
        grad_phi_y, grad_phi_x = np.gradient(phi, dx_m)
        
        # Only correct in fluid regions
        grad_phi_x[~fluid_mask] = 0.0
        grad_phi_y[~fluid_mask] = 0.0
        
        self.velocity_x -= grad_phi_x * dt * inv_rho_fluid
        self.velocity_y -= grad_phi_y * dt * inv_rho_fluid

    def apply_velocity_movement_vectorized(self, dt: float):
        """Vectorized velocity-based movement with collision handling.
        
        Key optimizations:
        1. Batch processing of cells with similar velocities
        2. Vectorized displacement calculations
        3. Efficient multi-field swapping
        4. Reduced Python loops
        """
        h, w = self.sim.material_types.shape
        
        # Calculate displacement for each cell
        dx = self.velocity_x * dt / self.sim.cell_size
        dy = self.velocity_y * dt / self.sim.cell_size
        
        # Only move cells with significant velocity
        move_threshold = 0.1  # Fraction of cell
        vel_mag = np.sqrt(dx**2 + dy**2)
        moving_mask = vel_mag > move_threshold
        
        if not np.any(moving_mask):
            return
        
        # Group cells by movement direction (quantized to integer displacements)
        max_move = 5
        dx_int = np.clip(np.round(dx), -max_move, max_move).astype(int)
        dy_int = np.clip(np.round(dy), -max_move, max_move).astype(int)
        
        # Process each unique movement vector
        unique_moves = np.unique(np.stack([dy_int[moving_mask], dx_int[moving_mask]], axis=1), axis=0)
        
        # Sort by magnitude (process larger movements first)
        move_mags = np.sqrt(unique_moves[:, 0]**2 + unique_moves[:, 1]**2)
        unique_moves = unique_moves[np.argsort(-move_mags)]
        
        # Track which cells have been processed
        processed = np.zeros((h, w), dtype=bool)
        
        # Initialize buffers with current state
        self.movement_buffer['materials'][:] = self.sim.material_types
        self.movement_buffer['temperature'][:] = self.sim.temperature
        self.movement_buffer['age'][:] = self.sim.age
        self.movement_buffer['velocity_x'][:] = self.velocity_x
        self.movement_buffer['velocity_y'][:] = self.velocity_y
        
        # Process each movement group
        for move_y, move_x in unique_moves:
            if move_y == 0 and move_x == 0:
                continue
                
            # Find all cells with this movement
            cells_mask = (dx_int == move_x) & (dy_int == move_y) & moving_mask & ~processed
            
            if not np.any(cells_mask):
                continue
                
            # Get source positions
            src_y, src_x = np.where(cells_mask)
            
            # Calculate target positions
            tgt_y = src_y + move_y
            tgt_x = src_x + move_x
            
            # Filter valid targets (in bounds)
            valid = (tgt_y >= 0) & (tgt_y < h) & (tgt_x >= 0) & (tgt_x < w)
            
            # Handle boundary collisions
            boundary_hits = ~valid
            if np.any(boundary_hits):
                # Reflect velocities at boundaries
                hit_y = src_y[boundary_hits]
                hit_x = src_x[boundary_hits]
                
                # Check which boundary was hit
                y_boundary = (tgt_y[boundary_hits] < 0) | (tgt_y[boundary_hits] >= h)
                x_boundary = (tgt_x[boundary_hits] < 0) | (tgt_x[boundary_hits] >= w)
                
                # Damped reflection
                if np.any(y_boundary):
                    self.movement_buffer['velocity_y'][hit_y[y_boundary], hit_x[y_boundary]] *= -0.5
                if np.any(x_boundary):
                    self.movement_buffer['velocity_x'][hit_y[x_boundary], hit_x[x_boundary]] *= -0.5
            
            # Process valid movements
            src_y = src_y[valid]
            src_x = src_x[valid]
            tgt_y = tgt_y[valid]
            tgt_x = tgt_x[valid]
            
            if len(src_y) == 0:
                continue
            
            # Check target occupancy
            target_is_space = self.movement_buffer['materials'][tgt_y, tgt_x] == MaterialType.SPACE
            target_not_processed = ~processed[tgt_y, tgt_x]
            can_move = target_is_space & target_not_processed
            
            # Perform movements where possible
            move_indices = np.where(can_move)[0]
            if len(move_indices) > 0:
                # Get movement coordinates
                from_y = src_y[move_indices]
                from_x = src_x[move_indices]
                to_y = tgt_y[move_indices]
                to_x = tgt_x[move_indices]
                
                # Swap all properties at once
                self._vectorized_swap(from_y, from_x, to_y, to_x)
                
                # Mark as processed
                processed[from_y, from_x] = True
                processed[to_y, to_x] = True
            
            # Handle collisions for blocked movements
            collision_indices = np.where(~can_move & target_not_processed)[0]
            if len(collision_indices) > 0:
                # Apply collision physics
                self._vectorized_collision(
                    src_y[collision_indices], src_x[collision_indices],
                    tgt_y[collision_indices], tgt_x[collision_indices]
                )
                processed[src_y[collision_indices], src_x[collision_indices]] = True
        
        # Copy buffers back to main arrays
        self.sim.material_types[:] = self.movement_buffer['materials']
        self.sim.temperature[:] = self.movement_buffer['temperature']
        self.sim.age[:] = self.movement_buffer['age']
        self.velocity_x[:] = self.movement_buffer['velocity_x']
        self.velocity_y[:] = self.movement_buffer['velocity_y']
        
        # Mark properties dirty
        self.sim._properties_dirty = True

    def _vectorized_swap(self, from_y, from_x, to_y, to_x):
        """Swap multiple cells at once using vectorized operations."""
        # Use temporary arrays to avoid overwriting during swap
        temp_mat = self.movement_buffer['materials'][from_y, from_x].copy()
        temp_temp = self.movement_buffer['temperature'][from_y, from_x].copy()
        temp_age = self.movement_buffer['age'][from_y, from_x].copy()
        temp_vx = self.movement_buffer['velocity_x'][from_y, from_x].copy()
        temp_vy = self.movement_buffer['velocity_y'][from_y, from_x].copy()
        
        # Move 'to' values to 'from' positions
        self.movement_buffer['materials'][from_y, from_x] = self.movement_buffer['materials'][to_y, to_x]
        self.movement_buffer['temperature'][from_y, from_x] = self.movement_buffer['temperature'][to_y, to_x]
        self.movement_buffer['age'][from_y, from_x] = self.movement_buffer['age'][to_y, to_x]
        self.movement_buffer['velocity_x'][from_y, from_x] = self.movement_buffer['velocity_x'][to_y, to_x]
        self.movement_buffer['velocity_y'][from_y, from_x] = self.movement_buffer['velocity_y'][to_y, to_x]
        
        # Move 'from' values to 'to' positions
        self.movement_buffer['materials'][to_y, to_x] = temp_mat
        self.movement_buffer['temperature'][to_y, to_x] = temp_temp
        self.movement_buffer['age'][to_y, to_x] = temp_age
        self.movement_buffer['velocity_x'][to_y, to_x] = temp_vx
        self.movement_buffer['velocity_y'][to_y, to_x] = temp_vy

    def _vectorized_collision(self, y1, x1, y2, x2):
        """Handle collisions for multiple cell pairs using vectorized operations."""
        # Get masses
        cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
        m1 = self.sim.density[y1, x1] * cell_volume
        m2 = self.sim.density[y2, x2] * cell_volume
        
        # Filter out zero-mass collisions
        valid = (m1 > 0) & (m2 > 0)
        if not np.any(valid):
            return
            
        y1, x1, y2, x2 = y1[valid], x1[valid], y2[valid], x2[valid]
        m1, m2 = m1[valid], m2[valid]
        
        # Get velocities
        v1x = self.movement_buffer['velocity_x'][y1, x1]
        v1y = self.movement_buffer['velocity_y'][y1, x1]
        v2x = self.movement_buffer['velocity_x'][y2, x2]
        v2y = self.movement_buffer['velocity_y'][y2, x2]
        
        # Calculate center of mass velocities
        total_mass = m1 + m2
        vcm_x = (m1 * v1x + m2 * v2x) / total_mass
        vcm_y = (m1 * v1y + m2 * v2y) / total_mass
        
        # Elastic collision with damping
        damping = 0.5
        self.movement_buffer['velocity_x'][y1, x1] = vcm_x + damping * (v2x - vcm_x) * m2 / total_mass
        self.movement_buffer['velocity_y'][y1, x1] = vcm_y + damping * (v2y - vcm_y) * m2 / total_mass
        self.movement_buffer['velocity_x'][y2, x2] = vcm_x + damping * (v1x - vcm_x) * m1 / total_mass
        self.movement_buffer['velocity_y'][y2, x2] = vcm_y + damping * (v1y - vcm_y) * m1 / total_mass

    def calculate_effective_density(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate effective density including thermal expansion"""
        return self.sim.calculate_effective_density(temperature)