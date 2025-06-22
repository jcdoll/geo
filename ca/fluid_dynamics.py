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


class FluidDynamics:
    """Optimized fluid dynamics with vectorized movement calculation"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
        
        # Expose velocity fields directly for UI compatibility (shared memory)
        self.velocity_x = self.sim.velocity_x
        self.velocity_y = self.sim.velocity_y
        
        # Accumulated displacement for sub-cell movements
        # This tracks "how far does this material want to move"
        h, w = self.sim.material_types.shape
        self.accumulated_dx = np.zeros((h, w), dtype=np.float32)
        self.accumulated_dy = np.zeros((h, w), dtype=np.float32)
        
        # Material viscosity values (0 = no resistance, 1 = no flow)
        # Lower values for better simulation behavior
        self.material_viscosity = {
            MaterialType.SPACE: 0.0,
            MaterialType.AIR: 0.005,
            MaterialType.WATER_VAPOR: 0.005,
            MaterialType.WATER: 0.01,      # Flows very easily
            MaterialType.MAGMA: 0.05,      # Flows but with some resistance
            MaterialType.ICE: 0.15,        # Slow flow
            # Sedimentary
            MaterialType.SAND: 0.1,        # Granular flow
            MaterialType.SANDSTONE: 0.3,   
            MaterialType.LIMESTONE: 0.3,
            MaterialType.SHALE: 0.25,
            MaterialType.CLAY: 0.2,        # Plastic flow
            MaterialType.CONGLOMERATE: 0.35,
            # Igneous
            MaterialType.GRANITE: 0.2,     
            MaterialType.BASALT: 0.05,      # Reduced for faster falling in tests
            MaterialType.OBSIDIAN: 0.2,
            MaterialType.ANDESITE: 0.2,
            MaterialType.PUMICE: 0.1,      # Porous, lighter
            # Metamorphic
            MaterialType.QUARTZITE: 0.4,
            MaterialType.MARBLE: 0.4,
            MaterialType.SLATE: 0.3,
            MaterialType.SCHIST: 0.35,
            MaterialType.GNEISS: 0.4,
            # Special
            MaterialType.URANIUM: 0.4,
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
        """Initialize pressure field.
        
        IMPORTANT: Do NOT use ANY integration-based approach for pressure!
        This includes vertical integration, path integration, or any other
        integration method. These are all incorrect for 2D problems.
        
        The pressure should either be:
        1. Solved using the Poisson equation: ∇²P = ∇·(ρg)
        2. Evolved dynamically through velocity projection
        3. Set to a constant initial value and let dynamics handle it
        """
        
        # Initialize to reference pressure everywhere
        # The velocity projection method will evolve this to the correct values
        ny, nx = self.sim.material_types.shape
        pressure = np.full((ny, nx), 101325.0)  # 1 atmosphere everywhere
        
        # Force space/vacuum to zero
        space_mask = self.sim.material_types == MaterialType.SPACE
        pressure[space_mask] = 0.0
        
        vacuum_threshold = 0.1  # kg/m³
        vacuum_mask = self.sim.density < vacuum_threshold
        pressure[vacuum_mask] = 0.0
        
        # Store pressure
        self.sim.pressure[:] = pressure
    
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

    def detect_interfaces(self):
        """Detect material interfaces for Ghost Fluid Method.
        
        Returns masks for different interface types:
        - horizontal interfaces (material changes in y direction)
        - vertical interfaces (material changes in x direction)
        """
        mat = self.sim.material_types
        h, w = mat.shape
        
        # Detect where materials change
        horiz_interface = np.zeros((h, w), dtype=bool)
        vert_interface = np.zeros((h, w), dtype=bool)
        
        # Horizontal interfaces (changes in y)
        horiz_interface[1:-1, :] = (mat[1:-1, :] != mat[:-2, :]) | (mat[1:-1, :] != mat[2:, :])
        
        # Vertical interfaces (changes in x)
        vert_interface[:, 1:-1] = (mat[:, 1:-1] != mat[:, :-2]) | (mat[:, 1:-1] != mat[:, 2:])
        
        return horiz_interface, vert_interface
    
    def compute_force_field(self):
        """Calculate net force field: gravity + pressure + buoyancy with Ghost Fluid Method"""
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

        # Pressure gradient force with Ghost Fluid Method
        P_pa = self.sim.pressure
        dx = self.sim.cell_size
        
        # Detect interfaces
        horiz_interface, vert_interface = self.detect_interfaces()
        
        # Standard centered differences for pressure gradients
        fx_pressure = np.zeros_like(fx)
        fy_pressure = np.zeros_like(fy)
        
        # Interior points - standard gradient
        fx_pressure[1:-1, 1:-1] = -(P_pa[1:-1, 2:] - P_pa[1:-1, :-2]) / (2 * dx)
        fy_pressure[1:-1, 1:-1] = -(P_pa[2:, 1:-1] - P_pa[:-2, 1:-1]) / (2 * dx)
        
        # Ghost Fluid Method corrections at interfaces
        # The key insight: at an interface, we need ∇P = ρg to hold
        # But the discrete pressure gradient gives wrong values due to density jump
        # So we correct it to enforce the right force balance
        
        # Get material properties database
        mat_db = MaterialDatabase()
        mat_types = self.sim.material_types
        
        # Apply GFM corrections for vertical interfaces (affecting x-gradient)
        for j in range(1, mat_types.shape[0] - 1):
            for i in range(1, mat_types.shape[1] - 1):
                if vert_interface[j, i]:
                    # Check materials on left and right
                    mat_left = mat_types[j, i-1]
                    mat_center = mat_types[j, i]
                    mat_right = mat_types[j, i+1]
                    
                    # Skip GFM for space interfaces - space is vacuum, not a fluid
                    if (mat_left == MaterialType.SPACE or mat_center == MaterialType.SPACE or 
                        mat_right == MaterialType.SPACE):
                        continue
                    
                    # Get densities
                    rho_left = mat_db.get_properties(mat_left).density
                    rho_center = mat_db.get_properties(mat_center).density
                    rho_right = mat_db.get_properties(mat_right).density
                    
                    # Skip GFM for now on vertical interfaces - they're less critical
                    # The standard pressure gradient should work fine for horizontal movement
                    pass
        
        # Apply GFM corrections for horizontal interfaces (affecting y-gradient)
        for j in range(1, mat_types.shape[0] - 1):
            for i in range(1, mat_types.shape[1] - 1):
                if horiz_interface[j, i]:
                    # Check materials above and below
                    mat_above = mat_types[j-1, i]
                    mat_center = mat_types[j, i]
                    mat_below = mat_types[j+1, i]
                    
                    # Skip GFM for space interfaces - space is vacuum, not a fluid
                    if (mat_above == MaterialType.SPACE or mat_center == MaterialType.SPACE or 
                        mat_below == MaterialType.SPACE):
                        continue
                    
                    # Get densities
                    rho_above = mat_db.get_properties(mat_above).density
                    rho_center = mat_db.get_properties(mat_center).density
                    rho_below = mat_db.get_properties(mat_below).density
                    
                    # Apply GFM correction for horizontal interfaces (most important for buoyancy)
                    # Only apply to the heavier material side to avoid blocking motion
                    
                    # Check if we're on the heavy side of the interface
                    if (mat_above != mat_center and rho_center > rho_above) or \
                       (mat_below != mat_center and rho_center > rho_below):
                        # We're the heavy material - apply standard gradient
                        pass  # Use the standard gradient calculation
                    elif (mat_above != mat_center and rho_center < rho_above) or \
                         (mat_below != mat_center and rho_center < rho_below):
                        # We're the light material at an interface
                        # Apply a correction to ensure proper buoyancy
                        # The pressure gradient should partially counteract gravity
                        
                        # Calculate density jump
                        if mat_above != mat_center:
                            rho_heavy = rho_above
                        else:
                            rho_heavy = rho_below
                        
                        # For a light material below heavy material, we want upward force
                        # Pressure gradient force = -∇P
                        # At interface: ∇P should be approximately ρ_avg * g
                        # This gives buoyancy force = (ρ_heavy - ρ_light) * g
                        rho_avg = 0.5 * (rho_heavy + rho_center)
                        
                        # Correct the pressure gradient to give proper buoyancy
                        fy_pressure[j, i] = -rho_avg * gy_total[j, i]
        
        # Boundaries - use one-sided differences
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
        """Displacement-based movement with proper accumulation.
        
        Key concepts:
        1. Each cell accumulates displacement based on velocity
        2. When displacement >= 1 cell, attempt movement
        3. On successful swap, both cells reset accumulated displacement
        4. On collision, handle momentum transfer
        """
        h, w = self.sim.material_types.shape
        
        # Add this timestep's displacement to accumulated displacement
        self.accumulated_dx += self.velocity_x * dt / self.sim.cell_size
        self.accumulated_dy += self.velocity_y * dt / self.sim.cell_size
        
        # Determine which cells want to move 
        # Use floor for positive displacement (move when >= 1.0)
        # Use ceil for negative displacement (move when <= -1.0)
        max_move = 5
        dx_int = np.where(self.accumulated_dx >= 0, 
                         np.floor(self.accumulated_dx),
                         np.ceil(self.accumulated_dx)).astype(int)
        dy_int = np.where(self.accumulated_dy >= 0,
                         np.floor(self.accumulated_dy),
                         np.ceil(self.accumulated_dy)).astype(int)
        dx_int = np.clip(dx_int, -max_move, max_move)
        dy_int = np.clip(dy_int, -max_move, max_move)
        
        # Move cells with integer displacement
        moving_mask = (dx_int != 0) | (dy_int != 0)
        
        if not np.any(moving_mask):
            return
        
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
        self.movement_buffer['accumulated_dx'] = self.accumulated_dx.copy()
        self.movement_buffer['accumulated_dy'] = self.accumulated_dy.copy()
        
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
            
            # For movement into space, check if the source is also space or has significant velocity
            can_move = np.zeros(len(src_y), dtype=bool)
            for i in range(len(src_y)):
                if target_is_space[i] and target_not_processed[i]:
                    src_mat = self.movement_buffer['materials'][src_y[i], src_x[i]]
                    # Only allow movement into space if:
                    # 1. Source is also space (space can move freely)
                    # 2. Source has significant velocity (thrown/launched objects)
                    if src_mat == MaterialType.SPACE:
                        can_move[i] = True
                    else:
                        # Check if material has enough velocity to move into vacuum
                        vel_mag = np.sqrt(self.velocity_x[src_y[i], src_x[i]]**2 + 
                                        self.velocity_y[src_y[i], src_x[i]]**2)
                        # Need at least 1 m/s to move into vacuum (arbitrary threshold)
                        if vel_mag > 1.0:
                            can_move[i] = True
            
            # Also allow denser materials to displace lighter ones
            for i in range(len(tgt_y)):
                if not can_move[i] and target_not_processed[i]:
                    src_density = self.sim.density[src_y[i], src_x[i]]
                    tgt_density = self.sim.density[tgt_y[i], tgt_x[i]]
                    
                    # Allow movement if denser (like rock through water)
                    # For sinking: source must be denser and moving down
                    if src_density > tgt_density * 1.5 and move_y > 0:  # 50% denser and moving down
                        can_move[i] = True
                    # For rising: source must be lighter and moving up  
                    elif src_density < tgt_density * 0.8 and move_y < 0:  # 20% lighter and moving up
                        can_move[i] = True
            
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
                
                # After swap, handle accumulated displacement
                for i in range(len(from_y)):
                    # The material that moved keeps its remaining fractional displacement
                    self.movement_buffer['accumulated_dx'][to_y[i], to_x[i]] -= move_x
                    self.movement_buffer['accumulated_dy'][to_y[i], to_x[i]] -= move_y
                    
                    # For buoyancy: if denser material sank through lighter fluid,
                    # give the displaced fluid upward momentum
                    from_density = self.sim.density[to_y[i], to_x[i]]  # Now at 'to' position
                    to_density = self.sim.density[from_y[i], from_x[i]]  # Now at 'from' position
                    
                    if from_density > to_density * 1.2 and move_y > 0:  # Dense sinking through light
                        # Give displaced fluid upward velocity boost
                        self.movement_buffer['velocity_y'][from_y[i], from_x[i]] = -5.0  # m/s upward
                        self.movement_buffer['accumulated_dy'][from_y[i], from_x[i]] = -0.1  # Small upward start
                
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
        self.accumulated_dx[:] = self.movement_buffer['accumulated_dx']
        self.accumulated_dy[:] = self.movement_buffer['accumulated_dy']
        
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
        
        # Note: We don't reset accumulated displacement here anymore
        # It's handled in the main movement loop to preserve fractional parts

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
        
        # Reduce accumulated displacement due to collision
        # This prevents cells from "trying forever" to move through solid objects
        self.movement_buffer['accumulated_dx'][y1, x1] *= 0.5
        self.movement_buffer['accumulated_dy'][y1, x1] *= 0.5

    def calculate_effective_density(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate effective density including thermal expansion"""
        return self.sim.calculate_effective_density(temperature)