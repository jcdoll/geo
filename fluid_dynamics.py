"""
Fluid dynamics module for geological simulation.
Handles pressure calculations, gravitational collapse, density stratification, and fluid migration.
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
    """Fluid dynamics calculations for geological simulation"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
        
        # Expose velocity fields directly for UI compatibility (shared memory)
        self.velocity_x = self.sim.velocity_x
        self.velocity_y = self.sim.velocity_y
        self.dv_thresh = 0.01  # m/s – velocity-difference threshold for swapping (reduced for surface tension)
        # Scale binding force appropriately for cell size (100m cells = 1e6 m³ volume)
        # Rock cohesion ~1-10 MPa, so binding force should be ~1e6 to 1e7 N for 100m cell
        self.solid_binding_force = 1e6  # N – reference cohesion between solid voxels (scaled for cell size)
        self.velocity_clamp = True # Enable/disable velocity clamping (enable if instability)
        self.velocity_threshold = True  # Enable/disable velocity threshold for swapping

        # Create binding matrix using material processes helper
        try:
            from .material_processes import MaterialProcesses
        except ImportError:
            from material_processes import MaterialProcesses
        self._binding_matrix, self._mat_index = MaterialProcesses.create_binding_matrix(self.solid_binding_force)
    
    def calculate_planetary_pressure(self):
        """Multigrid solve for pressure using self-gravity field plus surface tension."""

        # Ensure gravity field is up-to-date
        if hasattr(self.sim, 'calculate_self_gravity'):
            self.sim.calculate_self_gravity()

        gx = self.sim.gravity_x
        gy = self.sim.gravity_y

        # Divergence of gravity
        div_g = np.zeros_like(gx)
        dx = self.sim.cell_size
        div_g[1:-1, 1:-1] = (
            (gx[1:-1, 2:] - gx[1:-1, :-2]) + (gy[2:, 1:-1] - gy[:-2, 1:-1])
        ) / (2 * dx)

        # Build RHS: -ρ ∇·g  (MPa units -> divide 1e6)
        rhs = (self.sim.density * div_g) / 1e6   # Pa → MPa

        # Add surface tension / cohesive pressure for fluids
        surface_tension_rhs = self._compute_surface_tension_pressure() / 1e6  # Pa → MPa
        rhs += surface_tension_rhs

        # Solve Poisson
        pressure = solve_pressure(rhs, dx)

        # Store & add persistent offsets (don't clip negative pressures!)
        self.sim.pressure[:] = pressure + self.sim.pressure_offset
    
    def apply_unified_kinematics(self, dt: float) -> None:
        """Move solids, liquids and gases in one deterministic pass.

        The algorithm considers *all* voxels and performs density-driven
        rearrangements along the local gravity vector using a single sweep.
        It replaces the former density-stratification, fluid-migration and
        gravitational-collapse routines.

        Current simplifications (first implementation):
            • One swap per voxel per macro-step (prevents ping-pong).
            • 8-neighbour neighbourhood (isotropic).
            • Uses centre-of-mass as gravity direction (radial field).  A
              refined version could use the pre-computed `gravity_x/y`.
            • Operates in-place via vectorised NumPy masks; no RNG.

        Density-driven swapping has been removed in favor of pure physics-based approach.
        All material movement is now handled by:
        1. Force-based swapping (apply_force_based_swapping) - handles pressure/gravity forces
        2. Velocity-based swapping - ensures kinematic consistency  
        3. Surface tension - handles fluid interface dynamics
        This aligns with PHYSICS.md documentation which avoids simple heuristic swapping.
        """

        if not hasattr(self.sim, "_perf_times"):
            self.sim._perf_times = {}
        _t0 = _time.perf_counter()

        rho = self.sim.density

        # Assemble net body-force field
        _t_force_start = _time.perf_counter()
        fx, fy = self.compute_force_field()
        self.sim._perf_times["uk_force_assembly"] = _time.perf_counter() - _t_force_start

        # Velocity update
        _t_vel_start = _time.perf_counter()
        self.update_velocities(fx, fy, rho, dt)
        self.sim._perf_times["uk_velocity_update"] = _time.perf_counter() - _t_vel_start

        # ------------------------------------------------------------------
        # 3) Projection – make velocity approximately divergence-free
        #    using multigrid Poisson solve (pressure_solver.solve_pressure).
        #    TODO: implement FFT/DST version and benchmark.
        # ------------------------------------------------------------------
        _t_proj_start = _time.perf_counter()

        # Divergence of tentative velocity (1/s)
        div_u = np.zeros_like(self.velocity_x)
        dx_m = self.sim.cell_size
        fluid_mask = rho > 0.0
        div_u[1:-1, 1:-1] = (
            (self.velocity_x[1:-1, 2:] - self.velocity_x[1:-1, :-2]) +
            (self.velocity_y[2:, 1:-1] - self.velocity_y[:-2, 1:-1])
        ) / (2 * dx_m)

        div_u = div_u * fluid_mask  # Ignore vacuum divergence

        # ------------------------------------------------------------------
        # Variable-density Chorin projection
        #    ∇·(1/ρ ∇φ) = div_u / Δt
        # ------------------------------------------------------------------
        rhs = (div_u / dt) * fluid_mask
        inv_rho = np.where(rho > 0, 1.0 / rho, 0.0)

        # Projection does not need micro-Pascal precision; loosen tolerance for speed
        phi = solve_poisson_variable_multigrid(
            rhs,
            inv_rho,
            dx_m,
            tol=1e-3 if self.sim.quality >= 2 else 5e-4,
            max_cycles=10,
        )

        # Gradient of φ
        grad_phi_y, grad_phi_x = np.gradient(phi, dx_m)
        grad_phi_x[~fluid_mask] = 0.0
        grad_phi_y[~fluid_mask] = 0.0

        # Velocity correction  u = u* − Δt/ρ ∇φ
        self.velocity_x -= grad_phi_x * dt * inv_rho
        self.velocity_y -= grad_phi_y * dt * inv_rho

        self.sim._perf_times["uk_projection"] = _time.perf_counter() - _t_proj_start

        # ------------------------------------------------------------------
        # 4) Accumulate displacement and determine swaps via force criteria
        # ------------------------------------------------------------------
        # Instead of density-based heavier-than-target rule we now use the
        # force-based binding criterion defined in `apply_force_based_swapping`.
        _t_swap_force_start = _time.perf_counter()
        
        # Apply physics-based surface tension through local cohesive forces
        if getattr(self.sim, 'enable_surface_tension', True):
            surface_swaps = self.apply_physics_based_surface_tension()
        else:
            surface_swaps = 0
        
        # Apply group dynamics for rigid bodies (ice, rock)
        self.apply_group_dynamics()
        
        # Then apply general force-based swapping for all material interactions
        self.apply_force_based_swapping()
        
        self.sim._perf_times["uk_force_swaps"] = _time.perf_counter() - _t_swap_force_start

        # Optional solid drag (retain previous behaviour)
        if getattr(self.sim, "enable_solid_drag", True):
            solid_mask = self.sim._get_solid_mask()
            self.velocity_x[solid_mask] *= 0.2
            self.velocity_y[solid_mask] *= 0.2

    # Velocity integration dispatch method
    def update_velocities(self, fx: np.ndarray, fy: np.ndarray, rho: np.ndarray, dt: float):
        """Update velocities using selected integration method.
        
        Available methods:
        - 'explicit_euler': Simple first-order explicit integration (default)
        - Future: 'rk4', 'verlet', 'semi_implicit_euler', etc.
        
        To add a new method:
        1. Add method name to the dispatch table below
        2. Implement _update_velocities_<method_name>(fx, fy, rho, dt)
        3. Set sim.velocity_integration_method = '<method_name>'
        
        Args:
            fx, fy: Force field components (N)
            rho: Density field (kg/m³)
            dt: Time step (s)
        """
        # Dispatch based on integration method
        method = getattr(self.sim, 'velocity_integration_method', 'explicit_euler')
        
        if method == 'explicit_euler':
            self._update_velocities_explicit_euler(fx, fy, rho, dt)
        # elif method == 'rk4':
        #     self._update_velocities_rk4(fx, fy, rho, dt)
        # elif method == 'verlet':
        #     self._update_velocities_verlet(fx, fy, rho, dt)
        else:
            raise ValueError(f"Unknown velocity integration method: {method}")
            
    # Available integration methods (implement as needed):
    
    def _update_velocities_explicit_euler(self, fx: np.ndarray, fy: np.ndarray, rho: np.ndarray, dt: float):
        """Update velocities using explicit Euler integration.
        
        Simple first-order method: v_new = v_old + (F/ρ) * dt
        Fast and stable for small time steps with sufficient damping.
        
        Args:
            fx, fy: Force field components (N)
            rho: Density field (kg/m³) 
            dt: Time step (s)
        """
        dt_seconds = dt  # dt already in seconds
        
        # Calculate accelerations: a = F/m = F/(ρ*V) = F/ρ (since V=1 for unit cells)
        # Handle divide by zero for vacuum regions
        accel_x = np.where(rho > 0, fx / rho, 0.0)
        accel_y = np.where(rho > 0, fy / rho, 0.0)

        # Explicit Euler integration: v = v + a*dt
        self.velocity_x += accel_x * dt_seconds
        self.velocity_y += accel_y * dt_seconds

        # CFL-based velocity clamping for numerical stability (see PHYSICS.md)
        if self.velocity_clamp:
            max_u = 0.5 * self.sim.cell_size / dt_seconds  # CFL ≤ 0.5
            np.clip(self.velocity_x, -max_u, max_u, out=self.velocity_x)
            np.clip(self.velocity_y, -max_u, max_u, out=self.velocity_y)
    
    def calculate_effective_density(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate effective density including thermal expansion"""
        # Use the comprehensive method from CoreState
        return self.sim.calculate_effective_density(temperature)
    
    def _perform_material_swaps(self, src_y: np.ndarray, src_x: np.ndarray, 
                               tgt_y: np.ndarray, tgt_x: np.ndarray):
        """Perform material and temperature swaps between source and target cells"""
        if len(src_y) == 0:
            return
        
        # Store source values
        src_materials = self.sim.material_types[src_y, src_x].copy()
        src_temperatures = self.sim.temperature[src_y, src_x].copy()
        src_ages = self.sim.age[src_y, src_x].copy()
        
        # Store target values
        tgt_materials = self.sim.material_types[tgt_y, tgt_x].copy()
        tgt_temperatures = self.sim.temperature[tgt_y, tgt_x].copy()
        tgt_ages = self.sim.age[tgt_y, tgt_x].copy()
        
        # Perform swaps
        self.sim.material_types[src_y, src_x] = tgt_materials
        self.sim.temperature[src_y, src_x] = tgt_temperatures
        self.sim.age[src_y, src_x] = tgt_ages
        
        self.sim.material_types[tgt_y, tgt_x] = src_materials
        self.sim.temperature[tgt_y, tgt_x] = src_temperatures
        self.sim.age[tgt_y, tgt_x] = src_ages
        
        # Momentum conservation for velocity swaps
        if hasattr(self, 'velocity_x') and hasattr(self, 'velocity_y'):
            # Get current velocities
            src_vx = self.velocity_x[src_y, src_x].copy()
            src_vy = self.velocity_y[src_y, src_x].copy()
            tgt_vx = self.velocity_x[tgt_y, tgt_x].copy()
            tgt_vy = self.velocity_y[tgt_y, tgt_x].copy()
            
            # Get masses (density * cell volume)
            src_density = self.sim.density[src_y, src_x]
            tgt_density = self.sim.density[tgt_y, tgt_x]
            # Use cell_depth for proper 3D mass calculation
            cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
            
            src_mass = src_density * cell_volume
            tgt_mass = tgt_density * cell_volume
            
            # Calculate total momentum
            px_total = src_mass * src_vx + tgt_mass * tgt_vx
            py_total = src_mass * src_vy + tgt_mass * tgt_vy
            
            # After swap, materials have switched places but momentum is conserved
            # New velocities based on swapped masses
            # Note: After swap, src location has tgt material and vice versa
            new_src_mass = tgt_mass  # tgt material now at src location
            new_tgt_mass = src_mass  # src material now at tgt location
            
            # Conserve momentum: swap velocities with the materials
            # When materials swap, velocities should swap too to maintain momentum
            # (material carries its velocity with it)
            self.velocity_x[src_y, src_x] = tgt_vx
            self.velocity_y[src_y, src_x] = tgt_vy
            self.velocity_x[tgt_y, tgt_x] = src_vx
            self.velocity_y[tgt_y, tgt_x] = src_vy
        
        # Mark properties as dirty for recalculation
        self.sim._properties_dirty = True

    # ------------------------------------------------------------------
    # Helper: buoyancy force  F_b = (ρ_ref − ρ) g
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_buoyancy_force(rho: np.ndarray,
                                 gx: np.ndarray,
                                 gy: np.ndarray,
                                 *,
                                 kernel_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Return buoyancy force components for each cell.

        Parameters
        ----------
        rho : ndarray
            Density field (kg/m³).
        gx, gy : ndarray
            Gravitational acceleration components (m/s²).
        kernel_size : int, optional
            Side length of the uniform filter window used to estimate the
            local reference density.  Defaults to 3 (Moore neighbourhood).
        """
        rho_ref = ndimage.uniform_filter(rho, size=kernel_size, mode="nearest")
        drho = rho_ref - rho  # positive where cell is lighter than surroundings
        fbx = drho * gx
        fby = drho * gy
        return fbx, fby

    # ------------------------------------------------------------------
    # NEW: force-field assembly (gravity – ∇P) exposed for re-use
    # ------------------------------------------------------------------
    def compute_force_field(self):
        """Assemble net body force density f = ρ g − ∇P and store on the simulation.
        
        Returns force density (N/m³), not total force. This is more numerically stable
        and makes material binding thresholds scale-independent.
        """
        rho = self.sim.density

        # Ensure gravity is up-to-date
        if hasattr(self.sim, 'calculate_self_gravity'):
            self.sim.calculate_self_gravity()
        gx_total = self.sim.gravity_x
        gy_total = self.sim.gravity_y

        # Optional external constant gravity
        if hasattr(self.sim, 'external_gravity'):
            g_ext_x, g_ext_y = self.sim.external_gravity
            if g_ext_x != 0.0 or g_ext_y != 0.0:
                gx_total = gx_total + g_ext_x
                gy_total = gy_total + g_ext_y

        # Gravity term - force density (N/m³)
        fx = rho * gx_total
        fy = rho * gy_total

        # Pressure gradient term – compute via face normals (4-neighbour)
        P_pa = self.sim.pressure * 1e6  # MPa → Pa
        dx = self.sim.cell_size

        # Store pressure forces separately for debugging
        fx_pressure = np.zeros_like(fx)
        fy_pressure = np.zeros_like(fy)

        # X-direction forces - pressure gradient (force density N/m³)
        fx_pressure[:, :-1] -= (P_pa[:, :-1] - P_pa[:, 1:]) / dx  # interface with east neighbour
        fx_pressure[:, 1:]  += (P_pa[:, :-1] - P_pa[:, 1:]) / dx  # equal & opposite on west side

        # Y-direction forces
        fy_pressure[:-1, :] -= (P_pa[:-1, :] - P_pa[1:, :]) / dx  # south neighbour
        fy_pressure[1:, :]  += (P_pa[:-1, :] - P_pa[1:, :]) / dx  # north side

        # Add pressure forces to total
        fx += fx_pressure
        fy += fy_pressure

        # Store for any other module this macro-step
        self.sim.force_x = fx
        self.sim.force_y = fy
        return fx, fy

    def _binding_threshold(self, mt_a, mt_b, temp_avg):
        """Return binding force threshold between two materials (scalar, N).
        Uses precomputed matrix plus temperature weakening only above melting point.
        """
        from materials import MaterialType  # local import
        idx_a = self._mat_index[mt_a]
        idx_b = self._mat_index[mt_b]
        base_th = self._binding_matrix[idx_a, idx_b]
        
        # Temperature weakening only affects bonds where at least one side is solid
        if np.isfinite(base_th) and base_th > 0:
            # Only weaken binding above melting point (~1200°C)
            melting_point = 1473.15  # 1200°C in Kelvin
            if temp_avg > melting_point:
                # Gentle weakening above melting point, minimum 50% strength
                temp_factor = max(0.5, 1.0 - (temp_avg - melting_point) / 1000.0)
                return base_th * temp_factor
            else:
                # Full binding strength below melting point
                return base_th
        return base_th

    def apply_force_based_swapping(self):
        """Swap neighbouring cells that overcome their binding threshold.

        Implements the rule from PHYSICS.md Section *Cell-Swapping Mechanics*.
        """
        fx = getattr(self.sim, 'force_x', None)
        fy = getattr(self.sim, 'force_y', None)
        if fx is None or fy is None:
            fx, fy = self.compute_force_field()
        vx = self.sim.velocity_x
        vy = self.sim.velocity_y
        mt = self.sim.material_types
        temp = self.sim.temperature
        h, w = mt.shape
        
        # 4-neighbour offsets only (no diagonals)
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # gather candidate swaps
        cand_src_y = []
        cand_src_x = []
        cand_tgt_y = []
        cand_tgt_x = []

        # Helper uses existing _binding_threshold against a reference solid (granite)
        from materials import MaterialType  # local import to avoid circular

        # Iterate over all cells and check neighbors
        for y in range(h):
            for x in range(w):
                # Don't skip any cells - SPACE needs to participate in swaps too
                    
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    
                    # Get forces and velocities
                    fsrc_x, fsrc_y = fx[y, x], fy[y, x]
                    ftgt_x, ftgt_y = fx[ny, nx], fy[ny, nx]
                    vsrc_x, vsrc_y = vx[y, x], vy[y, x]
                    vtgt_x, vtgt_y = vx[ny, nx], vy[ny, nx]
                    
                    # Force and velocity differences
                    dFx, dFy = fsrc_x - ftgt_x, fsrc_y - ftgt_y
                    F_net = np.hypot(dFx, dFy)
                    dVx, dVy = vsrc_x - vtgt_x, vsrc_y - vtgt_y
                    V_diff = np.hypot(dVx, dVy)
                    
                    # Binding threshold between the two materials
                    temp_avg = 0.5 * (temp[y, x] + temp[ny, nx])
                    threshold = self._binding_threshold(mt[y, x], mt[ny, nx], temp_avg)
                    
                    # For solid materials, check if source can overcome its own binding
                    # For fluid materials, binding is typically 0
                    # Check material's self-binding (internal cohesion)
                    src_bind = self._binding_threshold(mt[y, x], mt[y, x], temp[y, x])
                    
                    # Directional force projection
                    proj_src = fsrc_x * dx + fsrc_y * dy
                    
                    # Check conditions for swapping:
                    # 1. Net force must overcome material binding threshold
                    # 2. Velocity difference must be significant (if threshold enabled)
                    # 3. For solids, source must overcome its own binding
                    cond_force = F_net > threshold
                    # Use velocity threshold only if enabled
                    cond_velocity = V_diff >= self.dv_thresh if self.velocity_threshold else True
                    
                    # For solids, also check if force can overcome binding
                    if src_bind > 0:  # Solid material
                        cond_src = abs(proj_src) > src_bind
                    else:  # Fluid material
                        cond_src = True
                    
                    if cond_force and cond_velocity and cond_src:
                        cand_src_y.append(y)
                        cand_src_x.append(x)
                        cand_tgt_y.append(ny)
                        cand_tgt_x.append(nx)

        # Perform swaps if any
        if cand_src_y:
            src_y = np.array(cand_src_y)
            src_x = np.array(cand_src_x)
            tgt_y = np.array(cand_tgt_y)
            tgt_x = np.array(cand_tgt_x)
            
            # Deduplicate conflicts
            src_y, src_x, tgt_y, tgt_x = self.sim._dedupe_swap_pairs(src_y, src_x, tgt_y, tgt_x)
            
            if len(src_y):
                self._perform_material_swaps(src_y, src_x, tgt_y, tgt_x)

    def apply_physics_based_surface_tension(self):
        """Apply surface tension through local cohesive forces between fluid cells.
        
        This hybrid approach uses physics-based cohesive forces but processes
        multiple cells per timestep to achieve visible effects in discrete simulations.
        """
        from materials import MaterialType
        from scipy import ndimage
        
        mt = self.sim.material_types
        temp = self.sim.temperature
        h, w = mt.shape
        
        # Identify fluid cells
        fluid_types = {MaterialType.WATER, MaterialType.MAGMA}
        fluid_mask = np.zeros((h, w), dtype=bool)
        for ftype in fluid_types:
            fluid_mask |= (mt == ftype)
        
        if not np.any(fluid_mask):
            return 0
            
        # Surface tension coefficient - strong for discrete cells
        sigma_base = 50000.0  # Pa·m (from PHYSICS.md)
        cell_size = self.sim.cell_size
        
        # For each fluid cell at an interface, calculate cohesive forces
        swaps_performed = 0
        max_swaps_per_pass = 50
        num_passes = 3  # Multiple passes for faster evolution
        
        for pass_num in range(num_passes):
            # Find interface cells fresh each pass
            non_fluid_mask = ~fluid_mask
            struct = np.ones((3, 3), dtype=bool)
            non_fluid_dilated = ndimage.binary_dilation(non_fluid_mask, structure=struct)
            interface_mask = fluid_mask & non_fluid_dilated
            
            interface_cells = np.where(interface_mask)
            if len(interface_cells[0]) == 0:
                break
                
            # Calculate curvature for all interface cells
            interface_data = []
            
            for i in range(len(interface_cells[0])):
                y, x = interface_cells[0][i], interface_cells[1][i]
                
                # Count neighbors to determine curvature
                fluid_count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mt[ny, nx] in fluid_types:
                                fluid_count += 1
                
                # Curvature: fewer neighbors = higher curvature = stronger force
                curvature = (8 - fluid_count) / 8.0
                
                if curvature > 0:
                    # Temperature adjustment
                    T = temp[y, x]
                    T_ref = 273.15
                    temp_factor = max(0.1, 1.0 - (T - T_ref) / 1000.0)
                    
                    # Cohesive force magnitude
                    force = sigma_base * temp_factor * curvature * cell_size / 1000.0  # Scale down
                    
                    interface_data.append({
                        'y': y, 'x': x,
                        'force': force,
                        'curvature': curvature,
                        'neighbors': fluid_count
                    })
            
            # Sort by curvature (process highest curvature first)
            interface_data.sort(key=lambda d: d['curvature'], reverse=True)
            
            # Process swaps for this pass
            swaps_this_pass = 0
            
            for data in interface_data[:max_swaps_per_pass]:
                y, x = data['y'], data['x']
                
                # Verify still fluid (might have been swapped)
                if mt[y, x] not in fluid_types:
                    continue
                
                # For high curvature cells, find best neighbor to swap with
                # Prefer swapping toward other fluid cells to create compact shapes
                best_swap = None
                best_score = -float('inf')
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                            
                        ny, nx = y + dy, x + dx
                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                            
                        # Only swap with space
                        if mt[ny, nx] != MaterialType.SPACE:
                            continue
                        
                        # Count how many fluid neighbors this space cell has
                        # (excluding the source cell)
                        space_fluid_neighbors = 0
                        for ddy in [-1, 0, 1]:
                            for ddx in [-1, 0, 1]:
                                if ddy == 0 and ddx == 0:
                                    continue
                                nny, nnx = ny + ddy, nx + ddx
                                if 0 <= nny < h and 0 <= nnx < w:
                                    if mt[nny, nnx] in fluid_types and not (nny == y and nnx == x):
                                        space_fluid_neighbors += 1
                        
                        # Score: prefer positions with more fluid neighbors
                        # This creates more compact shapes
                        score = space_fluid_neighbors - 0.1 * (abs(dy) + abs(dx))  # Slight preference for cardinal
                        
                        if score > best_score:
                            best_score = score
                            best_swap = (ny, nx)
                
                # Perform swap if beneficial
                if best_swap and data['curvature'] > 0.2:  # Only swap if significant curvature
                    ty, tx = best_swap
                    
                    # Check force against minimal threshold
                    if data['force'] > 1.0:  # Minimal force threshold
                        self._swap_cells(y, x, ty, tx, mt, temp)
                        swaps_this_pass += 1
                        swaps_performed += 1
                        
                        # Update fluid mask for this swap
                        fluid_mask[y, x] = False
                        fluid_mask[ty, tx] = True
            
            # Break if no swaps in this pass
            if swaps_this_pass == 0:
                break
        
        # Mark properties dirty if any swaps
        if swaps_performed > 0:
            self.sim._properties_dirty = True
            
        return swaps_performed

    def identify_rigid_groups(self):
        """Identify connected components of rigid materials (ice, rock, etc.)
        
        Returns a label array where each connected component has a unique ID.
        """
        from materials import MaterialType
        from scipy import ndimage
        
        # Define rigid materials that should move as groups
        rigid_types = {
            MaterialType.ICE,
            MaterialType.GRANITE, MaterialType.BASALT, MaterialType.SANDSTONE,
            MaterialType.SHALE, MaterialType.PUMICE, MaterialType.OBSIDIAN,
            MaterialType.ANDESITE, MaterialType.LIMESTONE, MaterialType.CONGLOMERATE,
            MaterialType.GNEISS, MaterialType.SCHIST, MaterialType.SLATE,
            MaterialType.MARBLE, MaterialType.QUARTZITE
        }
        
        # Create mask of rigid materials
        rigid_mask = np.zeros(self.sim.material_types.shape, dtype=bool)
        for mat_type in rigid_types:
            rigid_mask |= (self.sim.material_types == mat_type)
        
        # Label connected components
        labels, num_features = ndimage.label(rigid_mask)
        
        return labels, num_features
    
    def apply_group_dynamics(self):
        """Apply physics to connected groups of rigid materials.
        
        This allows icebergs, rock formations, etc. to move as coherent units
        while maintaining their shape.
        """
        # Get rigid body groups
        labels, num_groups = self.identify_rigid_groups()
        
        if num_groups == 0:
            return
        
        # For each group, calculate net force and apply motion
        for group_id in range(1, num_groups + 1):
            group_mask = (labels == group_id)
            
            # Skip very small groups
            group_size = np.sum(group_mask)
            if group_size < 4:  # Less than 2x2 block
                continue
            
            # Calculate group properties
            group_coords = np.where(group_mask)
            # Use cell_depth for proper 3D mass calculation
            cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
            group_mass = np.sum(self.sim.density[group_mask]) * cell_volume
            
            # Calculate center of mass of the group
            com_y = np.average(group_coords[0], weights=self.sim.density[group_mask])
            com_x = np.average(group_coords[1], weights=self.sim.density[group_mask])
            
            # Calculate net force on group
            if hasattr(self.sim, 'force_x') and hasattr(self.sim, 'force_y'):
                net_force_x = np.sum(self.sim.force_x[group_mask])
                net_force_y = np.sum(self.sim.force_y[group_mask])
            else:
                continue
            
            # Calculate net buoyancy (important for floating ice)
            buoyancy_y = 0
            for y, x in zip(*group_coords):
                # Check if cell is adjacent to fluid
                neighbors = [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]
                for ny, nx in neighbors:
                    if (0 <= ny < self.sim.height and 0 <= nx < self.sim.width and
                        self.sim.material_types[ny, nx] == MaterialType.WATER):
                        # Simplified buoyancy based on density difference
                        water_density = self.sim.density[ny, nx]
                        ice_density = self.sim.density[y, x]
                        if water_density > ice_density:
                            # Use cell_depth for proper 3D force calculation
                            cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
                            buoyancy_y -= (water_density - ice_density) * 9.81 * cell_volume
            
            net_force_y += buoyancy_y
            
            # Update group velocity
            if group_mass > 0:
                accel_x = net_force_x / group_mass
                accel_y = net_force_y / group_mass
                
                # Update velocities for all cells in group
                dt = 1.0  # 1 second substep
                self.velocity_x[group_mask] += accel_x * dt
                self.velocity_y[group_mask] += accel_y * dt
                
                # Apply damping to prevent instability
                self.velocity_x[group_mask] *= 0.95
                self.velocity_y[group_mask] *= 0.95
                
                # Check if group can move (simplified - just check if velocity is significant)
                avg_vel_x = np.mean(self.velocity_x[group_mask])
                avg_vel_y = np.mean(self.velocity_y[group_mask])
                
                if np.abs(avg_vel_x) > 0.1 or np.abs(avg_vel_y) > 0.1:
                    # Attempt to move entire group
                    self._attempt_group_move(group_mask, avg_vel_x, avg_vel_y)
    
    def _attempt_group_move(self, group_mask, vel_x, vel_y):
        """Attempt to move an entire group of cells based on velocity.
        
        This maintains rigid body coherence by moving all cells together.
        """
        # Determine movement direction (1 cell at a time for stability)
        move_x = int(np.sign(vel_x)) if np.abs(vel_x) > 0.5 else 0
        move_y = int(np.sign(vel_y)) if np.abs(vel_y) > 0.5 else 0
        
        if move_x == 0 and move_y == 0:
            return
        
        # Get all cells in the group
        group_cells = np.where(group_mask)
        src_y = group_cells[0]
        src_x = group_cells[1]
        
        # Calculate target positions
        tgt_y = src_y + move_y
        tgt_x = src_x + move_x
        
        # Check if all target positions are valid and empty (or fluid)
        valid_move = True
        for sy, sx, ty, tx in zip(src_y, src_x, tgt_y, tgt_x):
            # Skip if target is out of bounds
            if ty < 0 or ty >= self.sim.height or tx < 0 or tx >= self.sim.width:
                valid_move = False
                break
            
            # Check if target is empty or fluid (can be displaced)
            tgt_mat = self.sim.material_types[ty, tx]
            if tgt_mat not in {MaterialType.SPACE, MaterialType.AIR, MaterialType.WATER}:
                # Check if target is part of same group (internal movement)
                if not group_mask[ty, tx]:
                    valid_move = False
                    break
        
        if valid_move:
            # Perform the group move
            # First, store all source values
            src_materials = self.sim.material_types[src_y, src_x].copy()
            src_temps = self.sim.temperature[src_y, src_x].copy()
            src_ages = self.sim.age[src_y, src_x].copy()
            src_vx = self.velocity_x[src_y, src_x].copy()
            src_vy = self.velocity_y[src_y, src_x].copy()
            
            # Clear source positions
            self.sim.material_types[src_y, src_x] = MaterialType.SPACE
            self.sim.temperature[src_y, src_x] = self.sim.space_temperature
            self.velocity_x[src_y, src_x] = 0
            self.velocity_y[src_y, src_x] = 0
            
            # Set target positions
            self.sim.material_types[tgt_y, tgt_x] = src_materials
            self.sim.temperature[tgt_y, tgt_x] = src_temps
            self.sim.age[tgt_y, tgt_x] = src_ages
            self.velocity_x[tgt_y, tgt_x] = src_vx
            self.velocity_y[tgt_y, tgt_x] = src_vy
            
            # Mark properties dirty
            self.sim._properties_dirty = True

    def _swap_cells(self, y1, x1, y2, x2, mt, temp):
        """Helper to swap two cells preserving all properties"""
        # Swap materials
        mt[y1, x1], mt[y2, x2] = mt[y2, x2], mt[y1, x1]
        
        # Swap temperatures
        temp[y1, x1], temp[y2, x2] = temp[y2, x2], temp[y1, x1]
        
        # Swap velocities
        if hasattr(self, 'velocity_x') and hasattr(self, 'velocity_y'):
            vx_temp = self.velocity_x[y1, x1]
            vy_temp = self.velocity_y[y1, x1]
            self.velocity_x[y1, x1] = self.velocity_x[y2, x2]
            self.velocity_y[y1, x1] = self.velocity_y[y2, x2]
            self.velocity_x[y2, x2] = vx_temp
            self.velocity_y[y2, x2] = vy_temp
        
        # Swap ages
        if hasattr(self.sim, 'age'):
            age_temp = self.sim.age[y1, x1]
            self.sim.age[y1, x1] = self.sim.age[y2, x2]
            self.sim.age[y2, x2] = age_temp

    def _compute_surface_tension_pressure(self):
        """Compute surface tension pressure source term for fluids at interfaces.
        
        Returns pressure source (Pa) that creates cohesive forces to minimize surface area.
        """
        from materials import MaterialType
        
        # Surface tension coefficient (Pa·m) - much stronger than real water for discrete cells
        # Real water: ~0.072 N/m, but we need much stronger effect for discrete cells
        sigma = 50000.0  # Pa·m (increased from 5000.0 for much stronger effect)
        
        dx = self.sim.cell_size
        mt = self.sim.material_types
        rho = self.sim.density
        
        # Define fluid materials that should have surface tension
        fluid_types = {MaterialType.WATER, MaterialType.MAGMA}
        
        pressure_source = np.zeros_like(rho)
        
        # For each fluid cell, compute interface curvature and add cohesive pressure
        for y in range(1, self.sim.height - 1):
            for x in range(1, self.sim.width - 1):
                if mt[y, x] not in fluid_types:
                    continue
                
                # Count neighbors with significantly lower density (interface detection)
                neighbors = [
                    (y-1, x), (y+1, x), (y, x-1), (y, x+1)  # 4-connected
                ]
                
                interface_count = 0
                for ny, nx in neighbors:
                    if rho[ny, nx] < 0.5 * rho[y, x]:  # Interface with much lighter material
                        interface_count += 1
                
                # Add cohesive pressure proportional to interface exposure
                # More exposed cells (more interfaces) get higher internal pressure
                if interface_count > 0:
                    # Pressure scales with interface count and surface tension
                    # Factor of 2/dx gives correct dimensional scaling (Pa·m / m = Pa)
                    pressure_source[y, x] = sigma * interface_count * 2.0 / dx
        
        return pressure_source
