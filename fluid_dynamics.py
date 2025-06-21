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
        self.dv_thresh = 0.01  # m/s – velocity-difference threshold for swapping
        # Scale binding force appropriately for cell size (100m cells = 1e6 m³ volume)
        # Rock binding ~1-10 MPa, so binding force density should be pressure / length scale
        # For 100m cells: 1 MPa / 100m = 1e6 Pa / 100m = 1e4 N/m³
        # But we need to scale by actual cell size
        # For 50m cells with proper scaling: reduced to allow motion
        self.solid_binding_force = 1e-2  # N/m³ – reference binding force density between solid voxels
        self.velocity_clamp = True # Enable/disable velocity clamping (enable if instability)
        self.velocity_threshold = True  # Enable/disable velocity threshold for swapping

        # Create binding matrix using material processes helper
        try:
            from .material_processes import MaterialProcesses
        except ImportError:
            from material_processes import MaterialProcesses
        self._binding_matrix, self._mat_index = MaterialProcesses.create_binding_matrix(self.solid_binding_force)
    
    def calculate_planetary_pressure(self):
        """Multigrid solve for pressure using both self-gravity and external gravity fields."""

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

        # Divergence of total gravity field
        div_g = np.zeros_like(gx_total)
        dx = self.sim.cell_size
        div_g[1:-1, 1:-1] = (
            (gx_total[1:-1, 2:] - gx_total[1:-1, :-2]) + (gy_total[2:, 1:-1] - gy_total[:-2, 1:-1])
        ) / (2 * dx)

        # Build RHS for Poisson equation
        # For external gravity, we need to solve: ∇²P = ∇·(ρg)
        # This handles arbitrary geometries correctly
        rhs = np.zeros_like(self.sim.density)
        
        # Self-gravity contribution: -ρ∇·g_self (if enabled)
        if self.sim.enable_self_gravity:
            rhs -= (self.sim.density * div_g) / 1e6   # Pa → MPa (negative for attractive gravity)
        
        # External gravity contribution: ∇·(ρg_ext)
        # For uniform external gravity, ∇·(ρg_ext) = g_ext·∇ρ since ∇·g_ext = 0
        if hasattr(self.sim, 'external_gravity'):
            g_ext_x, g_ext_y = self.sim.external_gravity
            if abs(g_ext_x) > 1e-10 or abs(g_ext_y) > 1e-10:
                # For uniform external gravity field: ∇·(ρg) = g·∇ρ
                grad_rho_x = np.zeros_like(self.sim.density)
                grad_rho_y = np.zeros_like(self.sim.density)
                
                # Central differences for density gradient
                grad_rho_x[1:-1, 1:-1] = (self.sim.density[1:-1, 2:] - self.sim.density[1:-1, :-2]) / (2 * dx)
                grad_rho_y[1:-1, 1:-1] = (self.sim.density[2:, 1:-1] - self.sim.density[:-2, 1:-1]) / (2 * dx)
                
                # ∇·(ρg_ext) = g_ext·∇ρ
                div_rho_g = g_ext_x * grad_rho_x + g_ext_y * grad_rho_y
                
                # Add to RHS (convert Pa to MPa)
                rhs += div_rho_g / 1e6

        # Solve Poisson equation for pressure: ∇²P = RHS
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
        
        # Apply group dynamics for rigid bodies (ice, rock)
        self.apply_group_dynamics()
        
        # First handle rigid body group movements
        self.apply_rigid_body_movements()
        
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

        # NOTE: No explicit buoyancy term is added here.
        # The body-force term ρ g together with the projection/pressure solve
        # already reproduces Archimedean buoyancy: lighter cells accelerate
        # upward because they are pulled down less than their heavier
        # neighbours, and the incompressibility projection converts that
        # imbalance into a pressure field that pushes the light phase up.
        # Adding another (ρ_ref − ρ) g term would double-count the physics.

        

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
        
        # Get rigid body labels to avoid breaking them apart
        labels, _ = self.identify_rigid_groups()
        
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
                # Skip if this cell is part of a rigid body group
                if labels[y, x] > 0:
                    continue
                    
                # Don't skip any cells - SPACE needs to participate in swaps too
                    
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    
                    # Skip if target is part of a rigid body group (unless same group)
                    if labels[ny, nx] > 0 and labels[ny, nx] != labels[y, x]:
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
                    # Check material's self-binding
                    src_bind = self._binding_threshold(mt[y, x], mt[y, x], temp[y, x])
                    
                    # Directional force projection (force of source along neighbour direction)
                    proj_src = fsrc_x * dx + fsrc_y * dy
                    
                    # Directional requirement: source force must point toward the neighbour cell
                    cond_direction = proj_src > 0
                    
                    # Check conditions for swapping:
                    # 1. Net force must overcome material binding threshold
                    # 2. Velocity difference must be significant (if threshold enabled)
                    # 3. For solids, source must overcome its own binding
                    cond_force = F_net > threshold
                    # Use velocity threshold only if enabled, but relax for fluid ↔ space/fluid interactions
                    if self.velocity_threshold:
                        fluid_set = {MaterialType.WATER, MaterialType.MAGMA, MaterialType.AIR, MaterialType.WATER_VAPOR, MaterialType.SPACE}
                        if mt[y, x] in fluid_set or mt[ny, nx] in fluid_set:
                            # No velocity gate for fluid/space interactions
                            cond_velocity = True
                        else:
                            cond_velocity = V_diff >= self.dv_thresh
                    else:
                        cond_velocity = True
                    
                    
                    # Determine if source can overcome its own binding (solids)
                    if src_bind > 0:
                        cond_src = abs(proj_src) > src_bind
                    else:
                        cond_src = True
                    
                    if cond_force and cond_velocity and cond_src and cond_direction:
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

    def identify_rigid_groups(self):
        """Identify connected components of rigid materials (any non-fluid materials).
        
        This correctly handles metamorphosis - a donut made of granite that 
        partially metamorphoses into other rock types will still be considered
        one rigid body as long as the materials are solid (non-fluid).
        
        Returns a label array where each connected component has a unique ID.
        """
        from materials import MaterialType
        from scipy import ndimage
        
        # Define fluid materials that should NOT move as rigid groups
        fluid_types = {
            MaterialType.MAGMA,
            MaterialType.WATER,
            MaterialType.WATER_VAPOR, 
            MaterialType.AIR,
            MaterialType.SPACE
        }
        
        # All non-fluid materials are considered rigid
        # This includes rocks, sediments, metamorphic rocks, ice, etc.
        rigid_mask = np.ones(self.sim.material_types.shape, dtype=bool)
        for mat_type in fluid_types:
            rigid_mask &= (self.sim.material_types != mat_type)
        
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
            
            # Don't skip small groups - even single cells should fall
            group_size = np.sum(group_mask)
            if group_size == 0:
                continue
            
            # Calculate group properties
            group_coords = np.where(group_mask)
            # Use cell_depth for proper 3D mass calculation
            cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
            
            # Get material densities for the group
            group_densities = np.zeros(group_size)
            for i, (y, x) in enumerate(zip(*group_coords)):
                mat = self.sim.material_types[y, x]
                mat_props = self.sim.material_db.get_properties(mat)
                group_densities[i] = mat_props.density
            
            group_mass = np.sum(group_densities) * cell_volume
            
            # Calculate center of mass of the group
            com_y = np.average(group_coords[0], weights=group_densities)
            com_x = np.average(group_coords[1], weights=group_densities)
            
            # Calculate net force on group using force density
            if hasattr(self.sim, 'force_x') and hasattr(self.sim, 'force_y'):
                # Force density is already in N/m³, multiply by cell volume for total force
                net_force_x = np.sum(self.sim.force_x[group_mask]) * cell_volume
                net_force_y = np.sum(self.sim.force_y[group_mask]) * cell_volume
            else:
                continue
            
            # Update group velocity
            if group_mass > 0:
                accel_x = net_force_x / group_mass
                accel_y = net_force_y / group_mass
                
                # Update velocities for all cells in group
                dt = 1.0  # 1 second substep
                self.velocity_x[group_mask] += accel_x * dt
                self.velocity_y[group_mask] += accel_y * dt
                
                # Apply damping to prevent instability (reduced from 0.95)
                damping = 0.98  # Less damping to allow motion
                self.velocity_x[group_mask] *= damping
                self.velocity_y[group_mask] *= damping
                
                # Get average velocity for the group
                avg_vel_x = np.mean(self.velocity_x[group_mask])
                avg_vel_y = np.mean(self.velocity_y[group_mask])
                
                # Debug output for testing
                if abs(net_force_x) > 0 or abs(net_force_y) > 0:
                    if getattr(self.sim, 'debug_rigid_bodies', False):
                        print(f"Group {group_id}: size={group_size}, F=({net_force_x:.1e},{net_force_y:.1e}), v=({avg_vel_x:.2f},{avg_vel_y:.2f})")
    
    def apply_rigid_body_movements(self):
        """Move rigid body groups as coherent units based on net forces.
        
        This ensures rocks, ice, etc. fall and move as single objects rather than
        breaking apart cell by cell. Now includes fluid displacement mechanics.
        """
        # Get rigid body groups
        labels, num_groups = self.identify_rigid_groups()
        
        if num_groups == 0:
            return
            
        mt = self.sim.material_types
        h, w = mt.shape
        
        # Track which cells have already been moved to avoid double-swapping
        moved_mask = np.zeros((h, w), dtype=bool)
        
        # Process each rigid group
        for group_id in range(1, num_groups + 1):
            group_mask = (labels == group_id)
            
            # Skip if any cell in group has already moved
            if np.any(moved_mask[group_mask]):
                continue
                
            group_coords = np.where(group_mask)
            group_size = len(group_coords[0])
            
            if group_size == 0:
                continue
            
            # Get material densities and calculate total force on group
            cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
            total_force_x = 0.0
            total_force_y = 0.0
            total_mass = 0.0
            
            for y, x in zip(*group_coords):
                # Get force density and convert to total force
                force_x = self.sim.force_x[y, x] * cell_volume
                force_y = self.sim.force_y[y, x] * cell_volume
                total_force_x += force_x
                total_force_y += force_y
                
                # Get mass
                mat = mt[y, x]
                mat_props = self.sim.material_db.get_properties(mat)
                mass = mat_props.density * cell_volume
                total_mass += mass
            
            # Skip if no significant net force
            force_magnitude = np.hypot(total_force_x, total_force_y)
            if force_magnitude < 1e-10:
                continue
                
            # Determine movement direction based on force
            # For discrete grid, we can only move in 8 directions
            force_angle = np.arctan2(total_force_y, total_force_x)
            
            # Quantize to nearest 45-degree direction
            directions = [
                (1, 0),   # East
                (1, 1),   # Southeast  
                (0, 1),   # South
                (-1, 1),  # Southwest
                (-1, 0),  # West
                (-1, -1), # Northwest
                (0, -1),  # North
                (1, -1),  # Northeast
            ]
            
            # Find best direction
            best_dir = None
            best_dot = -1
            force_unit_x = total_force_x / force_magnitude
            force_unit_y = total_force_y / force_magnitude
            
            for dx, dy in directions:
                dot = dx * force_unit_x + dy * force_unit_y
                if dot > best_dot:
                    best_dot = dot
                    best_dir = (dx, dy)
            
            if best_dir is None or best_dot < 0.5:  # Require reasonable alignment
                continue
                
            dx, dy = best_dir
            
            # First detect any enclosed fluids that must move with the rigid body
            enclosed_fluids = self.detect_enclosed_fluids(group_mask)
            
            # Try to move with fluid displacement
            success = self.attempt_rigid_body_displacement(
                group_mask, dx, dy, force_magnitude, moved_mask, enclosed_fluids
            )
            
            if not success:
                continue
            
            if getattr(self.sim, 'debug_rigid_bodies', False):
                enclosed_count = np.sum(enclosed_fluids) if enclosed_fluids is not None else 0
                total_moved = group_size + enclosed_count
                print(f"Moved rigid body group {group_id} ({group_size} rigid + {enclosed_count} contained = {total_moved} total cells) by ({dx}, {dy})")
    
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

    
    
    
    def detect_enclosed_fluids(self, rigid_body_mask):
        """Detect fluid regions that are topologically enclosed by a rigid body.
        
        Uses flood-fill to identify disconnected fluid regions.
        Returns a mask of fluids that should move with the rigid body.
        """
        from materials import MaterialType
        h, w = self.sim.material_types.shape
        
        # Define fluid types
        fluid_types = {MaterialType.WATER, MaterialType.MAGMA, MaterialType.AIR, MaterialType.WATER_VAPOR}
        
        # Create fluid mask
        fluid_mask = np.zeros((h, w), dtype=bool)
        for fluid_type in fluid_types:
            fluid_mask |= (self.sim.material_types == fluid_type)
        
        # Find boundary of rigid body (cells adjacent to rigid body)
        boundary_mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if rigid_body_mask[y, x]:
                    continue
                # Check if adjacent to rigid body
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and rigid_body_mask[ny, nx]:
                        boundary_mask[y, x] = True
                        break
        
        # Flood fill from edges to find external fluids
        external_fluid = np.zeros((h, w), dtype=bool)
        
        # Start flood fill from all edge cells that are fluid or space
        edge_seeds = []
        # Top and bottom edges
        for x in range(w):
            if not rigid_body_mask[0, x] and not boundary_mask[0, x]:
                edge_seeds.append((0, x))
            if not rigid_body_mask[h-1, x] and not boundary_mask[h-1, x]:
                edge_seeds.append((h-1, x))
        # Left and right edges
        for y in range(h):
            if not rigid_body_mask[y, 0] and not boundary_mask[y, 0]:
                edge_seeds.append((y, 0))
            if not rigid_body_mask[y, w-1] and not boundary_mask[y, w-1]:
                edge_seeds.append((y, w-1))
        
        # Flood fill to mark all externally connected regions
        visited = set()
        queue = edge_seeds[:]
        
        while queue:
            y, x = queue.pop(0)
            if (y, x) in visited:
                continue
            visited.add((y, x))
            external_fluid[y, x] = True
            
            # Check neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and 
                    not rigid_body_mask[ny, nx] and 
                    (ny, nx) not in visited):
                    queue.append((ny, nx))
        
        # Enclosed fluids are fluids that are not externally connected
        enclosed_fluid_mask = fluid_mask & ~external_fluid
        
        return enclosed_fluid_mask
    
    def attempt_rigid_body_displacement(self, group_mask, dx, dy, force_magnitude, 
                                        moved_mask, enclosed_fluids=None):
        """Attempt to move a rigid body group with proper fluid displacement.
        
        Returns True if movement succeeded, False otherwise.
        """
        mt = self.sim.material_types
        h, w = mt.shape
        
        group_coords = np.where(group_mask)
        group_size = len(group_coords[0])
        
        # Include enclosed fluids in the moving group
        if enclosed_fluids is not None:
            combined_mask = group_mask | enclosed_fluids
            combined_coords = np.where(combined_mask)
        else:
            combined_mask = group_mask
            combined_coords = group_coords
        
        # Check all target positions
        target_positions = []
        fluids_to_displace = []
        
        for y, x in zip(*combined_coords):
            ny, nx = y + dy, x + dx
            
            # Check bounds
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                return False  # Can't move out of bounds
            
            target_positions.append((ny, nx))
            
            # Skip if target is part of same combined group
            if combined_mask[ny, nx]:
                continue
            
            # Check what's at target position
            target_mat = mt[ny, nx]
            if target_mat == MaterialType.SPACE:
                continue  # Empty space, can move freely
            
            # Define displaceable fluids
            displaceable_fluids = {MaterialType.WATER, MaterialType.AIR, MaterialType.WATER_VAPOR, MaterialType.MAGMA}
            
            # It's a fluid or another material - add to displacement list if it's displaceable
            if target_mat in displaceable_fluids and not moved_mask[ny, nx]:
                fluids_to_displace.append((ny, nx))
        
        # If there are fluids to displace, try to displace them
        if fluids_to_displace:
            # Calculate displacement pressure based on force
            cell_volume = self.sim.cell_size ** 2 * self.sim.cell_depth
            displacement_pressure = force_magnitude / (group_size * self.sim.cell_size ** 2)
            
            # Attempt to displace fluids
            if not self.displace_fluids(fluids_to_displace, -dx, -dy, displacement_pressure):
                return False  # Couldn't displace fluids
        
        # Movement is possible - execute it
        # Store all data from combined group
        src_data = []
        for y, x in zip(*combined_coords):
            src_data.append({
                'material': mt[y, x],
                'temperature': self.sim.temperature[y, x],
                'age': self.sim.age[y, x],
                'vx': self.velocity_x[y, x],
                'vy': self.velocity_y[y, x]
            })
        
        # Clear source positions
        for y, x in zip(*combined_coords):
            mt[y, x] = MaterialType.SPACE
            self.sim.temperature[y, x] = self.sim.space_temperature
            self.velocity_x[y, x] = 0
            self.velocity_y[y, x] = 0
            moved_mask[y, x] = True
        
        # Place at target positions
        for i, (ny, nx) in enumerate(target_positions):
            data = src_data[i]
            mt[ny, nx] = data['material']
            self.sim.temperature[ny, nx] = data['temperature']
            self.sim.age[ny, nx] = data['age']
            self.velocity_x[ny, nx] = data['vx']
            self.velocity_y[ny, nx] = data['vy']
            moved_mask[ny, nx] = True
        
        # Transfer momentum to displaced fluids
        if fluids_to_displace:
            # Calculate average rigid body velocity
            avg_vx = np.mean([d['vx'] for d in src_data[:group_size]])
            avg_vy = np.mean([d['vy'] for d in src_data[:group_size]])
            
            # Apply momentum transfer (partial, with damping)
            momentum_transfer = 0.5  # Transfer 50% of velocity
            for fy, fx in fluids_to_displace:
                # Find where the fluid ended up after displacement
                new_fy, new_fx = fy - dy, fx - dx
                if 0 <= new_fy < h and 0 <= new_fx < w:
                    self.velocity_x[new_fy, new_fx] += avg_vx * momentum_transfer
                    self.velocity_y[new_fy, new_fx] += avg_vy * momentum_transfer
        
        # Mark properties dirty
        self.sim._properties_dirty = True
        
        return True
    
    def displace_fluids(self, fluid_positions, disp_dx, disp_dy, pressure):
        """Displace fluids in response to rigid body motion.
        
        Uses simple swapping approach for now to ensure conservation.
        Returns True if displacement succeeded, False otherwise.
        """
        mt = self.sim.material_types
        h, w = mt.shape
        
        # For now, just try to move fluids out of the way using simple swaps
        # This ensures material conservation
        success = True
        
        for fy, fx in fluid_positions:
            # Try to find an empty space nearby
            found_space = False
            
            # Search in expanding rings
            for radius in range(1, max(h, w)):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        # Skip if not on ring perimeter
                        if abs(dy) != radius and abs(dx) != radius:
                            continue
                            
                        ny, nx = fy + dy, fx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mt[ny, nx] == MaterialType.SPACE:
                                # Swap fluid with space
                                mt[ny, nx] = mt[fy, fx]
                                self.sim.temperature[ny, nx] = self.sim.temperature[fy, fx]
                                self.sim.age[ny, nx] = self.sim.age[fy, fx]
                                self.velocity_x[ny, nx] = self.velocity_x[fy, fx]
                                self.velocity_y[ny, nx] = self.velocity_y[fy, fx]
                                
                                mt[fy, fx] = MaterialType.SPACE
                                self.sim.temperature[fy, fx] = self.sim.space_temperature
                                self.sim.age[fy, fx] = self.sim.age[0, 0]
                                self.velocity_x[fy, fx] = 0
                                self.velocity_y[fy, fx] = 0
                                
                                found_space = True
                                break
                    if found_space:
                        break
                if found_space:
                    break
            
            if not found_space:
                success = False
        
        return success
    
