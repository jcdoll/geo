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
        self.dv_thresh = 0.001  # m/s – velocity-difference threshold for swapping (reduced for surface tension)
        self.solid_binding_force = 2e-4  # N – reference cohesion between solid voxels
        # ------------------------------------------------------------------
        # Pre-compute binding-threshold lookup table for fast vectorised access
        # ------------------------------------------------------------------
        mt_list = list(MaterialType)
        n_mat = len(mt_list)
        self._binding_matrix = np.zeros((n_mat, n_mat), dtype=np.float32)

        # Map MaterialType to index for fast lookup
        self._mat_index = {m: i for i, m in enumerate(mt_list)}

        # Helper sets for readability
        fluid_set = {MaterialType.AIR, MaterialType.WATER_VAPOR, MaterialType.WATER, MaterialType.MAGMA}
        for i, a in enumerate(mt_list):
            for j, b in enumerate(mt_list):
                # Fluids set includes SPACE
                if (a in fluid_set) and (b in fluid_set):
                    self._binding_matrix[i, j] = 0.0
                elif (a in fluid_set) ^ (b in fluid_set):  # fluid–solid (SPACE counts as fluid)
                    self._binding_matrix[i, j] = 0.5 * self.solid_binding_force
                else:  # solid–solid
                    self._binding_matrix[i, j] = self.solid_binding_force
    
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
        """

        if not hasattr(self.sim, "_perf_times"):
            self.sim._perf_times = {}
        _t0 = _time.perf_counter()

        # ------------------------------------------------------------------
        # 1) Assemble net body-force field  f = ρ g  − ∇p  (N·m⁻³)
        # ------------------------------------------------------------------
        rho = self.sim.density

        _t_force_start = _time.perf_counter()
        fx, fy = self.compute_force_field()
        self.sim._perf_times["uk_force_assembly"] = _time.perf_counter() - _t_force_start

        # ------------------------------------------------------------------
        # 2) Velocity update  u ← u + (f/ρ) Δt
        # ------------------------------------------------------------------
        _t_vel_start = _time.perf_counter()
        dt_seconds = dt  # dt already in seconds
        accel_x = np.where(rho > 0, fx / rho, 0.0)
        accel_y = np.where(rho > 0, fy / rho, 0.0)

        # Velocity update and CFL clamp (≤0.5 cell/step)
        self.velocity_x += accel_x * dt_seconds
        self.velocity_y += accel_y * dt_seconds

        max_u = 0.5 * self.sim.cell_size / dt_seconds
        np.clip(self.velocity_x, -max_u, max_u, out=self.velocity_x)
        np.clip(self.velocity_y, -max_u, max_u, out=self.velocity_y)

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
        rhs = (div_u / dt_seconds) * fluid_mask
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
        self.velocity_x -= grad_phi_x * dt_seconds * inv_rho
        self.velocity_y -= grad_phi_y * dt_seconds * inv_rho

        self.sim._perf_times["uk_projection"] = _time.perf_counter() - _t_proj_start

        # ------------------------------------------------------------------
        # 4) Accumulate displacement and determine swaps via force criteria
        # ------------------------------------------------------------------
        # Instead of density-based heavier-than-target rule we now use the
        # force-based binding criterion defined in `apply_force_based_swapping`.
        _t_swap_force_start = _time.perf_counter()
        
        # Apply bulk surface tension first for rapid interface evolution
        surface_swaps = self.apply_bulk_surface_tension()
        
        # Apply group dynamics for rigid bodies (ice, rock)
        self.apply_group_dynamics()
        
        # Then apply regular force-based swapping for other interactions
        self.apply_force_based_swapping()
        
        self.sim._perf_times["uk_force_swaps"] = _time.perf_counter() - _t_swap_force_start

        # Optional solid drag (retain previous behaviour)
        if getattr(self.sim, "enable_solid_drag", True):
            solid_mask = self.sim._get_solid_mask()
            self.velocity_x[solid_mask] *= 0.2
            self.velocity_y[solid_mask] *= 0.2

        swaps_done = 0  # track additional swaps in sinking passes

        # ------------------------------------------------------------------
        # 5) Extra sinking passes – ensure heavier fluid settles through lighter
        # ------------------------------------------------------------------
        # Gravity-aligned deterministic settling passes
        n_settle = 3
        for _ in range(n_settle):
            gx = self.sim.gravity_x
            gy = self.sim.gravity_y

            abs_gx = np.abs(gx)
            abs_gy = np.abs(gy)

            dx_dir = np.where(abs_gx >= abs_gy, np.sign(gx).astype(np.int8), 0)
            dy_dir = np.where(abs_gy > abs_gx, np.sign(gy).astype(np.int8), 0)

            # Compute target indices and filter in-bounds
            yy, xx = np.indices(rho.shape)
            tgt_y = (yy + dy_dir).astype(np.int32)
            tgt_x = (xx + dx_dir).astype(np.int32)
            in_bounds = (tgt_y >= 0) & (tgt_y < self.sim.height) & (tgt_x >= 0) & (tgt_x < self.sim.width)

            src_y = yy[in_bounds].ravel(); src_x = xx[in_bounds].ravel()
            tgt_y = tgt_y[in_bounds].ravel(); tgt_x = tgt_x[in_bounds].ravel()

            # Only perform swap if target is NOT SPACE and source is heavier
            mt_src_flat = self.sim.material_types[src_y, src_x]
            mt_tgt_flat = self.sim.material_types[tgt_y, tgt_x]
            heavier = (rho[src_y, src_x] > rho[tgt_y, tgt_x] + 1e-3)
            if not np.any(heavier):
                break
            self._perform_material_swaps(src_y[heavier], src_x[heavier], tgt_y[heavier], tgt_x[heavier])
            swaps_done += int(np.sum(heavier))

            # Refresh density for subsequent passes
            self.sim._properties_dirty = True
            self.sim._update_material_properties()
            rho = self.sim.density  # view updated array
    
    def calculate_effective_density(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate effective density including thermal expansion"""
        # Base density from material properties
        base_density = self.sim.density.copy()
        
        # Apply thermal expansion
        thermal_expansion = np.zeros_like(temperature)
        
        for y in range(self.sim.height):
            for x in range(self.sim.width):
                if self.sim.material_types[y, x] == MaterialType.SPACE:
                    continue
                
                material_props = self.sim.material_db.get_properties(self.sim.material_types[y, x])
                expansion_coeff = getattr(material_props, 'thermal_expansion', 1e-5)
                
                # Effective density with thermal expansion
                temp_diff = temperature[y, x] - self.sim.reference_temperature
                expansion_factor = 1.0 + expansion_coeff * temp_diff
                thermal_expansion[y, x] = expansion_factor
        
        # Avoid division by zero
        thermal_expansion = np.maximum(thermal_expansion, 0.1)
        
        return base_density / thermal_expansion
    
    def calculate_effective_density_single(self, y: int, x: int) -> float:
        """Calculate effective density for a single cell"""
        if self.sim.material_types[y, x] == MaterialType.SPACE:
            return 0.0
        
        material_props = self.sim.material_db.get_properties(self.sim.material_types[y, x])
        expansion_coeff = getattr(material_props, 'thermal_expansion', 1e-5)
        
        temp_diff = self.sim.temperature[y, x] - self.sim.reference_temperature
        expansion_factor = max(1.0 + expansion_coeff * temp_diff, 0.1)
        
        return self.sim.density[y, x] / expansion_factor
    
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
            cell_volume = self.sim.cell_size ** 2  # 2D simulation
            
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
        """Assemble net body force F = ρ g − ∇P and store on the simulation."""
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

        # Gravity term
        fx = rho * gx_total
        fy = rho * gy_total

        # Pressure gradient term – compute via face normals (4-neighbour)
        P_pa = self.sim.pressure * 1e6  # MPa → Pa
        dx = self.sim.cell_size

        # Store pressure forces separately for debugging
        fx_pressure = np.zeros_like(fx)
        fy_pressure = np.zeros_like(fy)

        # X-direction forces
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
        Uses precomputed matrix plus simple temperature weakening for solids.
        """
        from materials import MaterialType  # local import
        idx_a = self._mat_index[mt_a]
        idx_b = self._mat_index[mt_b]
        base_th = self._binding_matrix[idx_a, idx_b]
        # Temperature weakening only affects bonds where at least one side is solid
        if np.isfinite(base_th) and base_th > 0:
            T_ref = 273.15
            temp_factor = max(0.1, 1.0 - (temp_avg - T_ref) / 1500.0)
            return base_th * temp_factor
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
        _ref_solid = MaterialType.GRANITE
        _cell_max_binding = np.vectorize(lambda m, t: self._binding_threshold(m, _ref_solid, t))

        # Simplified approach: iterate over all cells and check neighbors directly
        for y in range(h):
            for x in range(w):
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    
                    # Only process fluid-space interfaces for surface tension
                    fluid_types_local = {MaterialType.WATER, MaterialType.MAGMA}
                    src_is_fluid = mt[y, x] in fluid_types_local
                    tgt_is_space = mt[ny, nx] == MaterialType.SPACE
                    
                    # For surface tension: only allow fluid->space swaps (outward expansion)
                    if not (src_is_fluid and tgt_is_space):
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
                    
                    # Binding threshold
                    temp_avg = 0.5 * (temp[y, x] + temp[ny, nx])
                    threshold = self._binding_threshold(mt[y, x], mt[ny, nx], temp_avg)
                    
                    # Directional force projection
                    proj_src = fsrc_x * dx + fsrc_y * dy
                    
                    # Binding forces
                    src_bind = self._binding_threshold(mt[y, x], _ref_solid, temp[y, x])
                    
                    # Check all conditions
                    # For surface tension: we want swaps when force points AWAY from target (negative projection)
                    cond_src = abs(proj_src) > src_bind  # Use absolute value - direction doesn't matter for magnitude
                    cond_force = F_net > threshold
                    cond_velocity = V_diff >= self.dv_thresh
                    
                    if cond_src and cond_force and cond_velocity:
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

    def apply_bulk_surface_tension(self):
        """Apply surface tension using bulk interface processing.
        
        This method processes entire fluid-vacuum interfaces simultaneously,
        allowing 50-100+ swaps per timestep instead of the 3 achieved by
        individual cell processing.
        """
        from materials import MaterialType
        from scipy import ndimage
        
        mt = self.sim.material_types
        temp = self.sim.temperature
        h, w = mt.shape
        
        # Identify fluid cells
        fluid_mask = ((mt == MaterialType.WATER) | (mt == MaterialType.MAGMA))
        vacuum_mask = (mt == MaterialType.SPACE)
        
        if not np.any(fluid_mask) or not np.any(vacuum_mask):
            return 0
            
        # Count initial water for conservation check
        initial_water_count = np.sum(fluid_mask)
        
        # Find interface cells - fluid cells adjacent to vacuum
        vacuum_dilated = ndimage.binary_dilation(vacuum_mask, structure=np.ones((3,3)))
        interface_mask = fluid_mask & vacuum_dilated
        
        if not np.any(interface_mask):
            return 0
            
        # Strategy: Create a more circular/compact shape
        # 1. Find and fill gaps
        # 2. Move protruding cells inward
        # 3. Allow lateral movement to round out shapes
        
        swaps_performed = 0
        max_swaps = min(100, len(np.where(interface_mask)[0]))
        
        # Find center of mass of fluid
        fluid_y, fluid_x = np.where(fluid_mask)
        com_y = np.mean(fluid_y)
        com_x = np.mean(fluid_x)
        
        # Phase 1: Fill internal gaps
        water_neighbor_count = ndimage.convolve(
            fluid_mask.astype(float),
            np.ones((3, 3)),
            mode='constant',
            cval=0
        )
        
        gap_mask = vacuum_mask & (water_neighbor_count >= 4)
        gap_cells = np.where(gap_mask)
        
        for i in range(min(len(gap_cells[0]), max_swaps // 3)):
            gy, gx = gap_cells[0][i], gap_cells[1][i]
            
            if mt[gy, gx] != MaterialType.SPACE:
                continue
                
            # Find farthest interface cell from COM to move here
            interface_y, interface_x = np.where(interface_mask)
            if len(interface_y) == 0:
                break
                
            # Use distance from COM to find outlier cells
            distances_com = np.sqrt((interface_y - com_y)**2 + (interface_x - com_x)**2)
            farthest_idx = np.argmax(distances_com)
            sy, sx = interface_y[farthest_idx], interface_x[farthest_idx]
            
            if mt[sy, sx] not in {MaterialType.WATER, MaterialType.MAGMA}:
                continue
                
            # Perform swap
            self._swap_cells(sy, sx, gy, gx, mt, temp)
            swaps_performed += 1
            interface_mask[sy, sx] = False
        
        # Phase 2: Move protruding cells to create rounder shape
        # For horizontal lines, we need to move end cells up/down
        fluid_mask = ((mt == MaterialType.WATER) | (mt == MaterialType.MAGMA))
        interface_y, interface_x = np.where(fluid_mask & vacuum_dilated)
        
        # Process interface cells, prioritizing those far from COM
        for i in range(min(len(interface_y), max_swaps - swaps_performed)):
            y, x = interface_y[i], interface_x[i]
            
            if mt[y, x] not in {MaterialType.WATER, MaterialType.MAGMA}:
                continue
                
            # Count fluid neighbors
            fluid_neighbors = 0
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if mt[ny, nx] in {MaterialType.WATER, MaterialType.MAGMA}:
                        fluid_neighbors += 1
            
            # For cells with few neighbors, try to move them closer to COM
            if fluid_neighbors <= 2:
                # Find best adjacent space to move to
                best_target = None
                min_dist_to_com = float('inf')
                
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    ny, nx = y + dy, x + dx
                    
                    if (0 <= ny < h and 0 <= nx < w and mt[ny, nx] == MaterialType.SPACE):
                        # Calculate distance to COM from this position
                        dist_to_com = np.sqrt((ny - com_y)**2 + (nx - com_x)**2)
                        
                        # Also check if this position would have more fluid neighbors
                        potential_neighbors = 0
                        for ddy, ddx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nny, nnx = ny + ddy, nx + ddx
                            if 0 <= nny < h and 0 <= nnx < w:
                                if mt[nny, nnx] in {MaterialType.WATER, MaterialType.MAGMA}:
                                    potential_neighbors += 1
                        
                        # Prefer positions closer to COM or with more neighbors
                        if dist_to_com < min_dist_to_com or potential_neighbors > fluid_neighbors:
                            best_target = (ny, nx)
                            min_dist_to_com = dist_to_com
                
                if best_target:
                    ty, tx = best_target
                    self._swap_cells(y, x, ty, tx, mt, temp)
                    swaps_performed += 1
        
        # Phase 3: For very elongated shapes, actively reshape
        # Recalculate fluid positions after previous swaps
        fluid_mask = ((mt == MaterialType.WATER) | (mt == MaterialType.MAGMA))
        fluid_cells = np.where(fluid_mask)
        fluid_y, fluid_x = fluid_cells
        
        # Calculate shape metrics
        if len(fluid_y) > 4:
            # Recalculate COM with updated positions
            com_y = np.mean(fluid_y)
            com_x = np.mean(fluid_x)
            
            width = np.max(fluid_x) - np.min(fluid_x) + 1
            height = np.max(fluid_y) - np.min(fluid_y) + 1
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            
            # If very elongated, move extremal cells
            if aspect_ratio > 3 and swaps_performed < max_swaps:
                remaining = max_swaps - swaps_performed
                
                # Find extremal cells (far from COM)
                distances = np.sqrt((fluid_y - com_y)**2 + (fluid_x - com_x)**2)
                sorted_indices = np.argsort(-distances)  # Farthest first
                
                for idx in sorted_indices[:remaining]:
                    y, x = fluid_y[idx], fluid_x[idx]
                    
                    # Double-check cell is still water (important!)
                    if mt[y, x] not in {MaterialType.WATER, MaterialType.MAGMA}:
                        continue
                    
                    # Try to move toward COM
                    dy = int(np.sign(com_y - y)) if abs(com_y - y) > 0.5 else 0
                    dx = int(np.sign(com_x - x)) if abs(com_x - x) > 0.5 else 0
                    
                    # Try multiple directions prioritizing toward COM
                    directions = []
                    if dy != 0 and dx != 0:
                        directions = [(dy, dx), (dy, 0), (0, dx), (-dy, dx), (dy, -dx)]
                    elif dy != 0:
                        directions = [(dy, 0), (dy, 1), (dy, -1), (0, 1), (0, -1)]
                    elif dx != 0:
                        directions = [(0, dx), (1, dx), (-1, dx), (1, 0), (-1, 0)]
                    else:
                        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    
                    for ddy, ddx in directions:
                        ny, nx = y + ddy, x + ddx
                        if (0 <= ny < h and 0 <= nx < w and mt[ny, nx] == MaterialType.SPACE):
                            self._swap_cells(y, x, ny, nx, mt, temp)
                            swaps_performed += 1
                            break
        
        # Verify conservation
        final_water_count = np.sum((mt == MaterialType.WATER) | (mt == MaterialType.MAGMA))
        if final_water_count != initial_water_count:
            print(f"[WARNING] Water not conserved: {initial_water_count} -> {final_water_count}")
        
        # Mark properties dirty
        if swaps_performed > 0:
            self.sim._properties_dirty = True
            
        return swaps_performed
    
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
            group_mass = np.sum(self.sim.density[group_mask]) * self.sim.cell_size**2
            
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
                            buoyancy_y -= (water_density - ice_density) * 9.81 * self.sim.cell_size**2
            
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
