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
        # Cumulative sub‐cell displacements (in *cell* units)
        self.disp_accum_x = np.zeros_like(self.velocity_x, dtype=np.float32)
        self.disp_accum_y = np.zeros_like(self.velocity_y, dtype=np.float32)
    
    def calculate_planetary_pressure(self):
        """Multigrid solve for pressure using self-gravity field."""

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

        # Solve Poisson
        pressure = solve_pressure(rhs, dx)

        # Apply Dirichlet boundary (space/atmosphere = 0 MPa)
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.SPACE) |
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        pressure[atmosphere_mask] = 0.0

        # Store & add persistent offsets
        self.sim.pressure[:] = np.maximum(0.0, pressure + self.sim.pressure_offset)
    
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

        # Gravity: sum of self-gravity field and optional constant external field
        gx_total = self.sim.gravity_x.copy()
        gy_total = self.sim.gravity_y.copy()

        if hasattr(self.sim, "external_gravity"):
            g_ext_x, g_ext_y = self.sim.external_gravity
            if g_ext_x != 0.0 or g_ext_y != 0.0:
                gx_total = gx_total + g_ext_x
                gy_total = gy_total + g_ext_y

        fx = rho * gx_total
        fy = rho * gy_total

        # ------------------------------------------------------------------
        # Buoyancy term – delegated to helper for re-use in other routines
        # ------------------------------------------------------------------
        fbx, fby = self._compute_buoyancy_force(rho, gx_total, gy_total)
        fx += fbx
        fy += fby

        # Pressure gradient term – pressure stored in MPa → convert to Pa
        P_pa = self.sim.pressure * 1e6
        dx = self.sim.cell_size
        # Simple central differences
        gradP_y, gradP_x = np.gradient(P_pa, dx)

        fx -= gradP_x
        fy -= gradP_y

        self.sim._perf_times["uk_force_assembly"] = _time.perf_counter() - _t_force_start

        # ------------------------------------------------------------------
        # 2) Velocity update  u ← u + (f/ρ) Δt
        # ------------------------------------------------------------------
        _t_vel_start = _time.perf_counter()
        dt_seconds = dt  # dt already in seconds
        accel_x = np.where(rho > 0, fx / rho, 0.0)
        accel_y = np.where(rho > 0, fy / rho, 0.0)

        self.velocity_x += accel_x * dt_seconds
        self.velocity_y += accel_y * dt_seconds

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
        div_u[1:-1, 1:-1] = (
            (self.velocity_x[1:-1, 2:] - self.velocity_x[1:-1, :-2]) +
            (self.velocity_y[2:, 1:-1] - self.velocity_y[:-2, 1:-1])
        ) / (2 * dx_m)

        # ------------------------------------------------------------------
        # Variable-density Chorin projection
        #    ∇·(1/ρ ∇φ) = div_u / Δt
        # ------------------------------------------------------------------
        rhs = div_u / dt_seconds  # (1/s) / s → 1/s² but solver expects rhs units of φ / dx²
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

        # Velocity correction  u = u* − Δt/ρ ∇φ
        self.velocity_x -= grad_phi_x * dt_seconds * inv_rho
        self.velocity_y -= grad_phi_y * dt_seconds * inv_rho

        self.sim._perf_times["uk_projection"] = _time.perf_counter() - _t_proj_start

        # ------------------------------------------------------------------
        # 4) Accumulate displacement in *cell* units and move when |disp| ≥ 0.5
        # ------------------------------------------------------------------
        _t_disp_start = _time.perf_counter()
        _t_swap_start = _t_disp_start  # default so variable exists even if no swap executed
        swaps_done = 0

        cell_size = self.sim.cell_size
        self.disp_accum_x += (self.velocity_x * dt_seconds) / cell_size
        self.disp_accum_y += (self.velocity_y * dt_seconds) / cell_size

        # Determine integer cell moves (at most ±1 for stability)
        trigger = 0.5  # 50 % of a cell
        step_x = np.zeros_like(self.disp_accum_x, dtype=np.int8)
        step_y = np.zeros_like(self.disp_accum_y, dtype=np.int8)
        step_x[self.disp_accum_x >= trigger] = 1
        step_x[self.disp_accum_x <= -trigger] = -1
        step_y[self.disp_accum_y >= trigger] = 1
        step_y[self.disp_accum_y <= -trigger] = -1

        move_mask = (step_x != 0) | (step_y != 0)
        if not np.any(move_mask):
            self.sim._perf_times["uk_displacement"] = _time.perf_counter() - _t_disp_start
            self.sim._perf_times["uk_swaps"] = 0.0
            return  # nothing to advect this macro-step

        ys, xs = np.where(move_mask)
        tgt_y = ys + step_y[ys, xs]
        tgt_x = xs + step_x[ys, xs]

        # Filter out moves that would leave the grid
        in_bounds = (
            (tgt_y >= 0) & (tgt_y < self.sim.height) &
            (tgt_x >= 0) & (tgt_x < self.sim.width)
        )
        if not np.any(in_bounds):
            return

        src_y = ys[in_bounds]
        src_x = xs[in_bounds]
        tgt_y = tgt_y[in_bounds]
        tgt_x = tgt_x[in_bounds]

        # Deduplicate conflicting swaps
        src_y, src_x, tgt_y, tgt_x = self.sim._dedupe_swap_pairs(src_y, src_x, tgt_y, tgt_x)
        if len(src_y) == 0:
            self.sim._perf_times["uk_displacement"] = _time.perf_counter() - _t_disp_start
            self.sim._perf_times["uk_swaps"] = 0.0
            return

        # Only allow swap if source is denser than target (deterministic stratification)
        src_density = rho[src_y, src_x]
        tgt_density = rho[tgt_y, tgt_x]
        allowed = src_density > tgt_density + 1e-3  # tolerance for equal densities
        if np.any(allowed):
            _t_swap_start = _time.perf_counter()
            self._perform_material_swaps(src_y[allowed], src_x[allowed], tgt_y[allowed], tgt_x[allowed])
            
            self.disp_accum_x[src_y[allowed], src_x[allowed]] -= step_x[src_y[allowed], src_x[allowed]]
            self.disp_accum_y[src_y[allowed], src_x[allowed]] -= step_y[src_y[allowed], src_x[allowed]]
            swaps_done = len(src_y[allowed])

        # Apply drag only if enabled
        if getattr(self.sim, "enable_solid_drag", True):
            solid = self.sim._get_solid_mask()
            self.velocity_x[solid] *= 0.2
            self.velocity_y[solid] *= 0.2

        # Record timings
        self.sim._perf_times["uk_displacement"] = _time.perf_counter() - _t_disp_start
        self.sim._perf_times["uk_swaps"] = _time.perf_counter() - _t_swap_start
        self.sim._perf_times["uk_swaps_count"] = swaps_done
    
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
