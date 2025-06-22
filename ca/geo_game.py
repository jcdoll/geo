"""geo_game.py – Thin façade exposing CoreState + modules for the GUI.

Historically the pygame visualiser instantiated
``geo.simulation_engine.GeologySimulation`` which pulled in the entire
monolithic legacy engine.  This wrapper offers the *same public API* that
the GUI expects (step_forward, add_heat_source, …) but is backed by the
new modular code-base instead of the 2800-line blob.

The intent is **not** to re-implement every corner-case of the old
engine – only the parts exercised by the visualiser and existing unit
tests.  More sophisticated physics will be added module by module as we
progress through the roadmap (pressure solve, projection flow, …).
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import time, traceback

try:
    # Package-relative imports – preferred when running as ``python -m geo.main``
    from .core_state import CoreState
    from .core_tools import CoreToolsMixin
    from .heat_transfer import HeatTransfer
    from .fluid_dynamics import FluidDynamics
    from .gravity_solver import solve_potential, potential_to_gravity
    from .materials import MaterialType
except ImportError:  # Fallback for direct script execution without package context
    from core_state import CoreState  # type: ignore
    from core_tools import CoreToolsMixin  # type: ignore
    from heat_transfer import HeatTransfer  # type: ignore
    from fluid_dynamics import FluidDynamics  # type: ignore
    from gravity_solver import solve_potential, potential_to_gravity  # type: ignore
    from materials import MaterialType  # type: ignore


class GeoGame(CoreState, CoreToolsMixin):
    """Unified class used by the visualiser (inherits CoreState + helpers)."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        width: int,
        height: int,
        *,
        cell_size: float = 50.0,
        cell_depth: Optional[float] = None,
        quality: int = 1,
        log_level: str | int = "INFO",
        setup_planet: bool = True,
    ) -> None:
        super().__init__(width, height, cell_size=cell_size, cell_depth=cell_depth, quality=quality, log_level=log_level)

        # Physics sub-modules ------------------------------------------------
        self.heat_transfer = HeatTransfer(self)
        self.fluid_dynamics = FluidDynamics(self)

        # Allocate self-gravity arrays (filled on demand)
        self.gravitational_potential = np.zeros((height, width), dtype=np.float64)
        self.gravity_x = np.zeros((height, width), dtype=np.float64)
        self.gravity_y = np.zeros((height, width), dtype=np.float64)

        # Simple unified kinematics toggle flag (GUI convenience)
        self.unified_kinematics = True
        
        # Flag to enable/disable pressure calculation
        self.enable_pressure = True
        

        # Populate with a crude basalt-magma sphere so the visualiser shows something
        if setup_planet:
            self._setup_initial_planet()

    # ------------------------------------------------------------------
    # Initial planet seeding (very rough – replaces legacy _setup_planetary_conditions)
    # ------------------------------------------------------------------
    def _setup_initial_planet(self):
        """Fill the grid with a simple rocky planet + uranium-enriched core."""
        # Spherical planet parameters in grid units
        radius = int(min(self.width, self.height) * 0.35)
        core_radius = int(radius * 0.4)
        uranium_radius = int(radius * 0.25)  # Uranium concentrated in inner core
        cx, cy = self.planet_center

        yy, xx = np.ogrid[:self.height, :self.width]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Uranium-enriched inner core for heat generation
        uranium_mask = dist <= uranium_radius
        self.material_types[uranium_mask] = MaterialType.URANIUM
        self.temperature[uranium_mask] = self.core_temperature + 200.0  # hot inner core
        
        # Molten outer core
        core_mask = (dist > uranium_radius) & (dist <= core_radius)
        self.material_types[core_mask] = MaterialType.MAGMA
        self.temperature[core_mask] = self.core_temperature + 100.0  # slightly cooler than uranium core

        # Solid basalt mantle/crust
        mantle_mask = (dist > core_radius) & (dist <= radius)
        self.material_types[mantle_mask] = MaterialType.BASALT
        # Linear geothermal gradient from core to surface
        if np.any(mantle_mask):
            norm = (dist[mantle_mask] - core_radius) / max(radius - core_radius, 1)
            temps = self.core_temperature * (1 - norm) + self.surface_temperature * norm
            self.temperature[mantle_mask] = temps

        # Atmosphere: thin shell of AIR just outside surface (optional visual cue)
        atmo_mask = (dist > radius) & (dist <= radius + 2)
        self.material_types[atmo_mask] = MaterialType.AIR
        self.temperature[atmo_mask] = self.surface_temperature

        # Mark derived properties dirty and refresh immediately
        self._properties_dirty = True
        self._update_material_properties()
        
        # Calculate initial pressure distribution in hydrostatic equilibrium
        self._calculate_initial_pressure()

    def _calculate_initial_pressure(self):
        """Calculate initial pressure distribution in hydrostatic equilibrium.
        
        For a planet in hydrostatic equilibrium, pressure should satisfy:
        ∇P = ρg (force balance)
        
        This gives us the Poisson equation: ∇²P = ∇·(ρg)
        """
        # First calculate gravity field
        self.calculate_self_gravity()
        
        # Add any external gravity
        gx_total = self.gravity_x.copy()
        gy_total = self.gravity_y.copy()
        if hasattr(self, 'external_gravity'):
            g_ext_x, g_ext_y = self.external_gravity
            if g_ext_x != 0.0 or g_ext_y != 0.0:
                gx_total += g_ext_x
                gy_total += g_ext_y
        
        # Use fluid dynamics module to calculate pressure
        # This ensures consistency with the simulation's pressure calculation
        self.fluid_dynamics.calculate_planetary_pressure()
        
        # Log initial pressure statistics
        planet_mask = self.material_types != MaterialType.SPACE
        if np.any(planet_mask):
            p_min = np.min(self.pressure[planet_mask]) / 1e6  # Convert to MPa
            p_max = np.max(self.pressure[planet_mask]) / 1e6
            p_mean = np.mean(self.pressure[planet_mask]) / 1e6
            
            # Estimate pressure at center (should be highest)
            cx, cy = self.planet_center
            cx, cy = int(cx), int(cy)
            if 0 <= cx < self.width and 0 <= cy < self.height:
                p_center = self.pressure[cy, cx] / 1e6
            else:
                p_center = p_max
                
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Initial pressure distribution:")
                self.logger.info(f"  Min: {p_min:.1f} MPa")
                self.logger.info(f"  Max: {p_max:.1f} MPa") 
                self.logger.info(f"  Mean: {p_mean:.1f} MPa")
                self.logger.info(f"  Center: {p_center:.1f} MPa")
    
    # ------------------------------------------------------------------
    # Self-gravity
    # ------------------------------------------------------------------
    def calculate_self_gravity(self):
        """Solve Poisson equation and update *gravity_x/y* arrays (m/s²)."""
        # Ensure density grid reflects current materials
        if getattr(self, "_properties_dirty", False):
            self._update_material_properties()

        # Scale density by depth ratio to account for 2.5D simulation
        # The gravity solver assumes cubic cells, but we have a slab of depth cell_depth
        depth_ratio = self.cell_depth / self.cell_size
        effective_density = self.density * depth_ratio
        phi = solve_potential(effective_density, self.cell_size)
        
        # Higher-order gradient (5-point compact stencil) for isotropy
        gx, gy = self._compute_gravity_5pt(phi, self.cell_size)

        self.gravitational_potential[:] = phi
        self.gravity_x[:] = gx
        self.gravity_y[:] = gy
        return gx, gy

    # ------------------------------------------------------------------
    # High-order gradient helper (reduces directional artefacts)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_gravity_5pt(phi: np.ndarray, dx: float):
        """Return g = −∇Φ using a 4th-order 5-point central difference.

        Interior formula (x dimension):
            dΦ/dx ≈ (−Φ[i+2] + 8Φ[i+1] − 8Φ[i−1] + Φ[i−2]) / (12Δx)
        Same for y.  Edges (<2 cells from border) fall back to 2-point.
        """
        gx = np.zeros_like(phi)
        gy = np.zeros_like(phi)

        # Interior (avoid first/last 2 columns)
        gx[:, 2:-2] = (
            -phi[:, 4:] + 8 * phi[:, 3:-1] - 8 * phi[:, 1:-3] + phi[:, :-4]
        ) / (12 * dx)

        # Interior rows for y
        gy[2:-2, :] = (
            -phi[4:, :] + 8 * phi[3:-1, :] - 8 * phi[1:-3, :] + phi[:-4, :]
        ) / (12 * dx)

        # Fallback to simple central difference near edges (order ≈ 2)
        gx[:, 1] = (phi[:, 2] - phi[:, 0]) / (2 * dx)
        gx[:, 0] = (phi[:, 1] - phi[:, 0]) / dx
        gx[:, -2] = (phi[:, -1] - phi[:, -3]) / (2 * dx)
        gx[:, -1] = (phi[:, -1] - phi[:, -2]) / dx

        gy[1, :] = (phi[2, :] - phi[0, :]) / (2 * dx)
        gy[0, :] = (phi[1, :] - phi[0, :]) / dx
        gy[-2, :] = (phi[-1, :] - phi[-3, :]) / (2 * dx)
        gy[-1, :] = (phi[-1, :] - phi[-2, :]) / dx

        # Acceleration is −∇Φ
        return -gx, -gy

    # ------------------------------------------------------------------
    # Time stepping / undo
    # ------------------------------------------------------------------
    def step_forward(self, dt: Optional[float] = None):  # type: ignore[override]
        """Advance simulation by one macro-step (default: ``self.dt``)."""
        # --------------------------------------------------------------
        # Performance instrumentation – wall-clock timings per section
        # --------------------------------------------------------------
        step_start_total = time.perf_counter()
        self._perf_times: dict[str, float] = {}
        _last_cp = step_start_total

        try:
            # Update configurable dt if caller passed an override
            if dt is not None:
                self.dt = dt

            # 0) Snapshot for undo **before** mutating state
            self._save_state()
            self._perf_times["save_state"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 1) Refresh material-property caches when needed
            if getattr(self, "_properties_dirty", False):
                self._update_material_properties()
            self._perf_times["update_props"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 2) Heat diffusion & sources
            new_T, stability = self.heat_transfer.solve_heat_diffusion()
            self.temperature = new_T
            self._perf_times["heat_diffusion"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 3) Self-gravity (Poisson solve + gradient)
            self.calculate_self_gravity()
            self._perf_times["self_gravity"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 4) Pressure field (fluid-dynamics module)
            if self.enable_pressure:
                self.fluid_dynamics.calculate_planetary_pressure()
            self._perf_times["pressure"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 5) Unified kinematics – handles buoyancy, settling, fluid flow
            self.fluid_dynamics.apply_unified_kinematics(self.dt)
            self._perf_times["unified_kinematics"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 6) Material processes – metamorphism, weathering, phase transitions
            if getattr(self, 'enable_material_processes', True):
                step_count = int(self.time / self.dt)
                
                # Apply metamorphism every step (fundamental physics)
                self.material_processes.apply_metamorphism()
                
                # Apply phase transitions every step (water/ice/vapor)
                self.material_processes.apply_phase_transitions()
                
                # Apply weathering every 10th step (slower process)
                if getattr(self, 'enable_weathering', True) and step_count % 10 == 0:
                    self.material_processes.apply_weathering()
            
            self._perf_times["material_processes"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 7) Advance simulation clock using stability-scaled dt
            self.time += self.dt * stability
            self._last_stability_factor = stability
            self._actual_effective_dt = self.dt * stability
            self._perf_times["update_time"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 7) Optional analytics (graphs)
            self._record_time_series_data()
            self._perf_times["record_ts"] = time.perf_counter() - _last_cp

        except Exception:
            traceback.print_exc()
            raise
        finally:
            # Total wall-clock duration
            self._perf_times["total"] = time.perf_counter() - step_start_total

            # Emit nicely formatted per-line timings when verbose logging is on
            if getattr(self, "logging_enabled", False):
                self.logger.info("Performance timing (ms):")
                for name, seconds in self._perf_times.items():
                    self.logger.info("  %s: %.1f", f"{name:<15}", seconds * 1000.0)

    def step_backward(self):  # type: ignore[override]
        """Undo last step (if history available)."""
        if not self.history:
            return  # nothing to undo
        state = self.history.pop()
        self.material_types[:] = state["material_types"]
        self.temperature[:] = state["temperature"]
        self.pressure[:] = state["pressure"]
        self.pressure_offset[:] = state["pressure_offset"]
        self.age[:] = state["age"]
        self.time = state["time"]
        self.power_density[:] = state["power_density"]
        
        # Mark properties as dirty to trigger material property recalculation
        self._properties_dirty = True
        
        # Regenerate derived fields that depend on the restored state
        # 1. Update material properties (density etc.)
        if hasattr(self, "_update_material_properties"):
            self._update_material_properties()
        
        # 2. Recalculate gravity field (depends on density)
        if hasattr(self, "calculate_self_gravity"):
            self.calculate_self_gravity()
        
        # 3. Recalculate pressure field (depends on density and gravity)
        if self.enable_pressure and hasattr(self, "fluid_dynamics"):
            self.fluid_dynamics.calculate_planetary_pressure()
        
        # 4. Restore velocity fields if they were saved
        if hasattr(self, "fluid_dynamics"):
            if "velocity_x" in state and "velocity_y" in state:
                self.fluid_dynamics.velocity_x[:] = state["velocity_x"]
                self.fluid_dynamics.velocity_y[:] = state["velocity_y"]
            else:
                # Reset velocities to zero if we don't have historical data
                self.fluid_dynamics.velocity_x.fill(0.0)
                self.fluid_dynamics.velocity_y.fill(0.0)

    # ------------------------------------------------------------------
    # Complete reset (used by visualiser key *R*)
    # ------------------------------------------------------------------
    def reset(self):  # type: ignore[override]
        """Restore primordial state but keep grid dimensions & quality."""
        self.__init__(self.width, self.height, cell_size=self.cell_size, quality=self.quality) 