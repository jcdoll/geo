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
from scipy.ndimage import zoom, gaussian_filter, distance_transform_edt

try:
    # Package-relative imports – preferred when running as ``python -m geo.main``
    from .core_state import CoreState
    from .core_tools import CoreToolsMixin
    from .heat_transfer_optimized_params import HeatTransferOptimized
    from .heat_transfer_multigrid_correct import HeatTransferMultigridCorrect
    from .fluid_dynamics import FluidDynamics
    from .gravity_solver import solve_potential, potential_to_gravity
    from .materials import MaterialType
except ImportError:  # Fallback for direct script execution without package context
    from core_state import CoreState  # type: ignore
    from core_tools import CoreToolsMixin  # type: ignore
    from heat_transfer_optimized_params import HeatTransferOptimized  # type: ignore
    from heat_transfer_multigrid_correct import HeatTransferMultigridCorrect  # type: ignore
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
        cell_size: float = 1.0,
        cell_depth: Optional[float] = None,
        log_level: str | int = "INFO",
        setup_planet: bool = True,
    ) -> None:
        super().__init__(width, height, cell_size=cell_size, cell_depth=cell_depth, log_level=log_level)
        
        # Store setup_planet for reset functionality
        self.setup_planet = setup_planet

        # Physics sub-modules ------------------------------------------------
        # Use multigrid if requested
        use_multigrid = getattr(self, 'use_multigrid_heat', False)
        if use_multigrid:
            self.heat_transfer = HeatTransferMultigridCorrect(self)
        else:
            self.heat_transfer = HeatTransferOptimized(self)
        self.fluid_dynamics = FluidDynamics(self)

        # Allocate self-gravity arrays (filled on demand)
        self.gravitational_potential = np.zeros((height, width), dtype=np.float64)
        self.gravity_x = np.zeros((height, width), dtype=np.float64)
        self.gravity_y = np.zeros((height, width), dtype=np.float64)

        # Simple unified kinematics toggle flag (GUI convenience)
        self.unified_kinematics = True
        
        
        # Step counter for performance logging
        self.step_count = 0
        
        # Populate with a crude basalt-magma sphere so the visualiser shows something
        if setup_planet:
            self._setup_initial_planet()

    # ------------------------------------------------------------------
    # Initial planet seeding (very rough – replaces legacy _setup_planetary_conditions)
    # ------------------------------------------------------------------
    def _setup_initial_planet(self):
        """Create a more Earth-like planet with oceans, atmosphere, and surface features."""
        # Spherical planet parameters in grid units
        radius = int(min(self.width, self.height) * 0.35)
        uranium_radius = int(radius * 0.25)  # Uranium concentrated in inner core
        magma_radius = int(radius * 0.27)    # VERY thin magma layer
        mantle_radius = int(radius * 0.85)   # Thick basalt mantle
        crust_start = int(radius * 0.90)     # Thin granite crust
        cx, cy = self.planet_center

        yy, xx = np.ogrid[:self.height, :self.width]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Uranium-enriched inner core for heat generation
        uranium_mask = dist <= uranium_radius
        self.material_types[uranium_mask] = MaterialType.URANIUM
        self.temperature[uranium_mask] = self.core_temperature + 200.0  # hot inner core
        
        # VERY thin molten layer
        magma_mask = (dist > uranium_radius) & (dist <= magma_radius)
        self.material_types[magma_mask] = MaterialType.MAGMA
        self.temperature[magma_mask] = self.core_temperature + 100.0  # slightly cooler than uranium core

        # Thick solid basalt mantle
        mantle_mask = (dist > magma_radius) & (dist <= mantle_radius)
        self.material_types[mantle_mask] = MaterialType.BASALT
        # Linear geothermal gradient from core to surface
        if np.any(mantle_mask):
            norm = (dist[mantle_mask] - magma_radius) / max(mantle_radius - magma_radius, 1)
            temps = self.core_temperature * (1 - norm) + 600.0 * norm  # Cooler at mantle-crust boundary
            self.temperature[mantle_mask] = temps

        # Transition zone - mixed basalt/granite
        transition_mask = (dist > mantle_radius) & (dist <= crust_start)
        self.material_types[transition_mask] = MaterialType.BASALT
        if np.any(transition_mask):
            norm = (dist[transition_mask] - mantle_radius) / max(crust_start - mantle_radius, 1)
            temps = 600.0 * (1 - norm) + 400.0 * norm
            self.temperature[transition_mask] = temps

        # Granite crust with surface roughness
        crust_mask = (dist > crust_start) & (dist <= radius)
        self.material_types[crust_mask] = MaterialType.GRANITE
        
        # Create surface roughness using Perlin-like noise
        np.random.seed(42)  # For reproducibility
        # Add multiple octaves of noise for realistic terrain
        surface_height = np.zeros((self.height, self.width))
        for octave in range(3):
            scale = 2 ** (octave + 2)
            amplitude = 3 / (octave + 1)  # Slightly less amplitude for gentler slopes
            noise = np.random.randn(self.height // scale + 1, self.width // scale + 1)
            # Smooth the noise
            noise_upscaled = zoom(noise, scale, order=3)[:self.height, :self.width]
            # Additional smoothing for more realistic terrain
            noise_upscaled = gaussian_filter(noise_upscaled, sigma=1.0)
            surface_height += amplitude * noise_upscaled
        
        # Apply surface roughness to planet
        for y in range(self.height):
            for x in range(self.width):
                d = dist[y, x]
                if mantle_radius < d <= radius:
                    # Adjust surface based on noise
                    local_radius = radius + surface_height[y, x]
                    if d > local_radius:
                        self.material_types[y, x] = MaterialType.SPACE
                    # Temperature gradient in crust
                    if self.material_types[y, x] == MaterialType.GRANITE:
                        depth = local_radius - d
                        self.temperature[y, x] = self.surface_temperature + depth * 10.0  # 10K per cell depth

        # Add water in surface depressions
        # Simple approach: find cells that are SPACE but adjacent to rock and below a threshold
        water_added = 0
        max_water = int(0.1 * radius * radius)  # Limit water to 10% of planet area
        
        # Define sea level based on surface roughness
        sea_level = radius + np.mean(surface_height) - 1.0
        
        # Multiple passes to allow water to spread
        for water_pass in range(3):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.material_types[y, x] == MaterialType.SPACE and water_added < max_water:
                        # Check if adjacent to solid ground or water
                        adjacent_to_solid = False
                        adjacent_to_water = False
                        
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < self.height and 0 <= nx < self.width:
                                    if self.material_types[ny, nx] in [MaterialType.GRANITE, MaterialType.BASALT]:
                                        adjacent_to_solid = True
                                    elif self.material_types[ny, nx] == MaterialType.WATER:
                                        adjacent_to_water = True
                        
                        # Add water if:
                        # 1. Adjacent to solid and below sea level, OR
                        # 2. Adjacent to water and at similar elevation (allows spreading)
                        current_elevation = dist[y, x] + surface_height[y, x] * 0.3
                        
                        if adjacent_to_solid and current_elevation < sea_level:
                            self.material_types[y, x] = MaterialType.WATER
                            self.temperature[y, x] = self.surface_temperature - 5.0
                            water_added += 1
                        elif adjacent_to_water and current_elevation < sea_level + 0.5:
                            # Allow water to spread to adjacent cells at similar height
                            self.material_types[y, x] = MaterialType.WATER
                            self.temperature[y, x] = self.surface_temperature - 5.0
                            water_added += 1

        # Create continuous atmosphere using distance transform for efficiency
        max_atmo_height = 12
        
        # Create mask of solid surfaces
        solid_mask = np.isin(self.material_types, [MaterialType.GRANITE, MaterialType.BASALT, MaterialType.WATER])
        
        # Use scipy's distance transform to find distance to nearest solid
        distance_to_solid = distance_transform_edt(~solid_mask)
        
        # Fill atmosphere based on distance
        space_mask = self.material_types == MaterialType.SPACE
        atmo_mask = space_mask & (distance_to_solid <= max_atmo_height)
        
        # Air layer (closer to surface)
        air_mask = atmo_mask & (distance_to_solid < 8)
        self.material_types[air_mask] = MaterialType.AIR
        self.temperature[air_mask] = self.surface_temperature - distance_to_solid[air_mask] * 4.0
        
        # Water vapor layer (higher up)
        vapor_mask = atmo_mask & (distance_to_solid >= 8)
        self.material_types[vapor_mask] = MaterialType.WATER_VAPOR
        self.temperature[vapor_mask] = self.surface_temperature - distance_to_solid[vapor_mask] * 6.0

        # Set space temperature for all space cells
        space_mask = self.material_types == MaterialType.SPACE
        self.temperature[space_mask] = self.space_temperature
        
        # Mark derived properties dirty and refresh immediately
        self._properties_dirty = True
        self._update_material_properties()
        

    
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
            heat_time = time.perf_counter() - _last_cp
            self._perf_times["heat_diffusion"] = heat_time
            
            # Extract solar heating timing if available
            if hasattr(self.heat_transfer, 'solar_heating'):
                solar_stats = self.heat_transfer.solar_heating.get_timing_stats()
                self._perf_times["solar_ray_marching"] = solar_stats['ray_march_time']
                self._perf_times["heat_diffusion_core"] = heat_time - solar_stats['total_time']
            
            _last_cp = time.perf_counter()

            # 3) Self-gravity (Poisson solve + gradient)
            self.calculate_self_gravity()
            self._perf_times["self_gravity"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()


            # 5) Cell swapping – handles buoyancy, settling, fluid flow
            self.fluid_dynamics.apply_unified_kinematics(self.dt)
            self._perf_times["cell_swapping"] = time.perf_counter() - _last_cp
            
            # Get cell swapping statistics
            if hasattr(self.fluid_dynamics, 'get_stats'):
                swap_stats = self.fluid_dynamics.get_stats()
                self.cell_swap_stats = swap_stats  # Store for visualization
            
            _last_cp = time.perf_counter()

            # 6) Material processes – metamorphism, weathering, phase transitions
            if getattr(self, 'enable_material_processes', True):
                step_count = int(self.time / self.dt)
                
                # Apply metamorphism every step (fundamental physics)
                self.material_processes.apply_metamorphism()
                
                # Apply phase transitions every step (water/ice/vapor)
                self.material_processes.apply_phase_transitions()
                
                # Apply weathering every other step (slower process)
                if getattr(self, 'enable_weathering', True) and step_count % 2 == 0:
                    self.material_processes.apply_weathering()
            
            self._perf_times["material_processes"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 7) Advance simulation clock using stability-scaled dt
            self.time += self.dt * stability
            self._last_stability_factor = stability
            self._actual_effective_dt = self.dt * stability
            
            # Update solar angle (orbital rotation)
            solar_rotation_rate = 360.0 / 86400.0  # 360 degrees per 86400s (1 day)
            self.solar_angle += solar_rotation_rate * self.dt * stability
            # Wrap angle to stay in [0, 360) range
            self.solar_angle = self.solar_angle % 360.0
            self._perf_times["update_time"] = time.perf_counter() - _last_cp
            _last_cp = time.perf_counter()

            # 7) Optional analytics (graphs)
            self._record_time_series_data()
            self._perf_times["record_ts"] = time.perf_counter() - _last_cp
            
            # Increment step counter
            self.step_count += 1

        except Exception:
            traceback.print_exc()
            raise
        finally:
            # Total wall-clock duration
            self._perf_times["total"] = time.perf_counter() - step_start_total
            
            # Store timings for visualizer
            self.step_timings = self._perf_times.copy()
            # Calculate FPS
            total_time = self._perf_times.get("total", 0.001)  # Avoid division by zero
            self.fps = 1.0 / total_time if total_time > 0 else 0.0

            # Emit nicely formatted per-line timings when verbose logging is on
            if getattr(self, "logging_enabled", False):
                self.logger.info("Performance timing (ms):")
                # Report individual timings (excluding composite ones)
                for name, seconds in sorted(self._perf_times.items()):
                    # Skip composite timings that include others
                    if name in ['heat_diffusion', 'total']:
                        continue
                    self.logger.info("  %s: %.1f", f"{name:<20}", seconds * 1000.0)
                
                # Report totals separately
                if 'heat_diffusion' in self._perf_times:
                    self.logger.info("  %s: %.1f (includes solar)", f"{'heat_diffusion':<20}", 
                                   self._perf_times['heat_diffusion'] * 1000.0)
                self.logger.info("  %s: %.1f", f"{'TOTAL':<20}", self._perf_times['total'] * 1000.0)

    def step_backward(self):  # type: ignore[override]
        """Undo last step (if history available)."""
        if not self.history:
            return  # nothing to undo
        state = self.history.pop()
        self.material_types[:] = state["material_types"]
        self.temperature[:] = state["temperature"]
        self.age[:] = state["age"]
        self.time = state["time"]
        self.power_density[:] = state["power_density"]
        if "solar_angle" in state:
            self.solar_angle = state["solar_angle"]
        
        # Mark properties as dirty to trigger material property recalculation
        self._properties_dirty = True
        
        # Regenerate derived fields that depend on the restored state
        # 1. Update material properties (density etc.)
        if hasattr(self, "_update_material_properties"):
            self._update_material_properties()
        
        # 2. Recalculate gravity field (depends on density)
        if hasattr(self, "calculate_self_gravity"):
            self.calculate_self_gravity()
        
        
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
        """Restore primordial state but keep grid dimensions."""
        # Store current setup_planet preference
        setup_planet = getattr(self, 'setup_planet', True)
        # Reinitialize keeping the same dimensions and settings
        self.__init__(self.width, self.height, cell_size=self.cell_size, 
                      cell_depth=self.cell_depth, setup_planet=setup_planet) 