from __future__ import annotations

"""CoreState – lightweight base that provides shared state/utility code
for the modular geological simulation engine.

The goal is to remove every hard dependency on the legacy monolithic
``simulation_engine_original``.  This class contains only the grid
allocation, configuration constants, and helper functions that the new
physics modules (``heat_transfer.py``, ``fluid_dynamics.py``,
``atmospheric_processes.py``, ``material_processes.py``) expect.

Heavy-weight physics such as diffusion, stratification, collapse, etc.
are intentionally **not** included here – those responsibilities now live
in the dedicated modules.
"""

from typing import Optional, Tuple
import logging

import numpy as np
from scipy import ndimage

try:
    from .materials import MaterialType, MaterialDatabase
except ImportError:  # fallback for standalone unit tests
    from materials import MaterialType, MaterialDatabase


class CoreState:
    """Shared state and helpers for the modular simulation engine."""

    # ------------------------------------------------------------------
    # Construction & basic grid allocation
    # ------------------------------------------------------------------
    def __init__(
        self,
        width: int,
        height: int,
        *,
        cell_size: float = 50.0,
        quality: int = 1,
        log_level: str | int = "INFO",
    ) -> None:
        self.width = width
        self.height = height
        self.cell_size = float(cell_size)

        # ---------- logger -------------------------------------------------
        self.logger = logging.getLogger(f"GeologySimulation_{id(self)}")
        self.logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

        # ---------- performance/quality -----------------------------------
        self._setup_performance_config(quality)

        # ---------- core grids --------------------------------------------
        self.material_types = np.full((height, width), MaterialType.SPACE, dtype=object)
        self.temperature = np.zeros((height, width), dtype=np.float64)
        self.pressure = np.zeros((height, width), dtype=np.float64)
        self.pressure_offset = np.zeros((height, width), dtype=np.float64)
        self.age = np.zeros((height, width), dtype=np.float64)
        self.power_density = np.zeros((height, width), dtype=np.float64)  # W/m³ – diagnostic

        # Velocity fields (m/s) for unified kinematics
        self.velocity_x = np.zeros((height, width), dtype=np.float64)
        self.velocity_y = np.zeros((height, width), dtype=np.float64)

        # ---------- derived property grids --------------------------------
        self.density = np.zeros((height, width), dtype=np.float64)
        self.thermal_conductivity = np.zeros((height, width), dtype=np.float64)
        self.specific_heat = np.zeros((height, width), dtype=np.float64)

        # ---------- simulation parameters ---------------------------------
        self.time = 0.0  # years
        self.dt = 1.0    # years per macro-step

        # Unit conversions
        self.seconds_per_year = 365.25 * 24 * 3600
        self.stefan_boltzmann_geological = 5.67e-8 * self.seconds_per_year  # J/(year·m²·K⁴)

        # Planetary constants (identical to legacy engine)
        self.planet_radius_fraction = 0.8
        self.planet_center: Tuple[int, int] = (width // 2, height // 2)
        self.center_of_mass: Tuple[float, float] = (width / 2, height / 2)

        # Material DB ------------------
        self.material_db = MaterialDatabase()

        # Misc constants referenced by modules (copied verbatim)
        self.atmospheric_diffusivity_enhancement = 5.0
        self.interface_diffusivity_enhancement = 1.5
        self.surface_radiation_depth_fraction = 0.1
        self.radiative_cooling_efficiency = 0.9
        self.max_thermal_diffusivity = 1e-3
        self.space_temperature = 2.7  # K
        self.reference_temperature = 273.15
        self.core_temperature = 1200.0 + 273.15
        self.surface_temperature = 50.0 + 273.15
        self.temperature_decay_constant = 2.0
        self.melting_temperature = 1200 + 273.15
        self.hot_solid_temperature_threshold = 1200.0
        self.core_heating_depth_scale = 0.5
        self.surface_pressure = 0.1  # MPa
        self.atmospheric_scale_height = 8400  # m
        self.average_gravity = 9.81
        self.average_solid_density = 3000
        self.average_fluid_density = 2000
        self.solar_constant = 50
        self.solar_angle = 90.0
        self.planetary_distance_factor = 1.0
        self.base_greenhouse_effect = 0.2
        self.max_greenhouse_effect = 0.8
        self.greenhouse_vapor_scaling = 1000.0
        self.atmospheric_convection_mixing = 0.3

        # New default for modular heat-transfer code
        self.atmospheric_absorption_method = "directional_sweep"

        # Flags used by modules
        self.logging_enabled = False

        # ---------- caches / book-keeping ---------------------------------
        self._material_props_cache: dict = {}
        self._properties_dirty = True

        self._setup_neighbors()
        self._update_material_properties()

        # --------------------------------------------------------------
        # Additional parameters required by modular FluidDynamics layer
        # --------------------------------------------------------------
        # Store quality level explicitly so downstream modules can branch on it
        self.quality: int = quality
        # Probability that a solid voxel adjacent to a cavity will fall inward
        self.gravitational_fall_probability: float = 0.25  # tuned for visual stability
        # Probability used by density-driven stratification swaps
        self.density_swap_probability: float = 0.25

        # minimal history for undo (optional)
        self.history: list = []
        self.max_history = 100

        # Diffusion stencil selection ('radius1' or 'radius2')
        self.diffusion_stencil = "radius2"

        # ------------------------------------------------------------------
        # Place-holder time-series buffers so the visualiser's graphs work.
        # Modules are free to push to these lists at their leisure.
        # ------------------------------------------------------------------
        self.time_series = {
            'time': [],
            'avg_temperature': [],
            'max_temperature': [],
            'total_energy': [],
            'net_power': [],
            'greenhouse_factor': [],
            'planet_albedo': [],
        }

        # Running thermal flux bookkeeping (W).  Updated by HeatTransfer.
        self.thermal_fluxes = {
            'solar_input': 0.0,
            'radiative_output': 0.0,
            'internal_heating': 0.0,
            'atmospheric_heating': 0.0,
            'net_flux': 0.0,
        }

        # ------------------------------------------------------------------
        # Algorithmic method selectors used by modular solvers
        # ------------------------------------------------------------------
        self.thermal_diffusion_method = "explicit_euler"  # only method implemented so far
        self.radiative_cooling_method = "linearized_stefan_boltzmann"

    # ------------------------------------------------------------------
    #  Performance presets (copied from legacy engine)
    # ------------------------------------------------------------------
    def _setup_performance_config(self, quality: int) -> None:  # noqa: C901  complexity ok; copy-paste from legacy
        if quality == 1:
            self.process_fraction_mobile = 1.0
            self.process_fraction_solid = 1.0
            self.process_fraction_air = 1.0
            self.process_fraction_water = 1.0
            self.density_ratio_threshold = 1.05
            self.max_diffusion_substeps = 50
            self.neighbor_count = 8
        elif quality == 2:
            self.process_fraction_mobile = 0.5
            self.process_fraction_solid = 0.5
            self.process_fraction_air = 0.5
            self.process_fraction_water = 0.5
            self.density_ratio_threshold = 1.1
            self.max_diffusion_substeps = 35
            self.neighbor_count = 8
        elif quality == 3:
            self.process_fraction_mobile = 0.2
            self.process_fraction_solid = 0.33
            self.process_fraction_air = 0.25
            self.process_fraction_water = 0.25
            self.density_ratio_threshold = 1.2
            self.max_diffusion_substeps = 20
            self.neighbor_count = 4
        else:
            raise ValueError("quality must be 1, 2, or 3")

    # ------------------------------------------------------------------
    #  Kernel / neighbour setup
    # ------------------------------------------------------------------
    def _setup_neighbors(self) -> None:
        self.neighbors_4 = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.neighbors_8 = np.array([
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ])
        self.distance_factors_8 = np.ones(8)

        # Kernels used by diffusion & morphology
        self._circular_kernel_3x3 = self._create_circular_kernel(3)
        self._circular_kernel_5x5 = self._create_circular_kernel(5)

        # 13-point isotropic Laplacian (identical to legacy)
        self._laplacian_kernel_radius2 = (1.0 / 6.0) * np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 2, 1, 0],
                [1, 2, -16, 2, 1],
                [0, 1, 2, 1, 0],
                [0, 0, 1, 0, 0],
            ], dtype=np.float64,
        )

    def _create_circular_kernel(self, size: int) -> np.ndarray:
        kernel = np.zeros((size, size), dtype=bool)
        centre = size // 2
        radius = centre + 0.5
        for j in range(size):
            for i in range(size):
                if (i - centre) ** 2 + (j - centre) ** 2 <= radius ** 2:
                    kernel[j, i] = True
        return kernel

    # ------------------------------------------------------------------
    #  Misc helper functions referenced by modules
    # ------------------------------------------------------------------
    def _get_neighbors(self, num_neighbors: int = 8, *, shuffle: bool = True):
        if num_neighbors == 4:
            nbrs = self.neighbors_4.tolist()
        elif num_neighbors == 8:
            nbrs = self.neighbors_8.tolist()
        else:
            raise ValueError("num_neighbors must be 4 or 8")
        if shuffle:
            np.random.shuffle(nbrs)
        return nbrs

    def _get_distances_from_center(self, center_x: Optional[float] = None, center_y: Optional[float] = None):
        if center_x is None or center_y is None:
            center_x, center_y = self.center_of_mass
        yy, xx = np.ogrid[:self.height, :self.width]
        return np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    def _get_planet_radius(self) -> float:
        return min(self.width, self.height) * self.planet_radius_fraction / 2.0

    def _get_solar_direction(self):
        angle_rad = np.radians(self.solar_angle)
        return np.cos(angle_rad), -np.sin(angle_rad)

    def _dedupe_swap_pairs(self, src_y, src_x, tgt_y, tgt_x):
        if len(src_y) == 0:
            return src_y, src_x, tgt_y, tgt_x
        src_flat = src_y * self.width + src_x
        tgt_flat = tgt_y * self.width + tgt_x
        combined = np.concatenate([src_flat, tgt_flat])
        unique_cells, counts = np.unique(combined, return_counts=True)
        conflict_cells = unique_cells[counts > 1]
        conflict_src = np.isin(src_flat, conflict_cells)
        conflict_tgt = np.isin(tgt_flat, conflict_cells)
        keep = ~(conflict_src | conflict_tgt)
        return src_y[keep], src_x[keep], tgt_y[keep], tgt_x[keep]

    # ------------------------------------------------------------------
    #  Derived material properties
    # ------------------------------------------------------------------
    def _update_material_properties(self, force: bool = False):
        """Synchronise density / k / cp arrays with `material_types`.

        Recomputes only when `_properties_dirty` is True **or** the caller
        explicitly passes ``force=True``.  This keeps the method O(N_unique)
        rather than O(N_grid) when no materials have changed.
        """
        # Recompute every call unless caller explicitly wants to skip (rare).
        # This keeps external utility code simple: they can mutate
        # `material_types` directly and call this helper without touching the
        # private dirty flag.
        unique = set(self.material_types.flatten())
        obsolete = set(self._material_props_cache) - unique
        for mat in obsolete:
            del self._material_props_cache[mat]
        for mat in unique:
            if mat not in self._material_props_cache:
                props = self.material_db.get_properties(mat)
                self._material_props_cache[mat] = (
                    props.density,
                    props.thermal_conductivity,
                    props.specific_heat,
                )
            dens, k, cp = self._material_props_cache[mat]
            mask = self.material_types == mat
            if np.any(mask):
                self.density[mask] = dens
                self.thermal_conductivity[mask] = k
                self.specific_heat[mask] = cp
        self._properties_dirty = False

        # Recalculate centre-of-mass so that downstream gravity/pressure
        # calculations (and several unit-tests) see the updated mass
        # distribution immediately after a manual material modification.
        self._calculate_center_of_mass()

    # ------------------------------------------------------------------
    #  Solid mask utility
    # ------------------------------------------------------------------
    def _get_solid_mask(self):
        unique = set(self.material_types.flatten())
        solid_lookup = {m: self.material_db.get_properties(m).is_solid for m in unique}
        return np.vectorize(solid_lookup.get)(self.material_types)

    # ------------------------------------------------------------------
    #  Centre-of-mass (used by gravity & modules)
    # ------------------------------------------------------------------
    def _calculate_center_of_mass(self):
        matter_mask = self.material_types != MaterialType.SPACE
        if not np.any(matter_mask):
            self.center_of_mass = (self.width / 2, self.height / 2)
            return
        yy, xx = np.where(matter_mask)
        cell_volume = self.cell_size ** 2
        masses = self.density[matter_mask] * cell_volume
        total_mass = np.sum(masses)
        if total_mass > 0:
            cx = np.sum(masses * xx) / total_mass
            cy = np.sum(masses * yy) / total_mass
            self.center_of_mass = (cx, cy)

    # ------------------------------------------------------------------
    #  Minimal history helpers (undo)
    # ------------------------------------------------------------------
    def _save_state(self):
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append({
            "material_types": self.material_types.copy(),
            "temperature": self.temperature.copy(),
            "pressure": self.pressure.copy(),
            "pressure_offset": self.pressure_offset.copy(),
            "age": self.age.copy(),
            "time": self.time,
            "power_density": self.power_density.copy(),
        })

    # Time-series stub (modules may record)
    def _record_time_series_data(self):
        pass

    def _get_mobile_mask(self, temperature_threshold: float | None = None) -> np.ndarray:
        """Return boolean mask of *mobile* (liquid or gas) voxels.

        A voxel is considered *mobile* when its temperature exceeds
        ``temperature_threshold`` (defaults to 800 °C in Kelvin) **and** it is
        not SPACE.  This matches the legacy behaviour used by density
        stratification and buoyancy routines.
        """
        if temperature_threshold is None:
            temperature_threshold = 800.0 + 273.15  # 800 °C expressed in Kelvin

        return (self.temperature > temperature_threshold) & (self.material_types != MaterialType.SPACE)