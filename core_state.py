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

from typing import Optional, Tuple, Dict, List, Any
import logging

import numpy as np
from scipy import ndimage

try:
    from .materials import MaterialType, MaterialDatabase
    from .heat_transfer import HeatTransfer
    from .fluid_dynamics import FluidDynamics
    from .material_processes import MaterialProcesses
    from .atmospheric_processes import AtmosphericProcesses
except ImportError:  # fallback for standalone unit tests
    from materials import MaterialType, MaterialDatabase
    from heat_transfer import HeatTransfer
    from fluid_dynamics import FluidDynamics
    from material_processes import MaterialProcesses
    from atmospheric_processes import AtmosphericProcesses


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
        cell_depth: Optional[float] = None,
        quality: int = 1,
        log_level: str | int = "INFO",
    ) -> None:
        self.width = width
        self.height = height
        self.cell_size = float(cell_size)
        
        # Cell depth for 2.5D simulation (default: simulation width)
        self.cell_depth = float(cell_depth) if cell_depth is not None else float(width * cell_size)

        # ---------- logger -------------------------------------------------
        self.logger = logging.getLogger(f"GeoGame_{id(self)}")
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

        # ---------- simulation parameters (SI) ---------------------------
        self.time = 0.0      # seconds since start
        self.dt = 10_000.0   # seconds per macro-step (≈2.8 h)

        # Physical constant (SI)
        self.stefan_boltzmann = 5.670e-8  # W/(m²·K⁴)

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
        self.average_gravity = (0, 0) # m/s^2 for external sources
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
        self._material_props_cache: Dict[MaterialType, Tuple[float, float, float]] = {}
        self._properties_dirty = True
        
        # ---------- physics toggles ---------------------------------------
        self.enable_self_gravity = True
        self.external_gravity = (0, 0)  # External gravity field (m/s²)
        
        # Control flags for physics modules
        self.enable_internal_heating = True
        self.enable_solar_heating = True
        self.enable_radiative_cooling = True
        self.enable_heat_diffusion = True
        self.enable_material_processes = True
        self.enable_atmospheric_processes = True
        self.enable_weathering = True
        self.enable_solid_drag = True
        self.debug_rigid_bodies = False

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
        self.fluid_migration_probability: float = 0.25

        # minimal history for undo (optional)
        self.history: List[Dict[str, Any]] = []
        self.max_history = 50

        # Diffusion stencil selection ('radius1' or 'radius2')
        self.diffusion_stencil = "radius2"

        # ------------------------------------------------------------------
        # Place-holder time-series buffers so the visualiser's graphs work.
        # Modules are free to push to these lists at their leisure.
        # ------------------------------------------------------------------
        self.time_series_data = {
            'time': [],
            'avg_temperature': [],
            'max_temperature': [],
            'min_temperature': [],
            'avg_pressure': [],
            'max_pressure': [],
            'min_pressure': [],
            'thermal_flux_solar': [],
            'thermal_flux_radiative': [],
            'thermal_flux_internal': [],
            'thermal_flux_net': [],
            'material_counts': [],  # Dict of material type counts per timestep
            'center_of_mass_x': [],
            'center_of_mass_y': [],
            'planet_radius': [],
            'atmospheric_mass': [],
            'total_energy': [],
        }
        self.max_time_series_length = 1000  # Keep last 1000 timesteps

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
        self.velocity_integration_method = "explicit_euler"  # velocity update method

        # Constant external gravitational field (m/s²).  Positive y is downward in
        # array coordinates, so Earth-like gravity uses (0, +9.81).
        # Modules add these components to the self-gravity field produced by the
        # Poisson solve so that laboratory-scale scenarios still experience a
        # meaningful buoyancy force even when the radial self-gravity is nearly
        # zero.
        self.external_gravity: tuple[float, float] = self.average_gravity

        # Toggle for applying empirical solid drag in unified kinematics
        self.enable_solid_drag: bool = True
        
        # ------------------------------------------------------------------
        # Heat transfer control flags
        # ------------------------------------------------------------------
        self.enable_internal_heating: bool = True
        self.enable_solar_heating: bool = True  
        self.enable_radiative_cooling: bool = True
        self.enable_heat_diffusion: bool = True

        # Initialize modules (note: order matters)
        self.heat_transfer = HeatTransfer(self)
        self.fluid_dynamics = FluidDynamics(self)
        self.material_processes = MaterialProcesses(self)
        self.atmospheric_processes = AtmosphericProcesses(self)

        # Planetary setup
        self._setup_planetary_conditions()
        self._update_material_properties()
        self._calculate_center_of_mass()

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
        
        # Calculate force differences for each swap to prioritize
        if hasattr(self, 'force_x') and hasattr(self, 'force_y'):
            fx, fy = self.force_x, self.force_y
            force_diffs = []
            for i in range(len(src_y)):
                fsrc_x = fx[src_y[i], src_x[i]]
                fsrc_y = fy[src_y[i], src_x[i]]
                ftgt_x = fx[tgt_y[i], tgt_x[i]]
                ftgt_y = fy[tgt_y[i], tgt_x[i]]
                dFx = fsrc_x - ftgt_x
                dFy = fsrc_y - ftgt_y
                F_net = np.hypot(dFx, dFy)
                force_diffs.append(F_net)
            force_diffs = np.array(force_diffs)
            
            # Sort by force magnitude (highest first)
            priority_order = np.argsort(force_diffs)[::-1]
        else:
            # No forces available, use random order
            priority_order = np.arange(len(src_y))
            np.random.shuffle(priority_order)
        
        # Process swaps in priority order, marking cells as used
        used_cells = set()
        keep_indices = []
        
        for idx in priority_order:
            src_cell = (src_y[idx], src_x[idx])
            tgt_cell = (tgt_y[idx], tgt_x[idx])
            
            # Skip if either cell is already used
            if src_cell in used_cells or tgt_cell in used_cells:
                continue
            
            # Mark cells as used and keep this swap
            used_cells.add(src_cell)
            used_cells.add(tgt_cell)
            keep_indices.append(idx)
        
        # Convert back to arrays
        keep_indices = np.array(keep_indices)
        if len(keep_indices) > 0:
            return src_y[keep_indices], src_x[keep_indices], tgt_y[keep_indices], tgt_x[keep_indices]
        else:
            return np.array([], dtype=src_y.dtype), np.array([], dtype=src_x.dtype), \
                   np.array([], dtype=tgt_y.dtype), np.array([], dtype=tgt_x.dtype)

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
        
        # Use cell_depth for proper 3D mass calculation
        cell_volume = self.cell_size ** 2 * self.cell_depth
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
        """Record comprehensive time-series data for analytics and graphing"""
        # Skip recording if time hasn't advanced
        if len(self.time_series_data['time']) > 0 and self.time == self.time_series_data['time'][-1]:
            return
            
        # Get non-space mask for statistics
        non_space_mask = (self.material_types != MaterialType.SPACE)
        
        # Temperature statistics
        if np.any(non_space_mask):
            temps = self.temperature[non_space_mask]
            avg_temp = float(np.mean(temps))
            max_temp = float(np.max(temps))
            min_temp = float(np.min(temps))
        else:
            avg_temp = max_temp = min_temp = self.space_temperature
            
        # Pressure statistics
        if np.any(non_space_mask):
            pressures = self.pressure[non_space_mask]
            avg_pressure = float(np.mean(pressures))
            max_pressure = float(np.max(pressures))
            min_pressure = float(np.min(pressures))
        else:
            avg_pressure = max_pressure = min_pressure = 0.0
            
        # Material composition counts
        material_counts = {}
        unique_materials, counts = np.unique(self.material_types, return_counts=True)
        for material, count in zip(unique_materials, counts):
            material_counts[material.value] = int(count)
            
        # Atmospheric mass (air + water vapor)
        atmospheric_mask = ((self.material_types == MaterialType.AIR) | 
                          (self.material_types == MaterialType.WATER_VAPOR))
        atmospheric_mass = float(np.sum(self.density[atmospheric_mask])) if np.any(atmospheric_mask) else 0.0
        
        # Planet radius (distance from center to farthest non-space cell)
        if np.any(non_space_mask):
            non_space_coords = np.where(non_space_mask)
            distances = np.sqrt((non_space_coords[1] - self.center_of_mass[0])**2 + 
                              (non_space_coords[0] - self.center_of_mass[1])**2)
            planet_radius = float(np.max(distances) * self.cell_size)
        else:
            planet_radius = 0.0
            
        # Total thermal energy
        if np.any(non_space_mask):
            thermal_energy = np.sum(self.density[non_space_mask] * 
                                  self.specific_heat[non_space_mask] * 
                                  self.temperature[non_space_mask])
            # Use cell_depth for proper 3D volume
            cell_volume = self.cell_size ** 2 * self.cell_depth
            total_energy = float(thermal_energy * cell_volume)  # J
        else:
            total_energy = 0.0
            
        # Record all data
        self.time_series_data['time'].append(self.time)
        self.time_series_data['avg_temperature'].append(avg_temp)
        self.time_series_data['max_temperature'].append(max_temp)
        self.time_series_data['min_temperature'].append(min_temp)
        self.time_series_data['avg_pressure'].append(avg_pressure)
        self.time_series_data['max_pressure'].append(max_pressure)
        self.time_series_data['min_pressure'].append(min_pressure)
        self.time_series_data['thermal_flux_solar'].append(self.thermal_fluxes.get('solar_input', 0.0))
        self.time_series_data['thermal_flux_radiative'].append(self.thermal_fluxes.get('radiative_output', 0.0))
        self.time_series_data['thermal_flux_internal'].append(self.thermal_fluxes.get('internal_heating', 0.0))
        self.time_series_data['thermal_flux_net'].append(self.thermal_fluxes.get('net_flux', 0.0))
        self.time_series_data['material_counts'].append(material_counts)
        self.time_series_data['center_of_mass_x'].append(self.center_of_mass[0])
        self.time_series_data['center_of_mass_y'].append(self.center_of_mass[1])
        self.time_series_data['planet_radius'].append(planet_radius)
        self.time_series_data['atmospheric_mass'].append(atmospheric_mass)
        self.time_series_data['total_energy'].append(total_energy)
        
        # Trim old data to keep memory usage reasonable
        for key in self.time_series_data:
            if len(self.time_series_data[key]) > self.max_time_series_length:
                self.time_series_data[key] = self.time_series_data[key][-self.max_time_series_length:]

    def _get_mobile_mask(self, temperature_threshold: float | None = None) -> np.ndarray:
        """Get mask for mobile (liquid/gas) materials"""
        if temperature_threshold is None:
            temperature_threshold = 800.0 + 273.15  # Default: 800°C in Kelvin

        return ((self.temperature > temperature_threshold) &
                (self.material_types != MaterialType.SPACE))

    def calculate_effective_density(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate temperature-dependent effective densities using thermal expansion.
        
        This method implements the physics formula:
        ρ_eff = ρ₀ / (1 + β(T - T₀))
        
        Where:
        - ρ₀ = reference density from material database
        - β = volumetric thermal expansion coefficient 
        - T = current temperature
        - T₀ = reference temperature (273.15K)
        
        Args:
            temperature: Temperature grid in Kelvin
            
        Returns:
            Effective density grid accounting for thermal expansion
        """
        # Initialize effective density grid
        effective_density = np.zeros_like(self.density)
        non_space_mask = (self.material_types != MaterialType.SPACE)

        if not np.any(non_space_mask):
            return effective_density

        # Get coordinates and properties for all non-space cells
        non_space_coords = np.where(non_space_mask)
        materials = self.material_types[non_space_coords]
        temperatures = temperature[non_space_coords]
        base_densities = self.density[non_space_coords]

        # Vectorized thermal expansion calculation using material properties
        # ρ_eff = ρ₀ / (1 + β(T - T₀)) where β is volumetric expansion coefficient
        expansion_coeffs = np.array([
            self.material_db.get_properties(mat).thermal_expansion 
            for mat in materials.flat
        ])
        
        # Calculate volumetric expansion factor
        volumetric_expansion = 1.0 + expansion_coeffs * (temperatures - self.reference_temperature)
        
        # Prevent division by zero/negative and extreme values
        volumetric_expansion = np.maximum(0.1, volumetric_expansion)
        
        # Calculate effective densities
        effective_densities = base_densities / volumetric_expansion
        effective_densities = np.maximum(0.01, effective_densities)  # Prevent negative density

        # Fill the effective density grid
        effective_density[non_space_coords] = effective_densities

        return effective_density

    # TODO: Implement porosity-dependent fluid flow and material properties
    # Porosity affects:
    # - Fluid permeability and flow rates through porous materials
    # - Effective thermal conductivity (lower for high porosity materials)
    # - Mechanical strength (higher porosity = lower strength)
    # - Density calculations (effective density = (1-porosity)*solid_density + porosity*fluid_density)
    # - Weathering rates (higher porosity = faster weathering)
    # - Heat capacity (mixture of solid and fluid components)
    # This would enhance realism for sedimentary rocks, pumice, and weathered materials

    def _setup_planetary_conditions(self):
        # Implementation of _setup_planetary_conditions method
        pass