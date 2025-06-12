"""
Core simulation engine for 2D geological processes.
Handles heat transfer, pressure calculation, and rock state evolution.
"""

import numpy as np
import logging
from typing import Tuple, Optional
try:
    from .materials import MaterialType, MaterialDatabase
except ImportError:
    from materials import MaterialType, MaterialDatabase
from scipy import ndimage
import time

class GeologySimulation:
    """Main simulation engine for 2D geological processes"""

    def __init__(self, width: int, height: int, cell_size: float = 50.0, quality: int = 1, log_level: str = "INFO"):
        """
        Initialize simulation grid

        Args:
            width: Grid width in cells
            height: Grid height in cells
            cell_size: Size of each cell in meters
            quality: Quality setting (1=high quality, 2=balanced, 3=fast)
            log_level: Logging level ("INFO" or "DEBUG")
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size  # meters per cell

        # Setup logging
        self.logger = logging.getLogger(f"GeologySimulation_{id(self)}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # Clean formatter without timestamp and logger name
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Thermal flux tracking for debugging
        self.thermal_fluxes = {
            'solar_input': 0.0,      # W
            'radiative_output': 0.0,  # W
            'internal_heating': 0.0,  # W
            'atmospheric_heating': 0.0, # W
            'net_flux': 0.0          # W
        }

        # Performance configuration
        self._setup_performance_config(quality)

        # Core simulation grids
        self.material_types = np.full((height, width), MaterialType.GRANITE, dtype=object)
        self.temperature = np.zeros((height, width), dtype=np.float64)
        self.pressure = np.zeros((height, width), dtype=np.float64)
        self.pressure_offset = np.zeros((height, width), dtype=np.float64)  # User-applied pressure changes
        self.age = np.zeros((height, width), dtype=np.float64)

        # Power density tracking for visualization (W/m³)
        self.power_density = np.zeros((height, width), dtype=np.float64)

        # Derived properties (computed from material types)
        self.density = np.zeros((height, width), dtype=np.float64)
        self.thermal_conductivity = np.zeros((height, width), dtype=np.float64)
        self.specific_heat = np.zeros((height, width), dtype=np.float64)

        # Simulation parameters
        self.time = 0.0
        self.dt = 1.0  # Further reduced to 1 year for DuFort-Frankel stability

        # Unit conversion constants
        self.seconds_per_year = 365.25 * 24 * 3600
        self.stefan_boltzmann_geological = 5.67e-8 * self.seconds_per_year  # W/(m²⋅K⁴) → J/(year⋅m²⋅K⁴)

        # Planetary parameters
        self.planet_radius_fraction = 0.8  # Fraction of grid width for initial planet
        self.planet_center = (width // 2, height // 2)
        self.center_of_mass = (width / 2, height / 2)  # Will be calculated dynamically

        # Material database
        self.material_db = MaterialDatabase()

        # Radiative cooling method options:
        # - "linearized_stefan_boltzmann": Linearized approximation Q = h(T - T_space) for stability
        # - "newton_raphson_implicit": Full Stefan-Boltzmann T^4 with Newton-Raphson implicit solver
        self.radiative_cooling_method = "newton_raphson_implicit"  # Default: most accurate
        
        # Thermal diffusion method options:
        # - "explicit_euler": Standard explicit finite difference (stable, fast)
        self.thermal_diffusion_method = "explicit_euler"  # Default: stable and efficient
        
        # Thermal diffusion stability settings
        self.max_diffusion_substeps = 50               # Maximum substeps for diffusion stability
        
        # Atmospheric absorption method options:
        # - "directional_sweep": Fast O(N) Amanatides & Woo sweep with proper shadowing
        #   (all legacy methods removed – keep dispatcher for future extensions)
        self.atmospheric_absorption_method = "directional_sweep"

        # General physics settings (not performance-dependent)
        self.atmospheric_diffusivity_enhancement = 5.0  # Enhanced diffusivity in the atomsphere to mimic convection
        self.atmospheric_convection_mixing = 0.3        # Fraction of temperature difference to mix per step (fast convection)
        self.interface_diffusivity_enhancement = 1.5    # Enhanced heat transfer at material interfaces
        self.surface_radiation_depth_fraction = 0.1     # Fraction of cell depth that participates in surface radiation
        self.radiative_cooling_efficiency = 0.9         # Cooling efficiency factor for Stefan-Boltzmann radiation (increased for better balance)
        
        # Clamp for unrealistic vacuum thermal diffusivity to maintain numerical stability (m²/s)
        # Typical rocks/water have α ~1e-6–1e-4; we cap at a generous 1e-3 to stay safe
        self.max_thermal_diffusivity = 1e-3

        # Temperature constants
        self.space_temperature = 2.7                    # Cosmic background temperature (K)
        self.reference_temperature = 273.15             # Reference temperature for thermal expansion (K)
        self.core_temperature = 1200.0 + 273.15         # Initial planetary core temperature (K) - warmer for stability
        self.surface_temperature = 50.0 + 273.15        # Initial planetary surface temperature (K) - warmer for stability
        self.temperature_decay_constant = 2.0           # Temperature gradient decay factor - steeper gradient
        self.melting_temperature = 1200 + 273.15        # General melting temperature threshold (K)
        self.hot_solid_temperature_threshold = 1200.0   # °C: solids hotter than this behave as fluids in density stratification
        self.core_heating_depth_scale = 0.5             # Exponential scale factor for core heating vs depth

        # Pressure constants
        self.surface_pressure = 0.1                     # Surface pressure (MPa)
        self.atmospheric_scale_height = 8400            # Atmospheric scale height (m)
        self.average_gravity = 9.81                     # Average gravitational acceleration (m/s²)
        self.average_solid_density = 3000               # Average solid rock density (kg/m³)
        self.average_fluid_density = 2000               # Average fluid density (kg/m³)

        # Solar and greenhouse constants (balanced for stable temperatures)
        self.solar_constant = 50                       # Solar constant (W/m²)
        self.solar_angle = 90.0                         # Solar angle in degrees: 0°=equator, +90°=north pole, -90°=south pole
        self.planetary_distance_factor = 1             # Distance factor for solar intensity - further reduced to prevent hot spots (was 0.0001)
        self.base_greenhouse_effect = 0.2              # Base greenhouse effect fraction - increased to retain more heat
        self.max_greenhouse_effect = 0.8               # Maximum greenhouse effect fraction - increased
        self.greenhouse_vapor_scaling = 1000.0         # Water vapor mass scaling for greenhouse effect

        # Material mobility probabilities
        self.gravitational_fall_probability = 1.0       # Initial fall probability for collapse
        self.gravitational_fall_probability_later = 1.0 # Later fall probability for collapse
        self.fluid_migration_probability = 0.5          # Air/fluid migration probability
        self.density_swap_probability = 0.5             # Density stratification swap probability

        # Planet formation constants
        self.ice_region_depth = 0.6                    # Depth fraction where ice becomes common
        self.mantle_region_depth = 0.3                 # Depth fraction for mantle region
        self.surface_region_depth = 0.85               # Depth fraction for surface variations
        self.atmosphere_formation_probability = 0.3    # Probability of atmosphere formation at surface
        self.surface_variation_noise = 0.1             # Surface irregularity noise factor

        # History for time reversal
        self.max_history = 1000
        self.history = []
        self.history_step = 0

        # Time-series data tracking for graphs
        self.time_series = {
            'time': [],
            'avg_temperature': [],
            'max_temperature': [],
            'total_energy': [],
            'net_power': [],
            'greenhouse_factor': [],
            'planet_albedo': []
        }
        self.max_time_series = 500  # Keep last 500 data points

        # Performance optimization caches
        self._material_props_cache = {}  # Cache for material property lookups
        self._properties_dirty = True    # Flag to track when material properties need updating

        # Thermal diffusion state
        self._adaptive_time_steps = []   # Track adaptive time stepping history

        # Pre-compute neighbor arrays for vectorized operations
        self._setup_neighbors()

        # Initialize as an emergent planet
        self._setup_planetary_conditions()
        self._properties_dirty = True
        self._update_material_properties()
        self._calculate_center_of_mass()

        # Diffusion stencil options:
        # - "radius1": classic 8-neighbour (fast, slightly axis-biased)
        # - "radius2": 13-point isotropic stencil (default)
        self.diffusion_stencil = "radius2"

        # interval (macro-steps) between deterministic settle sweeps
        self.settle_interval = 1
        self._need_settle = True  # trigger initial settling once

        # Logging / instrumentation controls
        self.logging_enabled = False   # Toggle with visualizer (shortcut 'L')
        self._perf_times: dict[str, float] = {}  # Performance timings per step

        # Max cells a chunk can fall in one settle pass (set to float('inf') for unlimited)
        self.terminal_settle_velocity = 3

    def _setup_performance_config(self, quality: int):
        """Setup performance configuration based on quality level"""
        if quality == 1:
            # Full quality - maximum accuracy
            self.process_fraction_mobile = 1.0      # Process all mobile cells
            self.process_fraction_solid = 1.0       # Process all solid cells
            self.process_fraction_air = 1.0         # Process all air cells
            self.process_fraction_water = 1.0       # Process all liquid cells (water/magma)
            self.step_interval_differentiation = 1  # Every step
            self.step_interval_collapse = 1         # Every step
            self.step_interval_fluid = 1            # Every step
            self.density_sample_fraction = 1.0     # Process all cells for density sorting
            self.density_min_sample_size = 100     # Minimum cells to process
            self.density_ratio_threshold = 1.05    # Minimum 5% density difference for swapping
            self.max_fall_steps = 5                # Maximum gravitational collapse iterations
            self.enable_weathering = True           # Enable weathering
            self.neighbor_count = 8                 # Use 8 neighbors for full accuracy

        elif quality == 2:
            # Balanced quality
            self.process_fraction_mobile = 0.5      # Process 50% of mobile cells
            self.process_fraction_solid = 0.5       # Process 50% of solid cells
            self.process_fraction_air = 0.5         # Process 50% of air cells
            self.process_fraction_water = 0.5       # Process 50% of liquid cells
            self.step_interval_differentiation = 2  # Every 2nd step
            self.step_interval_collapse = 2         # Every 2nd step
            self.step_interval_fluid = 3            # Every 3rd step
            self.density_sample_fraction = 0.5     # Process 50% of cells for density sorting
            self.density_min_sample_size = 50      # Minimum cells to process
            self.density_ratio_threshold = 1.1     # Minimum 10% density difference for swapping
            self.max_fall_steps = 3                # Moderate gravitational collapse iterations
            self.enable_weathering = False          # Disable weathering
            self.neighbor_count = 8                 # Use 8 neighbors for good accuracy

        elif quality == 3:
            # Fast quality - maximum performance
            self.process_fraction_mobile = 0.2      # Process 20% of mobile cells
            self.process_fraction_solid = 0.33      # Process 33% of solid cells
            self.process_fraction_air = 0.25        # Process 25% of air cells
            self.process_fraction_water = 0.25      # Process 25% of liquid cells
            self.step_interval_differentiation = 3  # Every 3rd step
            self.step_interval_collapse = 4         # Every 4th step
            self.step_interval_fluid = 5            # Every 5th step
            self.density_sample_fraction = 0.2     # Process 20% of cells for density sorting
            self.density_min_sample_size = 25      # Minimum cells to process
            self.density_ratio_threshold = 1.2     # Minimum 20% density difference for swapping
            self.max_fall_steps = 2                # Minimal gravitational collapse iterations
            self.enable_weathering = False          # Disable weathering
            self.neighbor_count = 4                 # Use 4 neighbors for performance

        else:
            raise ValueError(f"Unknown quality level: {quality}. Use 1 (full), 2 (balanced), or 3 (fast)")

    def _setup_neighbors(self):
        """Pre-compute neighbor offset arrays for efficient operations"""
        # 4-neighbor offsets (cardinal directions)
        self.neighbors_4 = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])

        # 8-neighbor offsets (cardinal + diagonal)
        self.neighbors_8 = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])

        # Isotropic diffusion stencil: treat all 8 neighbours equally (eliminates octagonal bias)
        self.distance_factors_8 = np.ones(8)

        # Pre-compute coordinate grids for vectorized operations
        self.y_coords, self.x_coords = np.ogrid[:self.height, :self.width]

        # Create circular morphological kernels to reduce grid artifacts
        self._circular_kernel_3x3 = self._create_circular_kernel(3)
        self._circular_kernel_5x5 = self._create_circular_kernel(5)

        # Collapse kernels for geological processes
        self._collapse_kernel_4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)  # 4-neighbor
        # Use a 5×5 circular kernel for more isotropic neighbour detection (reduces octagonal bias)
        self._collapse_kernel_8 = self._circular_kernel_5x5.copy()
        self._collapse_kernel_8[2, 2] = False  # exclude centre

        # Radius-2 isotropic Laplacian (13-point) – coefficients sum to 0
        lap_kernel = (1.0/6.0) * np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2,-16,2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0]], dtype=np.float64)
        self._laplacian_kernel_radius2 = lap_kernel

    def _create_circular_kernel(self, size: int) -> np.ndarray:
        """Create a circular kernel for morphological operations to reduce grid artifacts"""
        kernel = np.zeros((size, size), dtype=bool)
        center = size // 2
        radius = center + 0.5  # Slightly larger than perfect circle for connectivity

        for i in range(size):
            for j in range(size):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                if distance <= radius:
                    kernel[i, j] = True

        return kernel

    def _setup_planetary_conditions(self):
        """Set up initial planetary conditions with emergent circular shape"""
        # Initialize everything as space
        self.material_types.fill(MaterialType.SPACE)
        self.temperature.fill(self.space_temperature)  # Space temperature (~3K cosmic background radiation)
        self.pressure.fill(0.0)  # Vacuum

        # Calculate initial planet radius in cells
        planet_radius = self._get_planet_radius()
        center_x, center_y = self.planet_center

        # Create roughly circular planet with layered structure
        for y in range(self.height):
            for x in range(self.width):
                # Distance from planet center
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

                if distance <= planet_radius:
                    # Within planet - set material type and temperature based on distance from center
                    relative_depth = distance / planet_radius  # 0 at center, 1 at surface

                    # Temperature: "dirty iceball" formation - cold surface, hot core (in Kelvin)
                    temp_gradient = (self.core_temperature - self.surface_temperature) * np.exp(-self.temperature_decay_constant * relative_depth)
                    self.temperature[y, x] = self.surface_temperature + temp_gradient

                    # Add some randomness for more interesting geology (further reduced to preserve ice)
                    self.temperature[y, x] += np.random.normal(0, 5)

                    # "Dirty iceball" formation - include ice pockets in outer regions
                    if relative_depth > self.ice_region_depth:  # Outer regions - more ice
                        material_types = [
                            MaterialType.GRANITE, MaterialType.BASALT, MaterialType.SANDSTONE,
                            MaterialType.LIMESTONE, MaterialType.SHALE, MaterialType.ICE, MaterialType.ICE, MaterialType.ICE  # More ice in outer regions
                        ]
                    elif relative_depth > self.mantle_region_depth:  # Middle regions - some ice
                        material_types = [
                            MaterialType.GRANITE, MaterialType.BASALT, MaterialType.GNEISS,
                            MaterialType.SANDSTONE, MaterialType.LIMESTONE, MaterialType.SHALE,
                            MaterialType.MARBLE, MaterialType.QUARTZITE, MaterialType.ICE  # Some ice
                        ]
                    else:  # Inner core - mostly rocks, minimal ice
                        material_types = [
                            MaterialType.GRANITE, MaterialType.BASALT, MaterialType.GNEISS,
                            MaterialType.SANDSTONE, MaterialType.LIMESTONE, MaterialType.SHALE,
                            MaterialType.MARBLE, MaterialType.QUARTZITE, MaterialType.SCHIST,
                            MaterialType.SLATE, MaterialType.ANDESITE
                        ]

                    self.material_types[y, x] = np.random.choice(material_types)

                    # Convert hot material to magma based on temperature
                    if self.temperature[y, x] > self.melting_temperature:  # Hot enough to melt
                        self.material_types[y, x] = MaterialType.MAGMA

                    # Add some surface variation (not perfectly circular)
                    if relative_depth > self.surface_region_depth:  # Near surface
                        # Add some randomness to make surface irregular
                        noise = np.random.random() * self.surface_variation_noise
                        if relative_depth + noise > 1.0:
                            # Sometimes extend into space or create atmosphere
                            if np.random.random() < self.atmosphere_formation_probability:  # Chance of atmosphere
                                self.material_types[y, x] = MaterialType.AIR
                                self.temperature[y, x] = self.surface_temperature

        # Calculate initial pressure using gravitational model
        self._calculate_planetary_pressure()

    def _update_material_properties(self):
        """Update of material property grids using efficient numpy operations"""
        if not self._properties_dirty:
            return  # Skip if properties haven't changed

        # Get unique material types to minimize property lookups
        # Convert to set to avoid sorting issues with MaterialType enums
        unique_materials = set(self.material_types.flatten())

        # Pre-compute properties for each unique material type
        prop_lookup = {}
        for material in unique_materials:
            if material not in self._material_props_cache:
                props = self.material_db.get_properties(material)
                self._material_props_cache[material] = (props.density, props.thermal_conductivity, props.specific_heat)
            prop_lookup[material] = self._material_props_cache[material]

        # Vectorized assignment using advanced indexing
        for material, (density, k_thermal, c_heat) in prop_lookup.items():
            mask = (self.material_types == material)
            if np.any(mask):
                self.density[mask] = density
                self.thermal_conductivity[mask] = k_thermal
                self.specific_heat[mask] = c_heat

        self._properties_dirty = False

    def _heat_diffusion(self) -> tuple[np.ndarray, float]:
        """Operator splitting approach: solve diffusion, radiation, and sources separately for maximum stability"""
        # Only process non-space cells
        non_space_mask = (self.material_types != MaterialType.SPACE)

        if not np.any(non_space_mask):
            return self.temperature, 1.0

        # Start with current temperature
        working_temp = self.temperature.copy()

        # Step 1: Solve pure diffusion (no sources)
        working_temp, diffusion_stability = self._solve_pure_diffusion(working_temp, non_space_mask)

        # Step 2: Solve radiative cooling using selected method (dispatcher)
        working_temp = self._solve_radiative_cooling(working_temp, non_space_mask)

        # Step 3: Solve other heat sources explicitly (internal, solar, atmospheric)
        working_temp = self._solve_non_radiative_sources(working_temp, non_space_mask)

        # Overall stability factor is dominated by diffusion (radiation and sources are stable)
        overall_stability = diffusion_stability

        # Store debugging info
        self._actual_substeps = getattr(self, '_diffusion_substeps', 1)
        self._actual_effective_dt = self.dt * overall_stability

        # Debug info
        avg_temp_before = np.mean(self.temperature[non_space_mask]) - 273.15 if np.any(non_space_mask) else 0.0
        avg_temp_after = np.mean(working_temp[non_space_mask]) - 273.15 if np.any(non_space_mask) else 0.0
        if self.logging_enabled:
            self.logger.debug(
                f"Diffusion sub-steps: {self._actual_substeps}, ΔT planet avg: {avg_temp_before:6.1f}→{avg_temp_after:6.1f} °C")

        return working_temp, overall_stability

    def _solve_pure_diffusion(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve pure diffusion (no sources) for maximum stability"""
        # Get thermal diffusivity for all cells (α = k / (ρ * cp))
        valid_thermal = (self.density > 0) & (self.specific_heat > 0) & (self.thermal_conductivity > 0)
        thermal_diffusivity = np.zeros_like(self.thermal_conductivity)
        thermal_diffusivity[valid_thermal] = (
            self.thermal_conductivity[valid_thermal] /
            (self.density[valid_thermal] * self.specific_heat[valid_thermal])
        )  # m²/s

        # Enhanced atmospheric convection (much faster heat transfer in gases)
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )
        atmospheric_cells = atmosphere_mask & valid_thermal
        if np.any(atmospheric_cells):
            thermal_diffusivity[atmospheric_cells] *= self.atmospheric_diffusivity_enhancement

        # Enhanced diffusion at material interfaces
        if hasattr(self, 'interface_diffusivity_enhancement'):
            neighbors = self._get_neighbors(4, shuffle=False)
            for dy, dx in neighbors:
                shifted_materials = np.roll(np.roll(self.material_types, dy, axis=0), dx, axis=1)
                interface_mask = (self.material_types != shifted_materials) & non_space_mask
                if np.any(interface_mask):
                    enhancement_mask = interface_mask & valid_thermal
                    if np.any(enhancement_mask):
                        thermal_diffusivity[enhancement_mask] *= self.interface_diffusivity_enhancement

        # Clamp extreme thermal diffusivity values (e.g., near-vacuum cells)
        thermal_diffusivity = np.clip(thermal_diffusivity, 0.0, self.max_thermal_diffusivity)

        # Stability analysis for PURE DIFFUSION only (no sources)
        dx_squared = self.cell_size ** 2
        max_alpha = np.max(thermal_diffusivity[non_space_mask]) if np.any(non_space_mask) else 0.0

        # Pure diffusion stability limit depends on stencil
        stencil_denominator = 4.0 if self.diffusion_stencil == "radius1" else 16.0
        diffusion_dt_limit = dx_squared / (stencil_denominator * max_alpha) if max_alpha > 0 else float('inf')

        # Adaptive time step for diffusion (clamp between min substep and full timestep)
        min_dt_seconds = self.dt * self.seconds_per_year / self.max_diffusion_substeps
        target_dt_seconds = np.clip(diffusion_dt_limit, min_dt_seconds, self.dt * self.seconds_per_year)

        # Convert back to years
        adaptive_dt = target_dt_seconds / self.seconds_per_year
        stability_factor = adaptive_dt / self.dt

        # Use sub-steps for stability
        num_substeps = max(1, min(self.max_diffusion_substeps, int(np.ceil(self.dt / adaptive_dt))))
        actual_effective_dt = self.dt / num_substeps
        actual_stability_factor = actual_effective_dt / self.dt

        # Store debugging info
        self._max_thermal_diffusivity = max_alpha
        self._diffusion_substeps = num_substeps

        # Pure diffusion sub-stepping (much simpler without sources)
        new_temp = temperature.copy()

        for step in range(num_substeps):
            # Pure diffusion step using selected method
            new_temp = self._solve_diffusion_step(new_temp, thermal_diffusivity, actual_effective_dt, non_space_mask)

        return new_temp, actual_stability_factor

    def _solve_diffusion_step(self, temperature: np.ndarray, thermal_diffusivity: np.ndarray,
                             dt: float, non_space_mask: np.ndarray) -> np.ndarray:
        """Master thermal diffusion solver - dispatches to selected implementation method"""
        
        if self.thermal_diffusion_method == "explicit_euler":
            return self._diffusion_step_explicit_euler(temperature, thermal_diffusivity, dt, non_space_mask)
        else:
            raise ValueError(f"Unknown thermal diffusion method: {self.thermal_diffusion_method}. "
                           f"Available options: 'explicit_euler'")

    def _diffusion_step_explicit_euler(self, temperature: np.ndarray, thermal_diffusivity: np.ndarray,
                                      dt: float, non_space_mask: np.ndarray) -> np.ndarray:
        """
        Explicit Euler thermal diffusion step
        
        Method: Forward Euler finite difference for dT/dt = α∇²T
        Advantages: Simple, fast, well-tested
        Disadvantages: Requires small time steps for stability
        """
        # Convert dt to seconds for proper units
        dt_seconds = dt * self.seconds_per_year
        dx_squared = self.cell_size ** 2

        if self.diffusion_stencil == "radius1":
            laplacian = np.zeros_like(temperature)
            for i, (dy, dx) in enumerate(self.neighbors_8):
                neighbor_temp = np.roll(np.roll(temperature, dy, axis=0), dx, axis=1)
                weight = self.distance_factors_8[i]
                laplacian += weight * (neighbor_temp - temperature)
            laplacian /= np.sum(self.distance_factors_8)
        else:  # radius2 isotropic default
            laplacian = ndimage.convolve(temperature, self._laplacian_kernel_radius2, mode="nearest")

        # Zero out Laplacian for space cells (no diffusion)
        laplacian[~non_space_mask] = 0.0

        # Combined diffusion update (vectorized)
        # dT/dt = α∇²T
        diffusion_change = thermal_diffusivity * dt_seconds * laplacian / dx_squared

        # Apply updates only to non-space cells
        new_temp = temperature.copy()
        if np.any(non_space_mask):
            new_temp[non_space_mask] += diffusion_change[non_space_mask]

        return new_temp

    def _apply_atmospheric_convection(self, temperature: np.ndarray) -> np.ndarray:
        """Apply fast atmospheric convection mixing using vectorized convolution (unconditionally stable)"""
        # Find atmospheric materials
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )

        if not np.any(atmosphere_mask):
            return temperature

        # Proper atmospheric-only temperature averaging
        # Only include atmospheric cell temperatures in the averaging, completely exclude non-atmospheric cells
        # Create temperature grid where ONLY atmospheric cells have their actual temperature
        atmo_temp_for_sum = np.where(atmosphere_mask, temperature, 0.0)
        atmo_mask_for_count = atmosphere_mask.astype(np.float64)

        # Use simple 3x3 averaging kernel for neighbor mixing
        mixing_kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],  # Center cell excluded (we want neighbors only)
            [1, 1, 1]
        ], dtype=np.float64)

        # Apply convolution to get sum of atmospheric neighbor temperatures and counts
        neighbor_atmo_temp_sum = ndimage.convolve(atmo_temp_for_sum, mixing_kernel, mode='constant', cval=0.0)
        neighbor_atmo_count = ndimage.convolve(atmo_mask_for_count, mixing_kernel, mode='constant', cval=0.0)

        # Calculate average temperature of atmospheric neighbors only
        valid_atmo_neighbors = neighbor_atmo_count > 0
        avg_atmo_neighbor_temp = np.zeros_like(temperature)
        avg_atmo_neighbor_temp[valid_atmo_neighbors] = (
            neighbor_atmo_temp_sum[valid_atmo_neighbors] / neighbor_atmo_count[valid_atmo_neighbors]
        )

        # Apply mixing only to atmospheric cells that have atmospheric neighbors
        mixing_mask = atmosphere_mask & valid_atmo_neighbors

        if np.any(mixing_mask):
            # Vectorized mixing: T_new = T_old + mixing_fraction * (T_avg_atmo_neighbors - T_old)
            temp_diff = avg_atmo_neighbor_temp[mixing_mask] - temperature[mixing_mask]
            temperature[mixing_mask] += self.atmospheric_convection_mixing * temp_diff

        return temperature

    def _calculate_center_of_mass(self):
        """Calculate center of mass for the planet using vectorized operations"""
        # Only consider cells that contain matter (not space)
        matter_mask = (self.material_types != MaterialType.SPACE)

        if not np.any(matter_mask):
            return  # No matter, keep current center

        # Get coordinates of matter cells
        matter_y, matter_x = np.where(matter_mask)

        # Calculate cell masses (vectorized)
        cell_volume = self.cell_size ** 2  # 2D simulation
        cell_masses = self.density[matter_mask] * cell_volume

        # Calculate center of mass (vectorized)
        total_mass = np.sum(cell_masses)
        if total_mass > 0:
            center_x = np.sum(cell_masses * matter_x) / total_mass
            center_y = np.sum(cell_masses * matter_y) / total_mass
            self.center_of_mass = (center_x, center_y)

    def _calculate_planetary_pressure(self):
        """Calculate pressure using vectorized gravitational model"""
        # Reset all pressures
        self.pressure.fill(0.0)

        # Get distance array for all cells (vectorized)
        distances = self._get_distances_from_center()
        distances_m = distances * self.cell_size
        distances_m = np.maximum(distances_m, self.cell_size)  # Avoid division by zero

        # Create masks for different material types (vectorized)
        space_mask = (self.material_types == MaterialType.SPACE)

        # Gas materials: exponential atmospheric pressure
        gas_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )

        # Fluid materials: hydrostatic pressure
        fluid_mask = (
            (self.material_types == MaterialType.WATER) |
            (self.material_types == MaterialType.MAGMA)
        )

        # Solid materials: lithostatic pressure (everything else that's not space)
        solid_mask = ~(space_mask | gas_mask | fluid_mask)

        # Space cells: vacuum pressure (already zero)
        # self.pressure[space_mask] = 0.0  # Already zero from fill

        # Gas materials: exponential atmosphere (vectorized)
        if np.any(gas_mask):
            surface_distance = self._get_planet_radius()
            height_above_surface = np.maximum(0, distances[gas_mask] - surface_distance) * self.cell_size
            self.pressure[gas_mask] = self.surface_pressure * np.exp(-height_above_surface / self.atmospheric_scale_height)

        # Fluid materials: hydrostatic pressure (vectorized)
        if np.any(fluid_mask):
            surface_distance = self._get_planet_radius()
            depth = np.maximum(0, surface_distance - distances[fluid_mask]) * self.cell_size

            # Use fluid density for hydrostatic pressure
            hydrostatic_pressure = self.average_fluid_density * self.average_gravity * depth / 1e6  # Convert to MPa

            self.pressure[fluid_mask] = np.maximum(self.surface_pressure, hydrostatic_pressure)

        # Solid materials: lithostatic pressure (vectorized)
        if np.any(solid_mask):
            surface_distance = self._get_planet_radius()
            depth = np.maximum(0, surface_distance - distances[solid_mask]) * self.cell_size

            # Use vectorized calculation for depth-based pressure
            lithostatic_pressure = self.average_solid_density * self.average_gravity * depth / 1e6  # Convert to MPa

            self.pressure[solid_mask] = np.maximum(self.surface_pressure, lithostatic_pressure)

        # Add user-applied pressure offsets
        self.pressure += self.pressure_offset

    def _apply_metamorphism(self):
        """Apply metamorphism using efficient boolean mask operations"""
        changes_made = False

        # Skip space cells
        non_space_mask = (self.material_types != MaterialType.SPACE)

        # General transition system - applies to all materials (vectorized)
        temp_celsius = self.temperature - 273.15  # Convert K to C for transition system
        pressure_mpa = self.pressure / 1e6  # Convert Pa to MPa

        # Process only material types that are actually present in the grid
        # This avoids unnecessary work and prevents edge cases where newly
        # created cache entries (e.g., MAGMA) could interact with cells that
        # are not of that type.
        transition_materials = set(self.material_types[non_space_mask].flat)

        for material_type in transition_materials:
            material_mask = (self.material_types == material_type)
            if not np.any(material_mask):
                continue

            props = self.material_db.get_properties(material_type)
            if not props.transitions:
                continue

            # Check each transition for this material type
            for transition in props.transitions:
                # Create vectorized condition check
                condition_mask = (
                    material_mask &
                    (temp_celsius >= transition.min_temp) &
                    (temp_celsius <= transition.max_temp) &
                    (pressure_mpa >= transition.min_pressure) &
                    (pressure_mpa <= transition.max_pressure)
                )

                if np.any(condition_mask):
                    self.material_types[condition_mask] = transition.target
                    changes_made = True

        # Rock melting and magma cooling are handled by the general transition system above
        # This provides more flexibility and allows materials to have different melting points

        # Mark properties as dirty if changes were made
        if changes_made:
            self._properties_dirty = True

        return changes_made

    def _get_distances_from_center(self, center_x: float = None, center_y: float = None) -> np.ndarray:
        """Get distance array from center point (or center of mass if not specified)"""
        if center_x is None or center_y is None:
            center_x, center_y = self.center_of_mass

        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        return np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

    def _get_mobile_mask(self, temperature_threshold: float = None) -> np.ndarray:
        """Get mask for mobile (liquid/gas) materials"""
        if temperature_threshold is None:
            temperature_threshold = 800.0 + 273.15  # Default: 800°C in Kelvin

        return ((self.temperature > temperature_threshold) &
                (self.material_types != MaterialType.SPACE))

    def _get_solid_mask(self) -> np.ndarray:
        """Return boolean grid of *rigid* solids.

        Historically this function treated every non‐SPACE cell as "solid",
        which incorrectly included fluids such as AIR, WATER, MAGMA, etc.  That
        caused several physics artefacts – most notably the chunk-settling
        routine (_settle_unsupported_chunks) would attempt to drop molten
        magma or gas pockets inward where they exchanged heat with much
        hotter rock, leading to runaway temperatures in isolated fluid cells.

        The corrected implementation asks the material database for the
        `is_solid` flag so only genuinely rigid phases participate in
        settling and similar solid-only algorithms.
        """

        # Vectorised lookup of material rigidity (solids = True, fluids = False)
        unique_mats = set(self.material_types.flatten())
        solid_lookup = {m: self.material_db.get_properties(m).is_solid for m in unique_mats}
        return np.vectorize(solid_lookup.get)(self.material_types)

    def _get_solid_material_mask(self) -> np.ndarray:
        """Get mask for solid materials (excluding space, air, water, magma)"""
        solid_mask = np.zeros_like(self.material_types, dtype=bool)

        # Get unique material types to avoid repeated property lookups
        # Convert to set to avoid sorting issues with MaterialType enums
        unique_materials = set(self.material_types.flatten())

        for material in unique_materials:
            props = self.material_db.get_properties(material)
            if props.density > 2000:  # Solid materials are typically denser than 2000 kg/m³
                solid_mask |= (self.material_types == material)

        return solid_mask

    def _get_planet_radius(self) -> float:
        """Get planet radius in cells"""
        return min(self.width, self.height) * self.planet_radius_fraction / 2
    
    def _get_solar_direction(self) -> tuple[float, float]:
        """
        Get solar direction vector based on solar angle
        
        Returns (dx, dy) unit vector pointing in direction that solar rays travel
        - solar_angle = 0°: rays come from east (1, 0)
        - solar_angle = +90°: rays come from north (0, -1) - NORTHERN hemisphere heating
        - solar_angle = -90°: rays come from south (0, +1) - SOUTHERN hemisphere heating
        """
        solar_angle_radians = np.radians(self.solar_angle)
        solar_dir_x = np.cos(solar_angle_radians)  # horizontal component
        solar_dir_y = -np.sin(solar_angle_radians)  # vertical component (negative = northward, positive = southward)
        return solar_dir_x, solar_dir_y

    def _get_neighbors(self, num_neighbors: int = 8, shuffle: bool = True) -> list:
        """Get neighbor offsets with standardized selection

        This is the unified neighbor selection method used throughout the simulation:
        - Heat diffusion: 8 neighbors with distance weighting for realistic thermal flow
        - Gravitational differentiation: 8 neighbors for comprehensive sorting
        - Gravitational collapse: 8 neighbors for realistic falling
        - Air migration: 8 neighbors for natural buoyancy patterns
        - Cavity filling: 4 neighbors for simple geological settling

        Args:
            num_neighbors: 4 for cardinal directions only, 8 for cardinal + diagonal
            shuffle: Whether to randomize order to avoid grid artifacts (default: True)

        Returns:
            List of (dy, dx) tuples representing relative neighbor offsets
        """
        if num_neighbors == 4:
            # Cardinal directions only
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif num_neighbors == 8:
            # Cardinal + diagonal directions
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            raise ValueError(f"num_neighbors must be 4 or 8, got {num_neighbors}")

        if shuffle:
            np.random.shuffle(neighbors)

        return neighbors

    def _dedupe_swap_pairs(
        self,
        src_y: np.ndarray, src_x: np.ndarray,
        tgt_y: np.ndarray, tgt_x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deduplicate swap pairs so that *no* grid cell appears in more than
        one pair – neither as a source nor as a destination.

        1.  Build flat indices for sources and destinations.
        2.  Identify any cell that appears more than once across the *union*
            of the two arrays.  These represent conflicting moves and are
            removed entirely (first‐come wins to keep logic simple).
        3.  Preserve the original ordering for reproducibility.
        """

        if len(src_y) == 0:
            return src_y, src_x, tgt_y, tgt_x

        src_flat = src_y * self.width + src_x
        tgt_flat = tgt_y * self.width + tgt_x

        # Identify cells that participate multiple times (either as src or tgt)
        combined = np.concatenate([src_flat, tgt_flat])
        unique_cells, counts = np.unique(combined, return_counts=True)
        conflict_cells = unique_cells[counts > 1]

        # Build mask of pairs that are conflict-free
        conflict_src = np.isin(src_flat, conflict_cells)
        conflict_tgt = np.isin(tgt_flat, conflict_cells)
        keep_mask = ~(conflict_src | conflict_tgt)

        if __debug__:
            # After filtering, every cell should appear exactly once overall
            sf = src_flat[keep_mask]; tf = tgt_flat[keep_mask]
            assert len(sf) == len(np.unique(sf))
            assert len(tf) == len(np.unique(tf))
            # Ensure disjoint sets
            assert np.intersect1d(sf, tf).size == 0

        return src_y[keep_mask], src_x[keep_mask], tgt_y[keep_mask], tgt_x[keep_mask]

    def _calculate_temperature_factors(self, threshold: float, scale: float = 200.0, max_factor: float = 10.0) -> np.ndarray:
        """Calculate exponential temperature factors for mobility"""
        temp_excess = np.maximum(0, self.temperature - threshold)
        return np.minimum(max_factor, np.exp(temp_excess / scale))

    def _apply_convective_phase_transitions(self):
        """Handle evaporation and condensation during convective processes"""
        # Water evaporation: hot water at low pressure becomes water vapor
        water_mask = (self.material_types == MaterialType.WATER)
        hot_water_mask = water_mask & (self.temperature > 350)  # 77°C, accounting for pressure

        if np.any(hot_water_mask):
            # Some hot water evaporates
            evap_coords = np.where(hot_water_mask)
            num_evap = len(evap_coords[0])
            # Only evaporate a fraction to avoid sudden changes
            evap_fraction = 0.05  # 5% chance per step
            evap_count = max(1, int(num_evap * evap_fraction))

            if evap_count > 0:
                evap_indices = np.random.choice(num_evap, size=evap_count, replace=False)
                for i in evap_indices:
                    y, x = evap_coords[0][i], evap_coords[1][i]
                    self.material_types[y, x] = MaterialType.WATER_VAPOR

        # Water vapor condensation: cool water vapor becomes water
        vapor_mask = (self.material_types == MaterialType.WATER_VAPOR)
        cool_vapor_mask = vapor_mask & (self.temperature < 320)  # 47°C

        if np.any(cool_vapor_mask):
            # Some cool vapor condenses
            cond_coords = np.where(cool_vapor_mask)
            num_cond = len(cond_coords[0])
            # Only condense a fraction
            cond_fraction = 0.05  # 5% chance per step
            cond_count = max(1, int(num_cond * cond_fraction))

            if cond_count > 0:
                cond_indices = np.random.choice(num_cond, size=cond_count, replace=False)
                for i in cond_indices:
                    y, x = cond_coords[0][i], cond_coords[1][i]
                    self.material_types[y, x] = MaterialType.WATER

        # ------------------------------------------------------------------
        # Deposition: very cold water vapor freezes directly to ICE
        # ------------------------------------------------------------------
        # TODO: This should just be a standard material phase transition, deterministic
        deposit_mask = vapor_mask & (self.temperature < 250)  # < -23 °C
        if np.any(deposit_mask):
            dep_coords = np.where(deposit_mask)
            num_dep = len(dep_coords[0])
            dep_fraction = 0.05
            dep_count = max(1, int(num_dep * dep_fraction))
            dep_indices = np.random.choice(num_dep, size=dep_count, replace=False)
            for i in dep_indices:
                y, x = dep_coords[0][i], dep_coords[1][i]
                self.material_types[y, x] = MaterialType.ICE

    def _get_porosity(self, material_type):
        """Get porosity for a material type"""
        props = self.material_db.get_properties(material_type)
        return props.porosity

    def _apply_weathering(self):
        """Apply surface weathering processes - chemical and physical weathering"""
        changes_made = False

        # Weathering only affects surface and near-surface materials
        distances = self._get_distances_from_center()
        planet_radius = self._get_planet_radius()

        # Define surface zone (within 10% of planet radius from surface)
        surface_distance = planet_radius * 0.9  # 90% of radius = near surface
        surface_mask = (distances >= surface_distance) & (self.material_types != MaterialType.SPACE)

        if not np.any(surface_mask):
            return False

        # Get surface coordinates
        surface_coords = np.where(surface_mask)

        # Temperature in Celsius for weathering calculations
        temp_celsius = self.temperature - 273.15

        # Process each surface cell
        for i in range(len(surface_coords[0])):
            y, x = surface_coords[0][i], surface_coords[1][i]
            material_type = self.material_types[y, x]

            # Skip if not a weatherable material type
            if material_type not in self.material_db.weathering_products:
                continue

            temperature = temp_celsius[y, x]

            # Calculate weathering factors

            # 1. Chemical weathering (enhanced by heat and water)
            chemical_factor = 0.0
            if temperature > 0:  # Above freezing
                # Arrhenius-like temperature dependence (doubles every 10°C)
                chemical_factor = np.exp((temperature - 15) / 14.4)  # Reference: 15°C

            # Water presence enhances chemical weathering
            water_factor = 1.0
            neighbors = self._get_neighbors(8, shuffle=True)  # Shuffled to avoid directional bias
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and
                    self.material_types[ny, nx] == MaterialType.WATER):
                    water_factor = 3.0  # 3x enhancement with water contact
                    break

            chemical_factor *= water_factor

            # 2. Physical weathering (freeze-thaw cycles, thermal expansion)
            physical_factor = 0.0

            # Freeze-thaw weathering (most effective around 0°C)
            if -10 <= temperature <= 10:  # Freeze-thaw zone
                freeze_thaw_intensity = 1.0 - abs(temperature / 10.0)
                physical_factor += freeze_thaw_intensity * 2.0

            # Thermal expansion weathering (extreme temperatures)
            if temperature > 40 or temperature < -20:
                thermal_stress = min(abs(temperature - 20) / 100.0, 1.0)
                physical_factor += thermal_stress

            # 3. Combine weathering factors
            total_weathering = (chemical_factor + physical_factor) * 0.00001  # Very slow process

            # Apply weathering with probability
            if np.random.random() < total_weathering * self.dt:
                # Get weathering products for this material type
                products = self.material_db.weathering_products[material_type]
                new_material = np.random.choice(products)

                self.material_types[y, x] = new_material
                changes_made = True

                # Weathering can create loose material that's easier to transport
                # Slightly reduce temperature due to endothermic weathering reactions
                self.temperature[y, x] = max(self.temperature[y, x] - 2, 2.7)

        return changes_made

    def _calculate_internal_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate internal heat generation source term Q/(ρcp) in K/year"""
        # Initialize source term
        source_term = np.zeros_like(self.temperature)

        # Only apply to non-space solid materials with valid properties (exclude atmospheric gases)
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )
        valid_heating = (
            non_space_mask &
            (self.density > 0) &
            (self.specific_heat > 0) &
            (self.material_types != MaterialType.SPACE) &
            ~atmosphere_mask  # Atmospheric gases don't have radioactive/internal heating
        )

        if not np.any(valid_heating):
            return source_term

        # Get reusable arrays
        distances = self._get_distances_from_center()

        # Heat generation rate based on depth (more heat from radioactive decay in deep materials)
        planet_radius = self._get_planet_radius()
        relative_depth = np.clip(1.0 - distances / planet_radius, 0.0, 1.0)

        # ---------------------------  RADIOGENIC + PRIMORDIAL HEATING  ---------------------------
        # 1. Radiogenic crust/mantle heating (dominates upper ~50 km)
        #    Earth-like average ≈ 1–3 µW/m³.  We taper it towards the centre with an exponential
        #    so that most heat is generated near the surface.

        CRUSTAL_SURFACE_RATE = 3e-6      # 3 µW/m³ at surface-adjacent rock
        crust_decay_length = 0.1         # non-dimensional thickness of radiogenic layer (~10 % of radius)

        crustal_heating_rate = CRUSTAL_SURFACE_RATE * np.exp(-(1.0 - relative_depth) / crust_decay_length)

        # 2. Core/primordial heating (latent heat + gravitational segregation)
        #    Set by "core_heating_depth_scale".  A Gaussian centred at r=0 gives a smooth core profile.

        CORE_CENTRE_RATE = 10e-6          # 10 µW/m³ at the very centre (can be adjusted)
        core_sigma = max(1e-3, self.core_heating_depth_scale)  # avoid div-by-zero

        core_heating_rate = CORE_CENTRE_RATE * np.exp(-(relative_depth / core_sigma) ** 2)

        # 3. Optional visibility boost (set to 1 for physical, >1 for pedagogy)
        internal_boost = getattr(self, "internal_heating_boost", 1.0)

        total_power_density_grid = (crustal_heating_rate + core_heating_rate) * internal_boost  # W/m³

        # Apply only to valid heating cells
        # Q/(ρcp) where Q is W/m³, ρ is kg/m³, cp is J/(kg⋅K)
        # Result is K/s, convert to K/year
        source_term[valid_heating] = (
            total_power_density_grid[valid_heating] /
            (self.density[valid_heating] * self.specific_heat[valid_heating])
        ) * self.seconds_per_year  # Convert K/s → K/year

        # Add to power density tracking for visualization
        self.power_density[valid_heating] += total_power_density_grid[valid_heating]

        # Track total internal heating for debugging
        total_power = np.sum(total_power_density_grid[valid_heating] * (self.cell_size ** 3))  # W
        self.thermal_fluxes['internal_heating'] = total_power

        return source_term

    def _calculate_solar_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate solar heating source term Q/(ρcp) in K/year"""
        source_term = np.zeros_like(self.temperature)

        # Find surface cells that can receive solar radiation
        space_mask = ~non_space_mask

        # Surface cells are those adjacent to space, but exclude atmospheric cells (they get separate atmospheric heating)
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )
        surface_candidates = ndimage.binary_dilation(space_mask, structure=self._circular_kernel_5x5) & non_space_mask & ~atmosphere_mask

        if not np.any(surface_candidates):
            return source_term

        # Calculate solar heating based on angle from solar direction
        center_x, center_y = self.center_of_mass
        solar_dir_x, solar_dir_y = self._get_solar_direction()

        # Create coordinate grids
        y_coords = np.arange(self.height).reshape(-1, 1)  # Shape: (height, 1)
        x_coords = np.arange(self.width).reshape(1, -1)   # Shape: (1, width)
        
        # Calculate position vectors from center
        pos_x = x_coords - center_x
        pos_y = y_coords - center_y
        
        # Calculate angle between position vector and solar direction
        # This gives us the "latitude" relative to the solar direction
        planet_radius_cells = self._get_planet_radius()
        distance_from_center = np.sqrt(pos_x**2 + pos_y**2)
        
        # Avoid division by zero at center
        safe_distance = np.maximum(distance_from_center, 0.1)
        
        # Dot product with solar direction gives cos(angle) 
        dot_product = (pos_x * solar_dir_x + pos_y * solar_dir_y) / safe_distance
        
        # Solar intensity follows cosine law: I = I₀ * max(0, cos(angle))
        # Negative values mean "night side" of planet
        solar_intensity_factor = np.maximum(0.0, dot_product)

        # Broadcast to full grid size
        solar_intensity_factor = np.broadcast_to(solar_intensity_factor, (self.height, self.width))

        # Calculate albedo from material properties - vectorized
        albedo = np.zeros((self.height, self.width))

        # Create albedo lookup table for all material types
        if not hasattr(self, '_albedo_lookup'):
            self._albedo_lookup = {}
            for material_type in MaterialType:
                self._albedo_lookup[material_type] = self.material_db.get_properties(material_type).albedo

        # Apply albedo for all non-space cells vectorized
        non_space_coords = np.where(non_space_mask)
        if len(non_space_coords[0]) > 0:
            # Get material types for all non-space cells
            materials = self.material_types[non_space_coords]

            # Vectorized albedo lookup
            albedo_values = np.array([self._albedo_lookup[mat] for mat in materials.flat])

            # Handle frozen water special case - vectorized
            water_mask = (materials == MaterialType.WATER)
            frozen_mask = water_mask & (self.temperature[non_space_coords] < 273.15)
            if np.any(frozen_mask):
                ice_albedo = self._albedo_lookup[MaterialType.ICE]
                albedo_values[frozen_mask] = ice_albedo

            # Apply albedo values to grid
            albedo[non_space_coords] = albedo_values

        # Calculate weighted average albedo for the planet
        surface_weights = surface_candidates.astype(float) * solar_intensity_factor
        if np.sum(surface_weights) > 0:
            planet_albedo = np.average(albedo[surface_candidates], weights=surface_weights[surface_candidates])
        else:
            planet_albedo = 0.2  # Default

        # Reduced solar input due to albedo reflection
        effective_solar_constant = self.solar_constant * (1.0 - planet_albedo)

        # Apply atmospheric absorption using selected method
        remaining_solar_flux = self._solve_atmospheric_absorption(non_space_mask, solar_intensity_factor, effective_solar_constant, source_term)

        # Apply solar heating to surface cells
        surface_coords = np.where(surface_candidates)
        if len(surface_coords[0]) == 0:
            return source_term

        # Get solar flux for surface cells
        surface_y, surface_x = surface_coords
        surface_flux = remaining_solar_flux[surface_y, surface_x]  # W/m² (already includes latitude effects)

        # Convert to source term Q/(ρcp) in K/year
        valid_heating = (self.density[surface_coords] > 0) & (self.specific_heat[surface_coords] > 0)

        if np.any(valid_heating):
            # Convert surface flux (W/m²) to volumetric power density (W/m³)
            # Solar heating is a surface phenomenon, spread over surface layer
            surface_layer_thickness = self.cell_size * self.surface_radiation_depth_fraction
            volumetric_power_density = surface_flux[valid_heating] / surface_layer_thickness  # W/m³

            # Convert to temperature source term: Q/(ρcp) in K/s, then to K/year
            heating_y, heating_x = surface_y[valid_heating], surface_x[valid_heating]
            heating_source = (
                volumetric_power_density /
                (self.density[heating_y, heating_x] * self.specific_heat[heating_y, heating_x])
            ) * self.seconds_per_year  # Convert K/s → K/year

            source_term[heating_y, heating_x] += heating_source

            # Track solar power density for visualization
            self.power_density[heating_y, heating_x] += volumetric_power_density

            # Track total solar input for debugging
            total_solar_power = np.sum(volumetric_power_density * (self.cell_size ** 3))  # W
            self.thermal_fluxes['solar_input'] = total_solar_power

        return source_term

    def _solve_atmospheric_absorption(self, non_space_mask: np.ndarray, solar_intensity_factor: np.ndarray,
                                    effective_solar_constant: float, source_term: np.ndarray) -> np.ndarray:
        """Master atmospheric absorption solver - dispatches to selected implementation method"""
        
        if self.atmospheric_absorption_method == "directional_sweep":
            return self._atmospheric_absorption_directional_sweep(non_space_mask, solar_intensity_factor, effective_solar_constant, source_term)
        else:
            raise ValueError(f"Unknown atmospheric absorption method: {self.atmospheric_absorption_method}. "
                           f"Available options: 'directional_sweep'")

    def _atmospheric_absorption_directional_sweep(self, non_space_mask: np.ndarray, solar_intensity_factor: np.ndarray,
                                                   effective_solar_constant: float, source_term: np.ndarray) -> np.ndarray:
        """Directional sweep (DDA) atmospheric absorption working for *any* solar angle."""

        initial_flux = effective_solar_constant * self.planetary_distance_factor * solar_intensity_factor
        remaining_flux = np.zeros_like(initial_flux)

        ux, uy = self._get_solar_direction()
        if ux == 0 and uy == 0:
            return initial_flux

        # DDA stepping direction: move OPPOSITE to incoming solar vector so that we march from the day-side boundary into the planet
        step_x = -1 if ux > 0 else 1
        step_y = -1 if uy > 0 else 1
        inv_dx = abs(1.0 / ux) if ux != 0 else float('inf')
        inv_dy = abs(1.0 / uy) if uy != 0 else float('inf')

        # Select all entry cells on the day-side boundary (once per frame)
        if abs(ux) >= abs(uy):  # shallow ray → enter from side opposite to ray direction
            entry_x = self.width - 1 if ux > 0 else 0  # rays travel towards −x when ux>0
            entry_cells = ((entry_x, y) for y in range(self.height))
        else:  # steep ray → enter from top/bottom opposite to ray direction
            entry_y = self.height - 1 if uy > 0 else 0
            entry_cells = ((x, entry_y) for x in range(self.width))

        for sx, sy in entry_cells: # main DDA march
            I = initial_flux[sy, sx]
            t_max_x = inv_dx
            t_max_y = inv_dy

            while 0 <= sx < self.width and 0 <= sy < self.height and I > 0:
                mat = self.material_types[sy, sx]

                if mat != MaterialType.SPACE:
                    k = self.material_db.get_solar_absorption(mat)
                    absorbed = I * k

                    if absorbed > 0 and self.density[sy, sx] > 0 and self.specific_heat[sy, sx] > 0:
                        vol_power = absorbed / self.cell_size
                        source_term[sy, sx] += (vol_power / (self.density[sy, sx] * self.specific_heat[sy, sx])) * self.seconds_per_year
                        self.power_density[sy, sx] += vol_power

                        # Categorise energy deposition for diagnostics
                        if mat in (MaterialType.AIR, MaterialType.WATER_VAPOR):
                            self.thermal_fluxes['atmospheric_heating'] += vol_power * (self.cell_size ** 3)
                        else:
                            self.thermal_fluxes['solar_input'] += vol_power * (self.cell_size ** 3)

                    I -= absorbed

                    if k >= 1.0 or I <= 0:
                        remaining_flux[sy, sx] += I
                        break  # ray terminated at opaque surface or fully absorbed

                # advance to next grid cell using DDA
                if t_max_x < t_max_y:
                    sx += step_x
                    t_max_x += inv_dx
                else:
                    sy += step_y
                    t_max_y += inv_dy

        return remaining_flux

    def _calculate_atmospheric_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate atmospheric heating source term Q/(ρcp) in K/year"""
        # Atmospheric heating is now handled directly in solar heating source calculation
        # This method remains as a placeholder for future atmospheric-specific heating
        return np.zeros_like(self.temperature)

    def _solve_radiative_cooling(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Master radiative cooling solver - dispatches to selected implementation method"""

        if self.radiative_cooling_method == "linearized_stefan_boltzmann":
            return self._solve_radiative_cooling_linearized_stefan_boltzmann(temperature, non_space_mask)
        elif self.radiative_cooling_method == "newton_raphson_implicit":
            return self._solve_radiative_cooling_newton_raphson_implicit(temperature, non_space_mask)
        else:
            raise ValueError(f"Unknown radiative cooling method: {self.radiative_cooling_method}. "
                           f"Available options: 'linearized_stefan_boltzmann', 'newton_raphson_implicit'")

    def _solve_radiative_cooling_linearized_stefan_boltzmann(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """
        Solve radiative cooling using linearized Stefan-Boltzmann approximation

        Method: Uses Newton cooling law Q = h(T - T_space) where h ≈ 4σεT₀³
        Advantages: Explicit, very stable, fast
        Disadvantages: Approximate, less accurate for large temperature differences
        """
        working_temp = temperature.copy()
        space_mask = ~non_space_mask

        # Find atmospheric materials
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )

        # Find all atmosphere connected to space using connected components
        labeled_atmo, num_features = ndimage.label(atmosphere_mask, structure=self._circular_kernel_3x3)

        if num_features == 0:
            outer_atmo_mask = np.zeros_like(atmosphere_mask, dtype=bool)
        else:
            # Find atmosphere regions adjacent to space (using circular kernel)
            space_neighbors = ndimage.binary_dilation(space_mask, structure=self._circular_kernel_5x5)
            space_adjacent_atmo = space_neighbors & atmosphere_mask

            if np.any(space_adjacent_atmo):
                # Get the labels of atmosphere regions connected to space
                space_connected_labels = np.unique(labeled_atmo[space_adjacent_atmo])
                space_connected_labels = space_connected_labels[space_connected_labels > 0]  # Remove 0 (background)

                # Outer atmosphere consists of all regions connected to space
                outer_atmo_mask = np.isin(labeled_atmo, space_connected_labels)
            else:
                outer_atmo_mask = np.zeros_like(atmosphere_mask, dtype=bool)

        # Find the solid surface beneath the outer atmosphere
        # This is solid material adjacent to outer atmosphere or space
        solid_mask = ~(space_mask | atmosphere_mask)  # All non-space, non-atmosphere

        # Surface solids are adjacent to outer atmosphere or space (using circular kernel)
        surface_candidates = ndimage.binary_dilation(outer_atmo_mask | space_mask, structure=self._circular_kernel_5x5)
        surface_solid_mask = surface_candidates & solid_mask

        # Combine outer atmosphere and surface solids for radiation
        radiative_mask = outer_atmo_mask | surface_solid_mask

        if not np.any(radiative_mask):
            return working_temp

        # Get coordinates and properties of radiating cells
        radiative_coords = np.where(radiative_mask)
        T_radiating = working_temp[radiative_coords]
        T_space = self.space_temperature

        # Only process cells that are actually cooling (T > T_space)
        cooling_mask = T_radiating > T_space
        if not np.any(cooling_mask):
            return working_temp

        cooling_idx = np.where(cooling_mask)[0]
        T_cooling = T_radiating[cooling_idx]

        # Get material properties
        material_types_cooling = self.material_types[radiative_coords][cooling_idx]
        emissivity = np.array([self.material_db.get_properties(mat_type).emissivity for mat_type in material_types_cooling])
        density_cooling = self.density[radiative_coords][cooling_idx]
        specific_heat_cooling = self.specific_heat[radiative_coords][cooling_idx]

        # Calculate dynamic greenhouse effect based on water vapor content
        water_vapor_mask = (self.material_types == MaterialType.WATER_VAPOR)
        total_water_vapor_mass = np.sum(self.density[water_vapor_mask]) if np.any(water_vapor_mask) else 0.0

        # Scale greenhouse effect by water vapor content (logarithmic to prevent runaway)
        if total_water_vapor_mass > 0:
            vapor_factor = np.log1p(total_water_vapor_mass / self.greenhouse_vapor_scaling) / 10.0
            greenhouse_factor = self.base_greenhouse_effect + (self.max_greenhouse_effect - self.base_greenhouse_effect) * np.tanh(vapor_factor)
        else:
            greenhouse_factor = self.base_greenhouse_effect

        # Linearized Stefan-Boltzmann: Q = h(T - T_space) where h = 4σεT₀³
        # This is the Newton cooling law approximation
        T_reference = 300.0  # Reference temperature for linearization (K)
        stefan_geological = self.stefan_boltzmann_geological / 1000.0  # Conservative scaling
        effective_stefan = stefan_geological * (1.0 - greenhouse_factor)

        # Linearized heat transfer coefficient (stable explicit form)
        h_effective = 4.0 * effective_stefan * emissivity * (T_reference ** 3)

        # Apply linearized cooling: P = h(T - T_space) per unit area
        temp_difference = T_cooling - T_space
        power_per_area = h_effective * self.radiative_cooling_efficiency * temp_difference  # J/(year⋅m²)

        # Convert surface power to volumetric power density and then to temperature change
        surface_layer_thickness = self.cell_size * self.surface_radiation_depth_fraction
        volumetric_power_density = power_per_area / (surface_layer_thickness * self.seconds_per_year)  # W/m³

        # Convert to temperature change: ΔT = -Q*dt/(ρcp) (negative for cooling)
        dt_seconds = self.dt * self.seconds_per_year
        temp_change = -(volumetric_power_density * dt_seconds) / (density_cooling * specific_heat_cooling)  # K

        # Apply temperature change
        cooling_y, cooling_x = radiative_coords[0][cooling_idx], radiative_coords[1][cooling_idx]
        working_temp[cooling_y, cooling_x] += temp_change

        # Ensure temperatures don't go below space temperature
        working_temp[cooling_y, cooling_x] = np.maximum(working_temp[cooling_y, cooling_x], T_space)

        # Track cooling power density for visualization (negative = heat loss)
        self.power_density[cooling_y, cooling_x] -= volumetric_power_density

        # Track total radiative output for debugging
        total_radiative_power = np.sum(volumetric_power_density * (self.cell_size ** 3))  # W (positive magnitude)
        self.thermal_fluxes['radiative_output'] = total_radiative_power

        return working_temp

    def _solve_non_radiative_sources(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Solve non-radiative heat sources (internal, solar, atmospheric) explicitly"""
        working_temp = temperature.copy()

        # Calculate all non-radiative heat sources
        internal_source = self._calculate_internal_heating_source(non_space_mask)
        solar_source = self._calculate_solar_heating_source(non_space_mask)
        atmospheric_source = self._calculate_atmospheric_heating_source(non_space_mask)

        # Combine all sources
        total_source = internal_source + solar_source + atmospheric_source

        # Apply sources explicitly (these are typically well-behaved)
        dt_years = self.dt
        source_change = total_source * dt_years

        # Apply to non-space cells only
        valid_sources = non_space_mask & np.isfinite(source_change)
        if np.any(valid_sources):
            working_temp[valid_sources] += source_change[valid_sources]

        return working_temp

    def _save_state(self):
        """Save current state for time reversal"""
        if len(self.history) >= self.max_history:
            self.history.pop(0)

        state = {
            'material_types': self.material_types.copy(),
            'temperature': self.temperature.copy(),
            'pressure': self.pressure.copy(),
            'pressure_offset': self.pressure_offset.copy(),
            'age': self.age.copy(),
            'time': self.time,
            'power_density': self.power_density.copy()
        }
        self.history.append(state)

    def _record_time_series_data(self):
        """Record time-series data for graphing"""
        # Calculate stats
        non_space_mask = (self.material_types != MaterialType.SPACE)
        # Temperature stats
        temps = self.temperature[non_space_mask]
        avg_temp = np.mean(temps) if len(temps) > 0 else 0.0
        max_temp = np.max(temps) if len(temps) > 0 else 0.0

        # Total energy (approximate as temperature * mass * specific_heat)
        cell_volume = self.cell_size ** 3
        masses = self.density[non_space_mask] * cell_volume
        specific_heats = self.specific_heat[non_space_mask]
        valid_energy = (masses > 0) & (specific_heats > 0)

        if np.any(valid_energy):
            energies = masses[valid_energy] * specific_heats[valid_energy] * temps[valid_energy]
            total_energy = np.sum(energies)
        else:
            total_energy = 0.0

        # Net power (sum of all power densities)
        net_power = np.sum(self.power_density)

        # Calculate greenhouse factor (for monitoring runaway greenhouse)
        water_vapor_mask = (self.material_types == MaterialType.WATER_VAPOR)
        total_water_vapor_mass = np.sum(self.density[water_vapor_mask]) if np.any(water_vapor_mask) else 0.0

        if total_water_vapor_mass > 0:
            vapor_factor = np.log1p(total_water_vapor_mass / self.greenhouse_vapor_scaling) / 10.0
            greenhouse_factor = self.base_greenhouse_effect + (self.max_greenhouse_effect - self.base_greenhouse_effect) * np.tanh(vapor_factor)
        else:
            greenhouse_factor = self.base_greenhouse_effect

        # Calculate planet albedo (approximate from surface conditions)
        ice_fraction = np.sum((self.temperature < 273.15) & non_space_mask) / np.sum(non_space_mask) if np.any(non_space_mask) else 0.0
        vapor_fraction = np.sum(water_vapor_mask) / np.sum(non_space_mask) if np.any(non_space_mask) else 0.0
        planet_albedo = 0.2 + ice_fraction * 0.6 + vapor_fraction * 0.5  # Rough estimate

        # Record data
        self.time_series['time'].append(self.time)
        self.time_series['avg_temperature'].append(avg_temp - 273.15)  # Convert to Celsius
        self.time_series['max_temperature'].append(max_temp - 273.15)  # Convert to Celsius
        self.time_series['total_energy'].append(total_energy)
        self.time_series['net_power'].append(net_power)
        self.time_series['greenhouse_factor'].append(greenhouse_factor)
        self.time_series['planet_albedo'].append(planet_albedo)

        # Trim data if too long
        for key in self.time_series:
            if len(self.time_series[key]) > self.max_time_series:
                self.time_series[key] = self.time_series[key][-self.max_time_series:]

    def step_forward(self, dt: Optional[float] = None):
        """Advance simulation by one time step - optimized version"""
        if dt is not None:
            self.dt = dt

        # Save state for potential reversal BEFORE modifying anything
        self._save_state()

        # Reset thermal flux tracking and power density for this timestep
        self.thermal_fluxes = {
            'solar_input': 0.0,
            'radiative_output': 0.0,
            'internal_heating': 0.0,
            'atmospheric_heating': 0.0,
            'net_flux': 0.0
        }
        self.power_density.fill(0.0)  # Reset for current timestep - all sources will add to this

        # Performance instrumentation – measure wall-clock time spent in each
        # major sub-routine.  Timing is collected only for the current frame
        # and printed if ``self.logging_enabled`` is True (toggled with the
        # visualiser shortcut 'L').

        step_start_total = time.perf_counter()
        self._perf_times = {}
        _last_cp = step_start_total

        # Core physics (every step)
        self.temperature, stability_factor = self._heat_diffusion()
        self._perf_times['heat_diffusion'] = time.perf_counter() - _last_cp
        _last_cp = time.perf_counter()

        # Apply stability factor to the time step for this step
        effective_dt = self.dt * stability_factor
        self._last_stability_factor = stability_factor

        # Update center of mass and pressure (every step - needed for thermal calculations)
        self._calculate_center_of_mass()
        self._calculate_planetary_pressure()
        self._perf_times['mass_pressure'] = time.perf_counter() - _last_cp
        _last_cp = time.perf_counter()

        # Apply metamorphic processes (every step - fundamental)
        metamorphic_changes = self._apply_metamorphism()
        self._perf_times['metamorphism'] = time.perf_counter() - _last_cp
        _last_cp = time.perf_counter()

        # Run geological processes based on performance configuration
        step_count = int(self.time / self.dt)

        # Unified density stratification (physics-correct vectorized for speed + realism)
        density_stratification_changes = False
        if step_count % self.step_interval_differentiation == 0:
            density_stratification_changes = self._apply_density_stratification_local_vectorized()
        self._perf_times['density_strat'] = time.perf_counter() - _last_cp
        _last_cp = time.perf_counter()

        # Gravitational collapse (vectorized for maximum speed)
        collapse_changes = False
        if step_count % self.step_interval_collapse == 0:
            collapse_changes = self._apply_gravitational_collapse_vectorized()
        self._perf_times['collapse'] = time.perf_counter() - _last_cp
        _last_cp = time.perf_counter()

        # Air migration (vectorized for maximum speed)
        fluid_changes = False
        if step_count % self.step_interval_fluid == 0:
            fluid_changes = self._apply_fluid_dynamics_vectorized()
        self._perf_times['fluid_dyn'] = time.perf_counter() - _last_cp
        _last_cp = time.perf_counter()

        # Weathering (configurable)
        weathering_changes = False
        if self.enable_weathering and step_count % 10 == 0:  # Every 10th step when enabled
            weathering_changes = self._apply_weathering()
        self._perf_times['weathering'] = time.perf_counter() - _last_cp
        _last_cp = time.perf_counter()

        # Flag when processes that can create voids have moved material
        if collapse_changes or fluid_changes or density_stratification_changes:
            self._need_settle = True

        settle_changes = False
        # Attempt settle only if flagged and on interval
        if self._need_settle and step_count % self.settle_interval == 0:
            settle_changes = self._settle_unsupported_chunks()
            # If nothing moved, unset flag until new voids appear
            self._need_settle = settle_changes
        self._perf_times['settle'] = time.perf_counter() - _last_cp
        _last_cp = time.perf_counter()

        # Update material properties if material types changed
        if (metamorphic_changes or density_stratification_changes or collapse_changes or
            fluid_changes or weathering_changes or settle_changes):
            self._update_material_properties()

        # Update age and time with the effective time step
        self.age += effective_dt
        self.time += effective_dt

        # Record time-series data for graphs
        self._record_time_series_data()

        # Final safety check: ensure SPACE cells stay as SPACE and at cosmic background temp
        space_mask = (self.material_types == MaterialType.SPACE)
        self.temperature[space_mask] = self.space_temperature  # Kelvin
        self.pressure[space_mask] = 0.0

        # Store adaptive time step info for analysis
        if hasattr(self, '_last_stability_factor'):
            self._adaptive_time_steps.append(self._last_stability_factor)
            # Keep only recent history
            if len(self._adaptive_time_steps) > 100:
                self._adaptive_time_steps = self._adaptive_time_steps[-100:]

        # Calculate net thermal flux and log debugging information
        self.thermal_fluxes['net_flux'] = (
            self.thermal_fluxes['solar_input'] +
            self.thermal_fluxes['atmospheric_heating'] +
            self.thermal_fluxes['internal_heating'] -
            self.thermal_fluxes['radiative_output']
        )

        # Debug logging for thermal balance
        if self.logger.isEnabledFor(logging.DEBUG):
            # ------------------------------------------------------------------
            # Diagnostic header – use fractional step index and higher-precision time
            # ------------------------------------------------------------------
            step_index = self.time / self.dt if self.dt > 0 else 0.0  # may be fractional due to sub-stepping

            non_space_mask = (self.material_types != MaterialType.SPACE)
            temp_celsius = self.temperature - 273.15
            avg_planet_temp = np.mean(temp_celsius[non_space_mask]) if np.any(non_space_mask) else 0.0
            min_temp = np.min(temp_celsius[non_space_mask]) if np.any(non_space_mask) else 0.0
            max_temp = np.max(temp_celsius[non_space_mask]) if np.any(non_space_mask) else 0.0

            # Clean, formatted header with step number and key info
            self.logger.debug("=" * 65)
            self.logger.debug(f"STEP {step_index:6.2f} | Time: {self.time:7.2f}y | Planet Avg: {avg_planet_temp:7.1f}°C")
            self.logger.debug("=" * 65)

            # Temperature statistics
            self.logger.debug(f"Temperature Range:  Min: {min_temp:7.1f}°C  |  Max: {max_temp:7.1f}°C")

            # Thermal flux balance (clean formatting)
            total_input = (self.thermal_fluxes['solar_input'] +
                          self.thermal_fluxes['atmospheric_heating'] +
                          self.thermal_fluxes['internal_heating'])

            self.logger.debug("THERMAL FLUX BALANCE:")
            self.logger.debug(f"  Solar Input:      {self.thermal_fluxes['solar_input']:10.3e} W")
            self.logger.debug(f"  Atmospheric Heat: {self.thermal_fluxes['atmospheric_heating']:10.3e} W")
            self.logger.debug(f"  Internal Heat:    {self.thermal_fluxes['internal_heating']:10.3e} W")
            self.logger.debug(f"  Total Input:      {total_input:10.3e} W")
            self.logger.debug(f"  Radiative Output: {self.thermal_fluxes['radiative_output']:10.3e} W")
            self.logger.debug(f"  NET FLUX:         {self.thermal_fluxes['net_flux']:10.3e} W")

            # Detailed material breakdowns
            air_mask = (self.material_types == MaterialType.AIR)
            water_vapor_mask = (self.material_types == MaterialType.WATER_VAPOR)
            water_mask = (self.material_types == MaterialType.WATER)
            ice_mask = (self.material_types == MaterialType.ICE)
            atmosphere_mask = air_mask | water_vapor_mask            # gases only
            water_total_mask = water_mask | ice_mask                 # condensed phases only
            # Remaining cells are counted as "rock" (includes magma & solid crust)
            rock_mask = non_space_mask & ~atmosphere_mask & ~water_total_mask

            self.logger.debug("MATERIAL BREAKDOWN:")

            # Atmosphere breakdown (air + water vapor)
            if np.any(atmosphere_mask):
                atmo_temp_avg = np.mean(temp_celsius[atmosphere_mask])
                atmo_temp_max = np.max(temp_celsius[atmosphere_mask])
                atmo_count = np.sum(atmosphere_mask)
                self.logger.debug(f"  Atmosphere ({atmo_count:4d} cells): Avg: {atmo_temp_avg:7.1f}°C  Max: {atmo_temp_max:7.1f}°C")

                if np.any(air_mask):
                    air_temp_avg = np.mean(temp_celsius[air_mask])
                    air_count = np.sum(air_mask)
                    self.logger.debug(f"    - Air      ({air_count:4d} cells): Avg: {air_temp_avg:7.1f}°C")

                if np.any(water_vapor_mask):
                    vapor_temp_avg = np.mean(temp_celsius[water_vapor_mask])
                    vapor_count = np.sum(water_vapor_mask)
                    self.logger.debug(f"    - W.Vapor  ({vapor_count:4d} cells): Avg: {vapor_temp_avg:7.1f}°C")

            # Water distribution
            total_water_cells = np.sum(water_total_mask)
            if total_water_cells > 0:
                water_temp_avg = np.mean(temp_celsius[water_total_mask])
                water_temp_max = np.max(temp_celsius[water_total_mask])
                self.logger.debug(f"  Water Total ({total_water_cells:4d} cells): Avg: {water_temp_avg:7.1f}°C  Max: {water_temp_max:7.1f}°C")

                if np.any(water_mask):
                    liquid_count = np.sum(water_mask)
                    liquid_temp_avg = np.mean(temp_celsius[water_mask])
                    self.logger.debug(f"    - Liquid   ({liquid_count:4d} cells): Avg: {liquid_temp_avg:7.1f}°C")

                if np.any(ice_mask):
                    ice_count = np.sum(ice_mask)
                    ice_temp_avg = np.mean(temp_celsius[ice_mask])
                    self.logger.debug(f"    - Ice      ({ice_count:4d} cells): Avg: {ice_temp_avg:7.1f}°C")

            # Rock breakdown (everything else)
            if np.any(rock_mask):
                rock_temp_avg = np.mean(temp_celsius[rock_mask])
                rock_temp_max = np.max(temp_celsius[rock_mask])
                rock_temp_min = np.min(temp_celsius[rock_mask])
                rock_count = np.sum(rock_mask)
                self.logger.debug(f"  Rock       ({rock_count:4d} cells): Avg: {rock_temp_avg:7.1f}°C  Min: {rock_temp_min:7.1f}°C  Max: {rock_temp_max:7.1f}°C")

            # Thermal diffusivity diagnostics
            if hasattr(self, '_max_thermal_diffusivity'):
                self.logger.debug(f"Max Thermal Diff:   {self._max_thermal_diffusivity:10.3e} m²/s")

            self.logger.debug("=" * 65)

            if self.logger.isEnabledFor(logging.DEBUG):
                total_cells = np.sum(self.material_types != MaterialType.SPACE)
                self.logger.debug(f"  Non-SPACE cells: {total_cells}")

        # # ------------------------------------------------------------------
        # # SAFETY CLAMP – keep temperatures within reasonable physical bounds
        # # ------------------------------------------------------------------
        # MAX_TEMP_K = 5000.0  # ~4727 °C, hotter than most realistic magma
        # MIN_TEMP_K = self.space_temperature  # ~3 K
        # np.clip(self.temperature, MIN_TEMP_K, MAX_TEMP_K, out=self.temperature)

        # Update planet center of mass (for gravity direction) periodically
        if step_count % 10 == 0:
            self._calculate_center_of_mass()

        # Performance – record total and optionally log
        self._perf_times['total'] = time.perf_counter() - step_start_total

        if self.logging_enabled:
            self.logger.info("Performance timing (ms):")
            for name, seconds in self._perf_times.items():
                self.logger.info(f"  {name:<15}: {seconds*1000:.1f}")

    def step_backward(self):
        """Reverse simulation by one time step"""
        if len(self.history) > 0:
            state = self.history.pop()
            self.material_types = state['material_types']
            self.temperature = state['temperature']
            self.pressure = state['pressure']
            self.pressure_offset = state['pressure_offset']
            self.age = state['age']
            self.time = state['time']
            self.power_density = state['power_density']

            # Clear adaptive time step history when stepping backward
            self._adaptive_time_steps = []

            # Mark properties as dirty and update
            self._properties_dirty = True
            self._update_material_properties()
            return True
        return False

    def _create_gaussian_intensity_field(self, center_x: float, center_y: float, radius: float,
                                       effective_radius_multiplier: float = 2.0) -> np.ndarray:
        """Create a Gaussian intensity field centered at given point"""
        effective_radius = radius * effective_radius_multiplier
        distances = self._get_distances_from_center(center_x, center_y)

        # Create soft rolloff mask (Gaussian-like falloff)
        falloff_mask = distances <= effective_radius
        normalized_distance = np.where(falloff_mask, distances / effective_radius, 1.0)

        # Smooth falloff function (1.0 at center, 0.0 at edge)
        return np.where(falloff_mask,
                       np.exp(-2.0 * normalized_distance**2),  # Gaussian falloff
                       0.0)

    def add_heat_source(self, x: int, y: int, radius: int, temperature: float):
        """Add a localized heat source with soft rolloff - more effective and realistic"""
        # Create intensity field with soft falloff
        intensity = self._create_gaussian_intensity_field(x, y, radius)

        # Apply heat using the specified temperature increase
        temp_addition = intensity * temperature

        # Apply the temperature increase
        self.temperature = np.maximum(self.temperature, self.temperature + temp_addition)

    def apply_tectonic_stress(self, x: int, y: int, radius: int, pressure_increase: float):
        """Apply tectonic stress to increase pressure locally (persistent)"""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        # Add to pressure offset so it persists across recalculations
                        self.pressure_offset[ny, nx] += pressure_increase

        # Recalculate pressure to include the new offset
        self._calculate_planetary_pressure()

    def get_visualization_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get data for visualization (material colors, temperature, pressure, power)"""
        # Create color array from material types
        colors = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Get unique material types and their colors
        # Convert to set to avoid sorting issues with MaterialType enums
        unique_materials = set(self.material_types.flatten())

        # Efficient color assignment
        for material in unique_materials:
            mask = (self.material_types == material)
            if np.any(mask):
                props = self.material_db.get_properties(material)
                colors[mask] = props.color_rgb

        return colors, self.temperature, self.pressure, self.power_density

    def get_stats(self) -> dict:
        """Get simulation statistics"""
        # Convert to string values for unique counting
        material_strings = np.array([material.value for material in self.material_types.flatten()])
        unique_materials, counts = np.unique(material_strings, return_counts=True)
        material_percentages = {material: count/len(self.material_types.flatten())*100
                               for material, count in zip(unique_materials, counts)}

        # Sort by percentage in descending order
        sorted_materials = dict(sorted(material_percentages.items(), key=lambda x: x[1], reverse=True))

        stats = {
            'time': self.time,
            'dt': self.dt,
            'effective_dt': getattr(self, '_actual_effective_dt', self.dt),  # Actual dt used per substep
            'stability_factor': getattr(self, '_last_stability_factor', 1.0),  # Actual factor used
            'substeps': getattr(self, '_actual_substeps', 1),  # Number of substeps actually used
            'max_thermal_diffusivity': getattr(self, '_max_thermal_diffusivity', 0.0),
            'avg_temperature': np.mean(self.temperature) - 273.15,  # Convert to Celsius for display
            'max_temperature': np.max(self.temperature) - 273.15,   # Convert to Celsius for display
            'avg_pressure': np.mean(self.pressure),
            'max_pressure': np.max(self.pressure),
            'material_composition': sorted_materials,
            'history_length': len(self.history)
        }

        # Add differentiation stats if available
        if hasattr(self, '_last_differentiation_stats'):
            stats.update(self._last_differentiation_stats)

        return stats

    def _apply_gravitational_collapse_vectorized(self):
        """Fast vectorized gravitational collapse using morphological operations"""
        if not hasattr(self, '_collapse_kernel'):
            # Pre-compute kernels for different neighbor configurations
            self._collapse_kernel_4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
            self._collapse_kernel_8 = np.ones((3, 3), dtype=bool)
            self._collapse_kernel_8[1, 1] = False

        changes_made = False

        # Get distance array for gravitational direction (vectorized)
        center_x, center_y = self.center_of_mass
        distances = self._get_distances_from_center(center_x, center_y)

        # Create solid and cavity masks
        solid_mask = self._get_solid_mask()
        # A cavity is any cell that is not solid (AIR, WATER, or SPACE). Including
        # SPACE ensures surface rocks adjacent to vacuum are free to detach and
        # fall inward when a deeper cell is available.
        cavity_mask = ~solid_mask

        if not np.any(solid_mask) or not np.any(cavity_mask):
            return False

        # Vectorized fall steps
        kernel = self._collapse_kernel_4 if self.neighbor_count == 4 else self._collapse_kernel_8

        for fall_step in range(self.max_fall_steps):
            step_changes = False

            # Find solid materials adjacent to cavities (fast morphological operation)
            solid_near_cavities = solid_mask & ndimage.binary_dilation(cavity_mask, structure=kernel)

            if not np.any(solid_near_cavities):
                break

            # For each solid cell near cavities, find the closest cavity (vectorized)
            solid_coords = np.where(solid_near_cavities)
            if len(solid_coords[0]) == 0:
                break

            # Sample based on performance settings
            total_solid = len(solid_coords[0])
            sample_size = max(1, int(total_solid * self.process_fraction_solid))
            if sample_size < total_solid:
                indices = np.random.choice(total_solid, size=sample_size, replace=False)
                solid_coords = (solid_coords[0][indices], solid_coords[1][indices])

            solid_distances = distances[solid_coords]

            # Find valid collapse targets using vectorized operations
            moves_made = 0
            max_moves_per_step = min(len(solid_coords[0]), 100)  # Limit for stability

            # Build an isotropic list of neighbor offsets (radius ≤2) for collapse moves
            if self.neighbor_count == 8:
                offsets = []
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:
                            continue
                        if self._circular_kernel_5x5[dy + 2, dx + 2]:
                            offsets.append((dy, dx))
                np.random.shuffle(offsets)
                neighbor_offsets_collapse = offsets
            else:
                neighbor_offsets_collapse = [(-1,0),(1,0),(0,-1),(0,1)]

            # ---------- NEW isotropic candidate gathering ----------
            # Collect all potential swaps first, then randomly choose up to max_moves_per_step
            cand_solid_y = []
            cand_solid_x = []
            cand_neighbor_y = []
            cand_neighbor_x = []

            for dy, dx in neighbor_offsets_collapse:
                # Neighbor positions for all solid cells
                neighbor_y = solid_coords[0] + dy
                neighbor_x = solid_coords[1] + dx

                # Bounds
                in_bounds = (
                    (neighbor_y >= 0) & (neighbor_y < self.height) &
                    (neighbor_x >= 0) & (neighbor_x < self.width)
                )
                if not np.any(in_bounds):
                    continue

                valid_idx = np.where(in_bounds)[0]
                v_solid_y = solid_coords[0][valid_idx]
                v_solid_x = solid_coords[1][valid_idx]
                v_neighbor_y = neighbor_y[valid_idx]
                v_neighbor_x = neighbor_x[valid_idx]

                neighbor_distances = distances[v_neighbor_y, v_neighbor_x]
                solid_distances_valid = solid_distances[valid_idx]

                is_non_solid = cavity_mask[v_neighbor_y, v_neighbor_x]
                is_not_space = self.material_types[v_neighbor_y, v_neighbor_x] != MaterialType.SPACE
                is_closer = neighbor_distances < solid_distances_valid
                can_collapse = is_non_solid & is_not_space & is_closer

                if np.any(can_collapse):
                    idx = np.where(can_collapse)[0]
                    cand_solid_y.append(v_solid_y[idx])
                    cand_solid_x.append(v_solid_x[idx])
                    cand_neighbor_y.append(v_neighbor_y[idx])
                    cand_neighbor_x.append(v_neighbor_x[idx])

            if cand_solid_y:
                # Concatenate lists
                cs_y = np.concatenate(cand_solid_y)
                cs_x = np.concatenate(cand_solid_x)
                cn_y = np.concatenate(cand_neighbor_y)
                cn_x = np.concatenate(cand_neighbor_x)

                total_candidates = len(cs_y)
                if total_candidates > 0:
                    # Apply fall probability
                    fall_prob = (
                        self.gravitational_fall_probability if fall_step == 0
                        else self.gravitational_fall_probability_later
                    )
                    rand_mask = np.random.random(total_candidates) < fall_prob
                    cs_y = cs_y[rand_mask]
                    cs_x = cs_x[rand_mask]
                    cn_y = cn_y[rand_mask]
                    cn_x = cn_x[rand_mask]

                    # Limit moves
                    limit = min(max_moves_per_step, len(cs_y))
                    if limit > 0:
                        sel = np.random.choice(len(cs_y), size=limit, replace=False)
                        cs_y, cs_x = cs_y[sel], cs_x[sel]
                        cn_y, cn_x = cn_y[sel], cn_x[sel]

                        # ------------------------------------------------------------------
                        # Ensure each grid index participates in AT MOST one swap this pass.
                        # This prevents material "loss" when two candidate moves target the
                        # same cell (or when one cell is both a source and a target).
                        # ------------------------------------------------------------------
                        cs_y, cs_x, cn_y, cn_x = self._dedupe_swap_pairs(cs_y, cs_x, cn_y, cn_x)

                        # Debug-mode assertion: verify true uniqueness
                        if __debug__:
                            assert len(np.unique(cs_y * self.width + cs_x)) == len(cs_y)
                            assert len(np.unique(cn_y * self.width + cn_x)) == len(cn_y)

                        # Perform the swaps
                        solid_materials = self.material_types[cs_y, cs_x].copy()
                        cavity_materials = self.material_types[cn_y, cn_x].copy()
                        self.material_types[cn_y, cn_x] = solid_materials
                        self.material_types[cs_y, cs_x] = cavity_materials

                        solid_temps = self.temperature[cs_y, cs_x].copy()
                        self.temperature[cn_y, cn_x] = solid_temps
                        self.temperature[cs_y, cs_x] = (solid_temps + 273.15) / 2

                        moves_made += limit
                        step_changes = True
                        changes_made = True

            # Update masks for next iteration if changes were made
            if step_changes:
                solid_mask = self._get_solid_mask()
                cavity_mask = ~solid_mask
            else:
                break  # No changes

        return changes_made

    def _apply_fluid_dynamics_vectorized(self):
        """Fast vectorized fluid dynamics using morphological operations"""
        changes_made = False

        # Treat true vacuum as the lightest fluid so that trapped SPACE pockets
        # can buoyantly migrate toward the surface just like AIR.

        fluid_mask_total = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.SPACE) |
            (self.material_types == MaterialType.WATER_VAPOR) |
            (self.material_types == MaterialType.MAGMA) |
            (self.material_types == MaterialType.WATER)
        )

        if not np.any(fluid_mask_total):
            return False

        # Only consider actual fluid *cells* as sources – exclude SPACE so we
        # never pick a vacuum cell as the thing that moves.  (Using a vacuum as
        # the "source" was letting AIR swap outward into SPACE.)
        air_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )

        # Fluid migration using vectorized operations (includes gases and magma/water)
        air_mask = fluid_mask_total
        distances = self._get_distances_from_center()

        # Find air cells adjacent to non-space materials (using circular kernel to reduce artifacts)
        non_space_mask = (self.material_types != MaterialType.SPACE)
        kernel = self._circular_kernel_3x3 if self.neighbor_count == 8 else self._collapse_kernel_4
        air_near_materials = air_mask & ndimage.binary_dilation(non_space_mask, structure=kernel)

        if np.any(air_near_materials):
            air_coords = np.where(air_near_materials)

            # Sample based on performance configuration
            total_air = len(air_coords[0])
            sample_size = max(1, int(total_air * self.process_fraction_air))
            if sample_size < total_air:
                indices = np.random.choice(total_air, size=sample_size, replace=False)
                air_coords = (air_coords[0][indices], air_coords[1][indices])

            air_distances = distances[air_coords]

            # Build isotropic neighbor offsets (radius ≤2)
            offsets = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    if self._circular_kernel_5x5[dy + 2, dx + 2]:
                        offsets.append((dy, dx))

            # Pool all potential migration pairs
            cand_air_y = []
            cand_air_x = []
            cand_nei_y = []
            cand_nei_x = []

            for dy, dx in offsets:
                neighbor_y = air_coords[0] + dy
                neighbor_x = air_coords[1] + dx

                in_bounds = (
                    (neighbor_y >= 0) & (neighbor_y < self.height) &
                    (neighbor_x >= 0) & (neighbor_x < self.width)
                )
                if not np.any(in_bounds):
                    continue

                vidx = np.where(in_bounds)[0]
                a_y = air_coords[0][vidx]
                a_x = air_coords[1][vidx]
                n_y = neighbor_y[vidx]
                n_x = neighbor_x[vidx]

                neighbor_materials = self.material_types[n_y, n_x]
                neighbor_distances = distances[n_y, n_x]
                air_distances_valid = air_distances[vidx]

                not_space = (neighbor_materials != MaterialType.SPACE)
                # Allow lateral movement in addition to upward buoyant rise.
                # "Toward surface" means strictly farther from the center.  A purely
                # horizontal (lateral) move keeps the radial distance ~equal.  Use a
                # small tolerance (10 % of a cell size) to treat such moves as
                # lateral rather than downward.

                toward_surface = neighbor_distances > air_distances_valid
                lateral_move = np.isclose(neighbor_distances, air_distances_valid, atol=self.cell_size * 0.1)

                toward_surface |= lateral_move

                # Porous or non-solid check
                props_ok = [not self.material_db.get_properties(m).is_solid or self.material_db.get_properties(m).porosity > 0.1 for m in neighbor_materials]
                props_ok = np.array(props_ok, dtype=bool)

                can_move = not_space & toward_surface & props_ok

                if np.any(can_move):
                    idx = np.where(can_move)[0]
                    cand_air_y.append(a_y[idx])
                    cand_air_x.append(a_x[idx])
                    cand_nei_y.append(n_y[idx])
                    cand_nei_x.append(n_x[idx])

            if cand_air_y:
                ay = np.concatenate(cand_air_y)
                ax = np.concatenate(cand_air_x)
                ny = np.concatenate(cand_nei_y)
                nx = np.concatenate(cand_nei_x)

                total = len(ay)
                if total > 0:
                    prob_mask = np.random.random(total) < self.fluid_migration_probability
                    ay, ax, ny, nx = ay[prob_mask], ax[prob_mask], ny[prob_mask], nx[prob_mask]

                    limit = min(len(ay), 50)
                    if limit > 0:
                        sel = np.random.choice(len(ay), size=limit, replace=False)
                        ay, ax, ny, nx = ay[sel], ax[sel], ny[sel], nx[sel]

                        # ------------------------------------------------------------------
                        # Ensure each grid index participates in AT MOST one swap this pass.
                        # This prevents material "loss" when two candidate moves target the
                        # same cell (or when one cell is both a source and a target).
                        # ------------------------------------------------------------------
                        ay, ax, ny, nx = self._dedupe_swap_pairs(ay, ax, ny, nx)

                        # Debug-mode assertion: verify true uniqueness
                        if __debug__:
                            assert len(np.unique(ay * self.width + ax)) == len(ay)
                            assert len(np.unique(ny * self.width + nx)) == len(ny)

                        # Perform the swaps
                        air_mats = self.material_types[ay, ax].copy()
                        nei_mats = self.material_types[ny, nx].copy()
                        self.material_types[ny, nx] = air_mats
                        self.material_types[ay, ax] = nei_mats
                        changes_made = True

        # ------------------------------------------------------------------
        # SECOND PASS – LIQUID INFILTRATION
        # Water and magma should be able to flow into neighbouring cavities
        # (AIR or SPACE) even when those cavities are beside or below the
        # fluid.  This pass fills holes by swapping liquid cells into adjacent
        # cavities at equal or lower gravitational potential.
        # ------------------------------------------------------------------

        liquid_mask = (
            (self.material_types == MaterialType.WATER) |
            (self.material_types == MaterialType.MAGMA)
        )

        cavity_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.SPACE)
        )

        if np.any(liquid_mask) and np.any(cavity_mask):
            liquids_near_cavity = liquid_mask & ndimage.binary_dilation(cavity_mask, structure=kernel)

            if np.any(liquids_near_cavity):
                ly_all, lx_all = np.where(liquids_near_cavity)

                # Sub-sample for performance; use process_fraction_water if defined.
                sample_frac = getattr(self, 'process_fraction_water', 0.2)
                total_liq = len(ly_all)
                sample_size = max(1, int(total_liq * sample_frac))
                if sample_size < total_liq:
                    idx = np.random.choice(total_liq, size=sample_size, replace=False)
                    ly_all, lx_all = ly_all[idx], lx_all[idx]

                liq_distances = distances[ly_all, lx_all]

                cand_ly, cand_lx, cand_cy, cand_cx = [], [], [], []

                for dy, dx in offsets:
                    cy = ly_all + dy
                    cx = lx_all + dx

                    in_bounds = (
                        (cy >= 0) & (cy < self.height) &
                        (cx >= 0) & (cx < self.width)
                    )
                    if not np.any(in_bounds):
                        continue

                    vidx = np.where(in_bounds)[0]
                    l_y = ly_all[vidx]; l_x = lx_all[vidx]
                    cav_y = cy[vidx]; cav_x = cx[vidx]

                    is_cavity = cavity_mask[cav_y, cav_x]
                    if not np.any(is_cavity):
                        continue

                    cav_idx = np.where(is_cavity)[0]
                    l_y = l_y[cav_idx]; l_x = l_x[cav_idx]
                    cav_y = cav_y[cav_idx]; cav_x = cav_x[cav_idx]

                    cav_distances = distances[cav_y, cav_x]
                    l_distances = liq_distances[vidx][cav_idx]
                    same_or_down = cav_distances <= l_distances + (self.cell_size * 0.1)

                    if np.any(same_or_down):
                        good = np.where(same_or_down)[0]
                        cand_ly.append(l_y[good])
                        cand_lx.append(l_x[good])
                        cand_cy.append(cav_y[good])
                        cand_cx.append(cav_x[good])

                if cand_ly:
                    ly = np.concatenate(cand_ly)
                    lx = np.concatenate(cand_lx)
                    cy = np.concatenate(cand_cy)
                    cx = np.concatenate(cand_cx)

                    total_pairs = len(ly)
                    prob_mask = np.random.random(total_pairs) < self.fluid_migration_probability
                    ly, lx, cy, cx = ly[prob_mask], lx[prob_mask], cy[prob_mask], cx[prob_mask]

                    ly, lx, cy, cx = self._dedupe_swap_pairs(ly, lx, cy, cx)

                    if len(ly) > 0:
                        liq_mats = self.material_types[ly, lx].copy()
                        cav_mats = self.material_types[cy, cx].copy()
                        self.material_types[cy, cx] = liq_mats
                        self.material_types[ly, lx] = cav_mats

                        liq_temps = self.temperature[ly, lx].copy()
                        cav_temps = self.temperature[cy, cx].copy()
                        self.temperature[cy, cx] = liq_temps
                        self.temperature[ly, lx] = cav_temps

                        changes_made = True

        return changes_made

    def _apply_density_stratification_local_vectorized(self):
        """Physics-correct vectorized density stratification preserving local gravitational interactions"""
        changes_made = False

        # Calculate effective density grid with proper thermal expansion
        effective_density_grid = self._calculate_effective_density(self.temperature)
        distances = self._get_distances_from_center()

        # Find all non-space materials that could potentially move
        non_space_mask = (self.material_types != MaterialType.SPACE)
        if not np.any(non_space_mask):
            return False

        non_space_coords = np.where(non_space_mask)
        total_cells = len(non_space_coords[0])

        # Sample cells based on performance configuration (just like original)
        sample_size = max(self.density_min_sample_size, int(total_cells * self.density_sample_fraction))
        sample_size = min(sample_size, total_cells)

        if sample_size == 0:
            return False

        # Vectorized sampling of cells to check
        cell_indices = np.random.choice(total_cells, size=sample_size, replace=False)
        sample_y = non_space_coords[0][cell_indices]
        sample_x = non_space_coords[1][cell_indices]

        # Filter to mobile materials only (preserve original mobility logic)
        sample_materials = self.material_types[sample_y, sample_x]
        sample_temps = self.temperature[sample_y, sample_x]

        # Original mobility conditions: gases, liquids, and hot solids
        is_gas = ((sample_materials == MaterialType.AIR) |
                  (sample_materials == MaterialType.WATER_VAPOR))
        # Liquids that can buoyantly migrate (exclude magma which should stay until erupting)
        is_liquid = (sample_materials == MaterialType.WATER)
        is_hot_solid = (sample_temps > self.hot_solid_temperature_threshold)  # Hot solids behave like fluids

        # Light cold solids (e.g. ICE, PUMICE) can also migrate through fluids
        is_light_solid = (
            (sample_materials == MaterialType.ICE) |
            (sample_materials == MaterialType.PUMICE)
        )

        mobile_mask = is_gas | is_liquid | is_hot_solid | is_light_solid
        if not np.any(mobile_mask):
            return False

        # Get mobile cells coordinates
        mobile_indices = np.where(mobile_mask)[0]
        mobile_y = sample_y[mobile_indices]
        mobile_x = sample_x[mobile_indices]

        # Build isotropic offset list (radius ≤2)
        offsets = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                if self._circular_kernel_5x5[dy + 2, dx + 2]:
                    offsets.append((dy, dx))

        # Pool all candidate swaps
        cand_m_y = []
        cand_m_x = []
        cand_n_y = []
        cand_n_x = []

        for dy, dx in offsets:
            neighbor_y = mobile_y + dy
            neighbor_x = mobile_x + dx

            in_bounds = (
                (neighbor_y >= 0) & (neighbor_y < self.height) &
                (neighbor_x >= 0) & (neighbor_x < self.width)
            )
            if not np.any(in_bounds):
                continue

            vidx = np.where(in_bounds)[0]
            m_y = mobile_y[vidx]
            m_x = mobile_x[vidx]
            n_y = neighbor_y[vidx]
            n_x = neighbor_x[vidx]

            neighbor_materials = self.material_types[n_y, n_x]
            is_non_space = (neighbor_materials != MaterialType.SPACE)
            if not np.any(is_non_space):
                continue

            ns_idx = np.where(is_non_space)[0]
            m_y = m_y[ns_idx]; m_x = m_x[ns_idx]
            n_y = n_y[ns_idx]; n_x = n_x[ns_idx]

            mobile_densities = effective_density_grid[m_y, m_x]
            neighbor_densities = effective_density_grid[n_y, n_x]
            mobile_distances = distances[m_y, m_x]
            neighbor_distances = distances[n_y, n_x]

            case1 = (mobile_distances < neighbor_distances) & (mobile_densities < neighbor_densities)
            case2 = (mobile_distances > neighbor_distances) & (mobile_densities > neighbor_densities)
            should_swap = case1 | case2

            min_d = np.minimum(mobile_densities, neighbor_densities)
            max_d = np.maximum(mobile_densities, neighbor_densities)
            ratio = np.divide(max_d, min_d, out=np.ones_like(max_d), where=(min_d > 0))
            significant = ratio >= self.density_ratio_threshold

            final = should_swap & significant
            if np.any(final):
                idx = np.where(final)[0]
                cand_m_y.append(m_y[idx])
                cand_m_x.append(m_x[idx])
                cand_n_y.append(n_y[idx])
                cand_n_x.append(n_x[idx])

        if cand_m_y:
            my = np.concatenate(cand_m_y)
            mx = np.concatenate(cand_m_x)
            ny = np.concatenate(cand_n_y)
            nx = np.concatenate(cand_n_x)

            total = len(my)
            if total > 0:
                prob_mask = np.random.random(total) < self.density_swap_probability
                my, mx, ny, nx = my[prob_mask], mx[prob_mask], ny[prob_mask], nx[prob_mask]

                # ------------------------------------------------------------------
                # Swap deduplication – ensure every cell appears at most once across
                # sources *and* destinations.
                # ------------------------------------------------------------------
                my, mx, ny, nx = self._dedupe_swap_pairs(my, mx, ny, nx)

                # ------------------------------------------------------------------
                # Safety filter: never allow a swap whose destination is SPACE. This
                # prevents subtle mass-loss bugs where material could be swapped into
                # vacuum and effectively removed from the planet.
                # ------------------------------------------------------------------
                if len(ny) > 0:
                    dest_space_mask = (self.material_types[ny, nx] == MaterialType.SPACE)
                    if np.any(dest_space_mask):
                        keep_mask = ~dest_space_mask
                        my, mx, ny, nx = my[keep_mask], mx[keep_mask], ny[keep_mask], nx[keep_mask]

                limit = len(my)
                if limit > 0:
                    if __debug__:
                        assert len(np.unique(my * self.width + mx)) == len(my)
                        assert len(np.unique(ny * self.width + nx)) == len(ny)
                        # Destination should never be SPACE after filtering
                        assert not np.any(self.material_types[ny, nx] == MaterialType.SPACE)

                    mobile_mats = self.material_types[my, mx].copy()
                    neighbor_mats = self.material_types[ny, nx].copy()
                    self.material_types[my, mx] = neighbor_mats
                    self.material_types[ny, nx] = mobile_mats

                    mobile_temps = self.temperature[my, mx].copy()
                    neighbor_temps = self.temperature[ny, nx].copy()
                    self.temperature[my, mx] = neighbor_temps
                    self.temperature[ny, nx] = mobile_temps

                    changes_made = True

        # Handle phase transitions for water/vapor (original logic)
        if changes_made:
            self._apply_convective_phase_transitions()

        return changes_made

    def _calculate_effective_density(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate temperature-dependent effective densities using thermal expansion"""

        # Get material properties for thermal expansion (uses material database)
        effective_density = np.zeros_like(self.density)
        non_space_mask = (self.material_types != MaterialType.SPACE)

        if not np.any(non_space_mask):
            return effective_density

        # Calculate effective densities for all non-space cells using proper volumetric expansion
        non_space_coords = np.where(non_space_mask)
        materials = self.material_types[non_space_coords]
        temperatures = temperature[non_space_coords]
        base_densities = self.density[non_space_coords]

        # Vectorized thermal expansion calculation using material properties
        # ρ_eff = ρ₀ / (1 + β(T - T₀)) where β is volumetric expansion coefficient
        expansion_coeffs = np.array([self.material_db.get_properties(mat).thermal_expansion for mat in materials.flat])
        volumetric_expansion = 1.0 + expansion_coeffs * (temperatures - self.reference_temperature)
        volumetric_expansion = np.maximum(0.1, volumetric_expansion)  # Prevent division by zero/negative
        effective_densities = base_densities / volumetric_expansion
        effective_densities = np.maximum(0.01, effective_densities)  # Prevent negative density

        # Fill the effective density grid
        effective_density[non_space_coords] = effective_densities

        return effective_density

    def _solve_radiative_cooling_newton_raphson_implicit(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """
        Solve radiative cooling using full Stefan-Boltzmann T^4 with Newton-Raphson implicit solver

        Method: Solves dT/dt = -α(T^4 - T_space^4) using Newton-Raphson iteration
        Advantages: Unconditionally stable, physically accurate, handles large temperature differences
        Disadvantages: More computationally expensive (3-5 iterations typically)
        """
        working_temp = temperature.copy()

        # Find surface cells that radiate to space
        space_mask = ~non_space_mask
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )

        # Surface cells are those adjacent to space or outer atmosphere
        surface_candidates = ndimage.binary_dilation(space_mask, structure=self._circular_kernel_5x5) & non_space_mask

        if not np.any(surface_candidates):
            return working_temp

        # Get radiating cells
        radiative_coords = np.where(surface_candidates)
        T_radiating = working_temp[radiative_coords]
        T_space = self.space_temperature

        # Only process cells that are actually cooling (T > T_space)
        cooling_mask = T_radiating > T_space
        if not np.any(cooling_mask):
            return working_temp

        cooling_idx = np.where(cooling_mask)[0]
        T_cooling = T_radiating[cooling_idx]

        # Get material properties
        material_types_cooling = self.material_types[radiative_coords][cooling_idx]
        emissivity = np.array([self.material_db.get_properties(mat_type).emissivity for mat_type in material_types_cooling])
        density_cooling = self.density[radiative_coords][cooling_idx]
        specific_heat_cooling = self.specific_heat[radiative_coords][cooling_idx]

        # Stefan-Boltzmann constant in geological units
        stefan_geological = self.stefan_boltzmann_geological / 1000.0  # Conservative scaling

        # Analytical solution for dT/dt = -α(T^4 - T_space^4) where α = σε/(ρcp·thickness)
        surface_thickness = self.cell_size * self.surface_radiation_depth_fraction
        alpha = (stefan_geological * emissivity * self.radiative_cooling_efficiency) / (density_cooling * specific_heat_cooling * surface_thickness)

        # For the equation dT/dt = -α(T^4 - T0^4), the analytical solution is complex
        # Use simpler Newton-Raphson approach: T_new = T_old - dt*α*(T_old^4 - T0^4)/(4*α*T_old^3)
        # This is equivalent to implicit Euler for radiation and is unconditionally stable

        dt_seconds = self.dt * self.seconds_per_year

        # Newton-Raphson iteration for implicit radiation (typically converges in 1-2 iterations)
        T_new = T_cooling.copy()
        for iteration in range(3):  # Usually converges quickly
            f = T_new - T_cooling + dt_seconds * alpha * (T_new**4 - T_space**4)
            df_dt = 1.0 + dt_seconds * alpha * 4.0 * T_new**3

            # Newton-Raphson update with safety bounds
            delta_T = -f / df_dt
            T_new += delta_T

            # Keep temperatures physical
            T_new = np.maximum(T_new, T_space)

            # Check convergence
            if np.max(np.abs(delta_T)) < 0.1:  # 0.1K tolerance
                break

        # Apply the updated temperatures
        final_coords_y = radiative_coords[0][cooling_idx]
        final_coords_x = radiative_coords[1][cooling_idx]
        working_temp[final_coords_y, final_coords_x] = T_new

        # ------------------  POWER / FLUX ACCOUNTING  ------------------
        # Re-compute instantaneous radiative flux for bookkeeping
        # Greenhouse attenuation (reuse same dynamic factor as linearized method)
        water_vapor_mask = (self.material_types == MaterialType.WATER_VAPOR)
        total_water_vapor_mass = np.sum(self.density[water_vapor_mask]) if np.any(water_vapor_mask) else 0.0
        if total_water_vapor_mass > 0:
            vapor_factor = np.log1p(total_water_vapor_mass / self.greenhouse_vapor_scaling) / 10.0
            greenhouse_factor = self.base_greenhouse_effect + (self.max_greenhouse_effect - self.base_greenhouse_effect) * np.tanh(vapor_factor)
        else:
            greenhouse_factor = self.base_greenhouse_effect

        effective_stefan = stefan_geological * (1.0 - greenhouse_factor)
        power_per_area = effective_stefan * emissivity * (T_new**4 - T_space**4)  # J/(year·m²)

        surface_thickness = self.cell_size * self.surface_radiation_depth_fraction
        volumetric_power_density = power_per_area / (surface_thickness * self.seconds_per_year)  # W/m³

        # Negative sign: cooling removes energy
        self.power_density[final_coords_y, final_coords_x] -= volumetric_power_density

        total_radiative_power = np.sum(volumetric_power_density * (self.cell_size ** 3))  # W (positive magnitude)
        self.thermal_fluxes['radiative_output'] = total_radiative_power

        return working_temp

    def _settle_unsupported_chunks(self) -> bool:
        """Original chunk-based settling algorithm (raft drop).

        Moves every *unsupported* connected solid component toward the centre
        of mass by up to ``terminal_settle_velocity`` cells in one shot.
        Returns ``True`` iff at least one chunk moved.  This implementation is
        the exact code that previously passed all regression tests – no
        incremental tweaks."""

        # Only rigid solids take part in chunk settling – fluids are excluded.
        solid = self._get_solid_mask()

        # Radial unit vectors pointing *toward* COM (-1, 0, +1 per axis)
        toward_y = np.sign(self.center_of_mass[1] - self.y_coords).astype(np.int8)
        toward_x = np.sign(self.center_of_mass[0] - self.x_coords).astype(np.int8)

        nbr_y = np.clip(self.y_coords + toward_y, 0, self.height - 1)
        nbr_x = np.clip(self.x_coords + toward_x, 0, self.width  - 1)

        supported = solid & solid[nbr_y, nbr_x]
        unsupported = solid & ~supported
        if not np.any(unsupported):
            return False

        # Label connected unsupported regions (8-neighbour connectivity)
        structure = np.ones((3, 3), dtype=bool)
        labels, n = ndimage.label(unsupported, structure=structure)

        moved = False
        max_cells = (
            self.height + self.width if np.isinf(self.terminal_settle_velocity)
            else int(self.terminal_settle_velocity)
        )

        # Pre-compute fluid mask once (anything that is *not* solid)
        is_fluid = ~solid

        for label_id in range(1, n + 1):
            mask = labels == label_id
            if not np.any(mask):
                continue

            ys, xs = np.where(mask)
            dy = np.sign(self.center_of_mass[1] - ys).astype(np.int8)
            dx = np.sign(self.center_of_mass[0] - xs).astype(np.int8)

            # Ray-cast one cell at a time (up to terminal velocity)
            fall = 0
            for step in range(1, max_cells + 1):
                y_new = ys + dy * step
                x_new = xs + dx * step
                in_bounds = (
                    (y_new >= 0) & (y_new < self.height) &
                    (x_new >= 0) & (x_new < self.width)
                )
                if not np.all(in_bounds):
                    break  # would leave grid

                if np.any(solid[y_new, x_new]):
                    break  # collided with solid – stop at previous step

                fall = step  # this step is OK; try next

            if fall == 0:
                continue  # chunk cannot move

            moved = True
            self._translate_chunk(mask, dy[0] * fall, dx[0] * fall)

        return moved

    # ------------------------------------------------------------------
    # Helper – translate a boolean mask by (dy, dx) swapping materials & temps
    # ------------------------------------------------------------------
    def _translate_chunk(self, mask: np.ndarray, dy: int, dx: int):
        """In-place translation of a material/temperature chunk by (dy, dx)."""
        if dy == 0 and dx == 0:
            return

        ys, xs = np.nonzero(mask)
        dest_y = ys + dy
        dest_x = xs + dx

        # Flat indices for fast swapping
        src_flat = ys * self.width + xs
        tgt_flat = dest_y * self.width + dest_x

        flat_types = self.material_types.ravel()
        flat_temp  = self.temperature.ravel()

        src_types = flat_types[src_flat].copy()
        src_temps = flat_temp[src_flat].copy()

        flat_types[src_flat] = flat_types[tgt_flat]
        flat_temp[src_flat] = flat_temp[tgt_flat]

        flat_types[tgt_flat] = src_types
        flat_temp[tgt_flat] = src_temps

    def reset(self):
        """Completely reset the simulation to its initial planetary state.

        This reproduces the same behavior as the visualizer's keyboard
        'R' shortcut and is now the canonical way to reset a simulation
        instance. It preserves the configuration parameters (grid size,
        performance options, etc.) while re-initializing all dynamic state.
        """
        # Re-initialize planetary setup and derived fields
        self._setup_planetary_conditions()
        self._calculate_planetary_pressure()

        # Reset per-cell dynamic fields
        self.age.fill(0.0)
        self.power_density.fill(0.0)

        # Reset global counters and history
        self.time = 0.0
        self.history.clear()
        self._adaptive_time_steps = []

        # Ensure subsequent steps will perform settling when needed
        if hasattr(self, "_need_settle"):
            self._need_settle = True

        # Clear time-series data used for graphs
        for key in self.time_series:
            self.time_series[key].clear()

        # Recompute material properties so all cached values are valid
        self._properties_dirty = True
        self._update_material_properties()

        # Reset thermal flux bookkeeping
        self.thermal_fluxes = {
            'solar_input': 0.0,
            'radiative_output': 0.0,
            'internal_heating': 0.0,
            'atmospheric_heating': 0.0,
            'net_flux': 0.0,
        }

    # ------------------------------------------------------------------
    # Convenience editing helpers
    # ------------------------------------------------------------------
    def delete_material_blob(self, x: int, y: int, radius: int = 1):
        """Remove material in a circular blob by replacing it with SPACE.

        This convenience method is primarily intended for testing and
        interactive experimentation (e.g., drilling craters).  It performs a
        simple in‐place edit of the ``material_types`` array without touching
        any of the dynamic state; material/thermal properties will be updated
        automatically on the next call to :py:meth:`step_forward`.
        """
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        self.material_types[ny, nx] = MaterialType.SPACE

        # Mark derived property grids dirty so they refresh on the next step
        self._properties_dirty = True

    def add_material_blob(self, x: int, y: int, radius: int, material_type):
        """Fill a circular patch with the chosen *material_type*.

        The helper is complementary to :py:meth:`delete_material_blob` and is
        mainly used by the GUI editor/visualiser.  It performs an in-place
        modification of ``material_types`` and then marks derived property
        grids dirty so densities, thermal properties, etc. are refreshed on the
        next simulation step.

        Parameters
        ----------
        x, y : int
            Centre of the blob in grid coordinates.
        radius : int
            Radius of the circular area in cells.
        material_type : MaterialType | int
            Material to paint.  Accepts either a ``MaterialType`` enum member
            or the underlying integer code (helpful for quick GUI usage).
        """

        # Convert plain integers to MaterialType if needed (robust GUI usage)
        if not isinstance(material_type, MaterialType):
            try:
                material_type = MaterialType(material_type)
            except ValueError:
                raise ValueError(f"Invalid material_type: {material_type}")

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        self.material_types[ny, nx] = material_type
                        if material_type == MaterialType.MAGMA:
                            self.temperature[ny, nx] = max(
                                self.temperature[ny, nx],
                                getattr(self, 'melting_temperature', 1200 + 273.15) + 100.0,
                            )
                        else:
                            # Default to near-surface "room" temperature so freshly
                            # placed rocks don't immediately melt or freeze.
                            self.temperature[ny, nx] = 300.0  # Kelvin

        # Mark derived grids dirty so they update next step
        self._properties_dirty = True

        # Refresh material property caches right away for the new cells only
        self._update_material_properties()

    # User-visible helper – toggle verbose logging / performance metrics
    def toggle_logging(self):
        """Toggle verbose DEBUG logging and timing output.

        Called from the visualiser when the user presses the 'L' shortcut.
        """
        self.logging_enabled = not self.logging_enabled

        # Raise or lower the logger level accordingly
        new_level = logging.DEBUG if self.logging_enabled else logging.INFO
        self.logger.setLevel(new_level)

        # Brief confirmation so the user sees feedback even when entering
        # DEBUG mode (which may be very chatty afterwards)
        self.logger.info(f"Logging {'ENABLED' if self.logging_enabled else 'DISABLED'} – level set to {logging.getLevelName(new_level)}")

