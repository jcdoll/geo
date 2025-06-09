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
        
        # General physics settings (not performance-dependent)
        self.atmospheric_diffusivity_enhancement = 5.0  # Now safe to enhance with DuFort-Frankel!
        self.atmospheric_convection_mixing = 0.3        # Fraction of temperature difference to mix per step (fast convection)
        self.interface_diffusivity_enhancement = 1.5    # Enhanced heat transfer at material interfaces
        self.surface_radiation_depth_fraction = 0.1     # Fraction of cell depth that participates in surface radiation
        self.radiative_cooling_efficiency = 0.8         # Cooling efficiency factor for Stefan-Boltzmann radiation (increased for better balance)
        
        # Temperature constants
        self.space_temperature = 2.7                    # Cosmic background temperature (K)
        self.reference_temperature = 273.15             # Reference temperature for thermal expansion (K)
        self.core_temperature = 1200.0 + 273.15         # Initial planetary core temperature (K) - warmer for stability
        self.surface_temperature = 50.0 + 273.15        # Initial planetary surface temperature (K) - warmer for stability
        self.temperature_decay_constant = 2.0           # Temperature gradient decay factor - steeper gradient
        self.melting_temperature = 1200 + 273.15        # General melting temperature threshold (K)
        self.core_heating_depth_scale = 0.5             # Exponential scale factor for core heating vs depth
        
        # Pressure constants  
        self.surface_pressure = 0.1                     # Surface pressure (MPa)
        self.atmospheric_scale_height = 8400            # Atmospheric scale height (m)
        self.average_gravity = 9.81                     # Average gravitational acceleration (m/s²)
        self.average_solid_density = 3000               # Average solid rock density (kg/m³)
        self.average_fluid_density = 2000               # Average fluid density (kg/m³)
        
        # Solar and greenhouse constants (balanced for stable temperatures)
        self.solar_constant = 1361                      # Solar constant (W/m²)
        self.planetary_distance_factor = 0.00001       # Distance factor for solar intensity - further reduced to prevent hot spots (was 0.0001)
        self.atmospheric_absorption_per_cell = 0.0005  # Atmospheric absorption fraction per cell (0.05% per layer) - reduced to prevent hot spots
        self.base_greenhouse_effect = 0.6              # Base greenhouse effect fraction - increased to retain more heat
        self.max_greenhouse_effect = 0.85              # Maximum greenhouse effect fraction - increased
        self.greenhouse_vapor_scaling = 1000.0         # Water vapor mass scaling for greenhouse effect
        
        # Material mobility probabilities
        self.gravitational_fall_probability = 0.5       # Initial fall probability for collapse
        self.gravitational_fall_probability_later = 0.3 # Later fall probability for collapse
        self.fluid_migration_probability = 0.3          # Air/fluid migration probability
        self.density_swap_probability = 0.3             # Density stratification swap probability
        
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
    
    def _setup_performance_config(self, quality: int):
        """Setup performance configuration based on quality level"""
        if quality == 1:
            # Full quality - maximum accuracy
            self.process_fraction_mobile = 1.0      # Process all mobile cells
            self.process_fraction_solid = 1.0       # Process all solid cells  
            self.process_fraction_air = 1.0         # Process all air cells
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
        
        # Distance factors for diagonal neighbors (1/√2 ≈ 0.707)
        self.distance_factors_8 = np.array([0.707, 1.0, 0.707, 1.0, 1.0, 0.707, 1.0, 0.707])
        
        # Pre-compute coordinate grids for vectorized operations
        self.y_coords, self.x_coords = np.ogrid[:self.height, :self.width]
        
        # Create circular morphological kernels to reduce grid artifacts
        self._circular_kernel_3x3 = self._create_circular_kernel(3)
        self._circular_kernel_5x5 = self._create_circular_kernel(5)
        
        # Collapse kernels for geological processes
        self._collapse_kernel_4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)  # 4-neighbor
        self._collapse_kernel_8 = self._circular_kernel_3x3  # Use circular kernel for 8-neighbor
    
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
        
        # Step 2: Solve radiative cooling analytically (unconditionally stable)
        working_temp = self._solve_radiative_cooling_analytical(working_temp, non_space_mask)
        
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
        print(f"DEBUG OPERATOR SPLIT: dt={self.dt:.3f}y, diff_factor={diffusion_stability:.3f}")
        print(f"DEBUG TEMP: {avg_temp_before:.1f}°C → {avg_temp_after:.1f}°C (Δ={avg_temp_after-avg_temp_before:.1f}°C)")
        
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
        
        # Stability analysis for PURE DIFFUSION only (no sources)
        dx_squared = self.cell_size ** 2
        max_alpha = np.max(thermal_diffusivity[non_space_mask]) if np.any(non_space_mask) else 0.0
        
        # Pure diffusion stability limit
        diffusion_dt_limit = dx_squared / (4.0 * max_alpha) if max_alpha > 0 else float('inf')
        
        # Adaptive time step for diffusion
        target_dt_seconds = min(diffusion_dt_limit, self.dt * self.seconds_per_year)
        target_dt_seconds = max(target_dt_seconds, self.dt * self.seconds_per_year / 50.0)  # Max 50 substeps
        
        # Convert back to years
        adaptive_dt = target_dt_seconds / self.seconds_per_year
        stability_factor = adaptive_dt / self.dt
        
        # Use sub-steps for stability
        num_substeps = max(1, min(50, int(np.ceil(self.dt / adaptive_dt))))
        actual_effective_dt = self.dt / num_substeps
        actual_stability_factor = actual_effective_dt / self.dt
        
        # Store debugging info
        self._max_thermal_diffusivity = max_alpha
        self._diffusion_substeps = num_substeps
        
        # Pure diffusion sub-stepping (much simpler without sources)
        new_temp = temperature.copy()
        
        for step in range(num_substeps):
            # Pure diffusion step only
            new_temp = self._pure_diffusion_step(new_temp, thermal_diffusivity, actual_effective_dt, non_space_mask)
        
        return new_temp, actual_stability_factor
    
    def _pure_diffusion_step(self, temperature: np.ndarray, thermal_diffusivity: np.ndarray, 
                            dt: float, non_space_mask: np.ndarray) -> np.ndarray:
        """Pure diffusion step only"""
        # Convert dt to seconds for proper units
        dt_seconds = dt * self.seconds_per_year
        dx_squared = self.cell_size ** 2
        
        # Calculate Laplacian using simple 4-neighbor stencil (most efficient)
        temp_left = np.roll(temperature, 1, axis=1)
        temp_right = np.roll(temperature, -1, axis=1)
        temp_up = np.roll(temperature, 1, axis=0)
        temp_down = np.roll(temperature, -1, axis=0)
        
        # Standard 4-neighbor Laplacian
        temp_laplacian = temp_left + temp_right + temp_up + temp_down - 4 * temperature
        
        # Zero out Laplacian for space cells (no diffusion)
        temp_laplacian[~non_space_mask] = 0.0
        
        # Combined diffusion update (vectorized)
        # dT/dt = α∇²T
        diffusion_change = thermal_diffusivity * dt_seconds * temp_laplacian / dx_squared
        
        # Apply updates only to non-space cells
        new_temp = temperature.copy()
        if np.any(non_space_mask):
            new_temp[non_space_mask] += diffusion_change[non_space_mask]
        
        return new_temp
    
    def _dufort_frankel_step(self, thermal_diffusivity: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """DuFort-Frankel diffusion step (unconditionally stable for pure diffusion)"""
        # Initialize temperature arrays for DuFort-Frankel scheme
        if not hasattr(self, '_temp_previous'):
            self._temp_previous = self.temperature.copy()
            self._step_count = 0
        
        # DuFort-Frankel scheme: explicit but unconditionally stable
        dt = self.dt  # years
        dx_squared = self.cell_size ** 2  # m²
        
        # Calculate Laplacian using 4-neighbor stencil (more stable than 8-neighbor)
        temp_left = np.roll(self.temperature, 1, axis=1)
        temp_right = np.roll(self.temperature, -1, axis=1)
        temp_up = np.roll(self.temperature, 1, axis=0)
        temp_down = np.roll(self.temperature, -1, axis=0)
        
        # Standard 4-neighbor Laplacian
        temp_laplacian = temp_left + temp_right + temp_up + temp_down - 4 * self.temperature
        
        # Zero out Laplacian for space cells (no diffusion)
        temp_laplacian[~non_space_mask] = 0.0
        
        # DuFort-Frankel update: T^(n+1) = T^(n-1) + 2*dt*α*∇²T^n / dx²
        diffusion_term = 2.0 * dt * thermal_diffusivity * temp_laplacian / dx_squared
        
        if self._step_count == 0:
            # First step: use forward Euler (T^(-1) = T^(0))
            new_temp = self.temperature + diffusion_term
            else:
            # DuFort-Frankel: use previous temperature
            new_temp = self._temp_previous + diffusion_term
        
        # Update previous temperature for next step
        self._temp_previous = self.temperature.copy()
        self._step_count += 1
        
        return new_temp
    
    def _adaptive_source_integration(self, temperature: np.ndarray, heat_source: np.ndarray, 
                                   non_space_mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Adaptive integration of heat source terms with error control"""
        if not np.any(non_space_mask):
            return temperature, 1.0
        
        # Calculate maximum allowable temperature change per step for stability
        max_source_magnitude = np.max(np.abs(heat_source[non_space_mask])) if np.any(non_space_mask) else 0.0
        
        if max_source_magnitude == 0:
            return temperature, 1.0  # No sources, no integration needed
        
        # Error-based adaptive stepping instead of fixed temperature limits
        max_error_per_step = 10.0  # Maximum 10K error per full timestep (adjustable)
        
        # Handle case where there are no significant heat sources
        if max_source_magnitude < 1e-10:  # Essentially zero sources
            return temperature, 1.0  # No adaptation needed
        
        safe_dt = max_error_per_step / max_source_magnitude
        safe_dt = max(safe_dt, self.dt / 100.0)  # Don't go smaller than 1/100 of timestep
        
        # Number of sub-steps needed (with safety checks)
        if safe_dt <= 0 or not np.isfinite(safe_dt):
            num_substeps = 1  # Fallback to single step
        else:
            num_substeps = max(1, int(np.ceil(self.dt / safe_dt)))
            
        effective_dt = self.dt / num_substeps
        stability_factor = effective_dt / self.dt
        
        # Limit excessive subdivision for performance
        if num_substeps > 50:
            num_substeps = 50
            effective_dt = self.dt / num_substeps
            stability_factor = effective_dt / self.dt
        
        # Apply source terms with sub-stepping
        new_temp = temperature.copy()
        for step in range(num_substeps):
            source_change = heat_source * effective_dt
            
            # Apply source terms with safety checks
            valid_sources = non_space_mask & np.isfinite(source_change)
            if np.any(valid_sources):
                new_temp[valid_sources] += source_change[valid_sources]
            
            # Optional: Richardson extrapolation for higher accuracy (if needed)
            # This could be added later for better error control
        
        return new_temp, stability_factor
    
    def _implicit_diffusion_step(self, temperature: np.ndarray, thermal_diffusivity: np.ndarray, 
                               dt: float, non_space_mask: np.ndarray) -> np.ndarray:
        """Unconditionally stable implicit diffusion using backward Euler"""
        # Create working copy
        new_temp = temperature.copy()
        
        # Pre-compute diffusion kernel (cached)
        if not hasattr(self, '_diffusion_kernel'):
            # Laplacian kernel for implicit diffusion
            self._diffusion_kernel = np.array([
                [0, 1, 0],
                [1, -4, 1], 
                [0, 1, 0]
            ], dtype=np.float64)
        
        # Create "padded" temperature field where space cells take the temperature of nearest non-space neighbor
        temp_for_diffusion = temperature.copy()
        
        # For space cells, set temperature to nearby non-space cells to create insulating boundary
        space_mask = ~non_space_mask
        if np.any(space_mask):
            # Find space cells that are adjacent to non-space cells
            space_near_matter = space_mask & ndimage.binary_dilation(non_space_mask, structure=self._circular_kernel_3x3)
            
            if np.any(space_near_matter):
                # Use distance transform to find nearest non-space cell
                nearest_matter_indices = ndimage.distance_transform_edt(space_mask, return_indices=True)[1]
                
                # Set space cell temperatures to their nearest non-space neighbor temperature
                for space_y, space_x in zip(*np.where(space_near_matter)):
                    nearest_y = nearest_matter_indices[0, space_y, space_x]
                    nearest_x = nearest_matter_indices[1, space_y, space_x]
                    temp_for_diffusion[space_y, space_x] = temperature[nearest_y, nearest_x]
        
        # Simple explicit diffusion with small time step (stable when dt is small enough)
        # This is much simpler than full implicit solving but still stable with our adaptive time stepping
        temp_laplacian = ndimage.convolve(temp_for_diffusion, self._diffusion_kernel, mode='constant', cval=0.0)
        
        # Zero out Laplacian for space cells
        temp_laplacian[space_mask] = 0.0
        
        # Explicit diffusion update: T^(n+1) = T^n + dt*α*∇²T^n
        dx_squared = self.cell_size ** 2
        dt_seconds = dt * self.seconds_per_year
        
        # Apply diffusion only to non-space cells
        if np.any(non_space_mask):
            diffusion_change = thermal_diffusivity[non_space_mask] * dt_seconds * temp_laplacian[non_space_mask] / dx_squared
            new_temp[non_space_mask] += diffusion_change
        
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
        
        # FIXED: Proper atmospheric-only temperature averaging
        # Only include atmospheric cell temperatures in the averaging, completely exclude non-atmospheric cells        
        # Create temperature grid where ONLY atmospheric cells have their actual temperature
        # Non-atmospheric cells are marked as NaN to exclude them from averaging
        atmo_temp_for_sum = np.where(atmosphere_mask, temperature, 0.0)
        atmo_mask_for_count = atmosphere_mask.astype(np.float64)
        
        # Use simple 3x3 averaging kernel for neighbor mixing  
        mixing_kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],  # Center cell excluded (we want neighbors only)
            [1, 1, 1]
        ], dtype=np.float64)
        
        # Apply convolution to get sum of atmospheric neighbor temperatures and atmospheric neighbor counts
        neighbor_atmo_temp_sum = ndimage.convolve(atmo_temp_for_sum, mixing_kernel, mode='constant', cval=0.0)
        neighbor_atmo_count = ndimage.convolve(atmo_mask_for_count, mixing_kernel, mode='constant', cval=0.0)
        
        # Calculate average temperature of atmospheric neighbors only (completely excluding non-atmospheric cells)
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
        
        # Process all material types that have transitions
        transition_materials = list(MaterialType)  # Process all material types
        
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
        
        # Note: Rock melting and magma cooling are now handled by the general transition system above
        # This provides more flexibility and allows materials to have different melting points and cooling products
        
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
        """Get mask for solid materials (excluding space)"""
        return (self.material_types != MaterialType.SPACE)
    
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
    
    def _calculate_all_heat_sources(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate all heat source terms Q_total/(ρcp) in K/year"""
        # Calculate individual heat sources
        internal_source = self._calculate_internal_heating_source(non_space_mask)
        solar_source = self._calculate_solar_heating_source(non_space_mask)
        radiative_source = self._calculate_radiative_cooling_source(non_space_mask)
        atmospheric_source = self._calculate_atmospheric_heating_source(non_space_mask)
        
        # Combine all sources
        total_source = internal_source + solar_source + radiative_source + atmospheric_source
        
        return total_source
    
    def _calculate_internal_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate internal heat generation source term Q/(ρcp) in K/year"""
        # Initialize source term
        source_term = np.zeros_like(self.temperature)
        
        # Only apply to non-space solid materials with valid properties
        valid_heating = (
            non_space_mask & 
            (self.density > 0) & 
            (self.specific_heat > 0) &
            (self.material_types != MaterialType.SPACE)
        )
        
        if not np.any(valid_heating):
            return source_term
        
        # Get reusable arrays
        distances = self._get_distances_from_center()
        
        # Heat generation rate based on depth (more heat from radioactive decay in deep materials)
        planet_radius = self._get_planet_radius()
        relative_depth = np.clip(1.0 - distances / planet_radius, 0.0, 1.0)
        
        # Physics-based radioactive heating (W/m³)
        # Crustal rocks: ~1-3 µW/m³ from K, U, Th decay
        # Core: much higher due to pressure concentration and primordial heat
        # Increased to make internal heating visible in power visualization
        
        # More dramatic core heating profile for visibility
        crustal_heating_rate = 1e-6 * relative_depth**2  # W/m³ - crustal heating increases with depth
        
        # Much stronger core heating that's clearly visible
        # Use quadratic profile for stronger contrast: deeper = much more heating
        core_heating_rate = 1e-3 * relative_depth**3  # W/m³ - cubic profile for dramatic core heating (up to 1 mW/m³)
        
        # Calculate total power density grid
        total_power_density_grid = crustal_heating_rate + core_heating_rate  # W/m³
        
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
        
        # Surface cells are those adjacent to space or outer atmosphere (using circular kernel)
        surface_candidates = ndimage.binary_dilation(space_mask, structure=self._circular_kernel_3x3) & non_space_mask
        
        if not np.any(surface_candidates):
            return source_term
        
        # Calculate solar heating based on latitude (distance from equator)
        center_x, center_y = self.center_of_mass
        
        # For a circular planet, equator is the horizontal line through center
        # Distance from equator = |y - center_y|
        y_coords = np.arange(self.height).reshape(-1, 1)  # Shape: (height, 1)
        x_coords = np.arange(self.width).reshape(1, -1)   # Shape: (1, width) 
        distance_from_equator = np.abs(y_coords - center_y)
        planet_radius_cells = self._get_planet_radius()
        
        # Normalize distance from equator (0 at equator, 1 at poles)
        normalized_latitude = distance_from_equator / max(planet_radius_cells, 1.0)  # Avoid division by zero
        normalized_latitude = np.clip(normalized_latitude, 0.0, 1.0)
        
        # Solar intensity follows cosine law: I = I₀ * cos(latitude)
        latitude_radians = normalized_latitude * (np.pi / 2)  # 0 to π/2
        solar_intensity_factor = np.cos(latitude_radians)
        
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
        
        # Apply layer-by-layer atmospheric absorption (physically accurate)
        remaining_solar_flux = self._apply_layered_atmospheric_heating_source(non_space_mask, solar_intensity_factor, effective_solar_constant, source_term)
        
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
    
    def _apply_layered_atmospheric_heating_source(self, non_space_mask: np.ndarray, solar_intensity_factor: np.ndarray, 
                                               effective_solar_constant: float, source_term: np.ndarray) -> np.ndarray:
        """Apply physically accurate layer-by-layer atmospheric absorption as source terms, return remaining flux"""
        
        # Initialize remaining solar flux grid (starts with full intensity, modified by latitude)
        initial_solar_flux = effective_solar_constant * self.planetary_distance_factor * solar_intensity_factor
        remaining_flux = initial_solar_flux.copy()
        
        # Find atmospheric materials
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )
        
        if not np.any(atmosphere_mask):
            return remaining_flux  # No atmosphere, all solar radiation reaches surface
        
        space_mask = ~non_space_mask
        cell_area = self.cell_size ** 2
        cell_volume = self.cell_size ** 3
        
        # Start from outermost atmospheric layer (adjacent to space) and grow inward
        processed_atmosphere = np.zeros_like(atmosphere_mask, dtype=bool)  # Track processed cells
        
        # Find initial layer: atmosphere adjacent to space
        space_neighbors = ndimage.binary_dilation(space_mask, structure=self._circular_kernel_3x3)
        current_layer = atmosphere_mask & space_neighbors & ~processed_atmosphere
        
        layer_count = 0
        max_layers = 50  # Safety limit to prevent infinite loops
        
        # Process layers from outside to inside
        while np.any(current_layer) and layer_count < max_layers:
            layer_count += 1
            
            # Get coordinates of current atmospheric layer
            layer_coords = np.where(current_layer)
            layer_y, layer_x = layer_coords
            
            # Get incoming solar flux for this layer
            incoming_flux = remaining_flux[layer_y, layer_x]  # W/m²
            
            # Each atmospheric cell absorbs a fraction of the incoming flux
            absorbed_flux = incoming_flux * self.atmospheric_absorption_per_cell  # W/m²
            transmitted_flux = incoming_flux * (1.0 - self.atmospheric_absorption_per_cell)  # W/m²
            
            # Update remaining flux grid for next layer
            remaining_flux[layer_y, layer_x] = transmitted_flux
            
            # Convert absorbed flux to source term Q/(ρcp) in K/year
            layer_density = self.density[layer_coords]
            layer_specific_heat = self.specific_heat[layer_coords]
            
            # Apply heating to valid atmospheric cells in this layer
            valid_heating = (layer_density > 0) & (layer_specific_heat > 0) & (absorbed_flux > 0)
            if np.any(valid_heating):
                valid_idx = np.where(valid_heating)[0]
                valid_y = layer_y[valid_idx]
                valid_x = layer_x[valid_idx]
                
                # Convert surface flux (W/m²) to volumetric power density (W/m³)
                volumetric_power_density = absorbed_flux[valid_idx] / self.cell_size  # W/m³
                
                # Convert to temperature source term: Q/(ρcp) in K/s, then to K/year
                heating_source = (
                    volumetric_power_density / 
                    (layer_density[valid_idx] * layer_specific_heat[valid_idx])
                ) * self.seconds_per_year  # Convert K/s → K/year
                
                # Add to source term
                source_term[valid_y, valid_x] += heating_source
                
                # Track atmospheric solar power density for visualization
                self.power_density[valid_y, valid_x] += volumetric_power_density
                
                # Track total atmospheric heating for debugging
                total_atmo_power = np.sum(volumetric_power_density * cell_volume)  # W
                self.thermal_fluxes['atmospheric_heating'] += total_atmo_power
            
            # Mark this layer as processed
            processed_atmosphere |= current_layer
            
            # Find next inner layer: atmosphere adjacent to processed atmosphere (but not processed yet)
            processed_neighbors = ndimage.binary_dilation(processed_atmosphere, structure=self._circular_kernel_3x3)
            current_layer = atmosphere_mask & processed_neighbors & ~processed_atmosphere
            
            # If no more layers found, stop
            if not np.any(current_layer):
                break
        
        return remaining_flux
    
    def _calculate_radiative_cooling_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate radiative cooling source term Q/(ρcp) in K/year"""
        source_term = np.zeros_like(self.temperature)
        space_mask = ~non_space_mask
        
        # Find outer atmosphere - atmospheric materials connected to space (not interior air pockets)
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
            space_neighbors = ndimage.binary_dilation(space_mask, structure=self._circular_kernel_3x3)
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
        surface_candidates = ndimage.binary_dilation(outer_atmo_mask | space_mask, structure=self._circular_kernel_3x3)
        surface_solid_mask = surface_candidates & solid_mask
        
        # Combine outer atmosphere and surface solids for radiation
        radiative_mask = outer_atmo_mask | surface_solid_mask
        
        if not np.any(radiative_mask):
            return source_term
        
        # Get coordinates and properties of radiating cells
        radiative_coords = np.where(radiative_mask)
        T_radiating = self.temperature[radiative_coords]
        T_space = self.space_temperature
        
        # Stefan-Boltzmann cooling with greenhouse effect
        valid_cooling = (T_radiating > T_space) & (self.density[radiative_coords] > 0) & (self.specific_heat[radiative_coords] > 0)
        
        if np.any(valid_cooling):
            valid_idx = np.where(valid_cooling)[0]
            T_valid = T_radiating[valid_idx]
            density_valid = self.density[radiative_coords][valid_idx]
            specific_heat_valid = self.specific_heat[radiative_coords][valid_idx]
            
            # Get emissivity from material properties
            material_types_valid = self.material_types[radiative_coords][valid_idx]
            emissivity = np.array([self.material_db.get_properties(mat_type).emissivity 
                                 for mat_type in material_types_valid])
            
            # Calculate dynamic greenhouse effect based on water vapor content
            water_vapor_mask = (self.material_types == MaterialType.WATER_VAPOR)
            total_water_vapor_mass = np.sum(self.density[water_vapor_mask]) if np.any(water_vapor_mask) else 0.0
            
            # Scale greenhouse effect by water vapor content (logarithmic to prevent runaway)
            if total_water_vapor_mass > 0:
                vapor_factor = np.log1p(total_water_vapor_mass / self.greenhouse_vapor_scaling) / 10.0  # Dampened scaling
                greenhouse_factor = self.base_greenhouse_effect + (self.max_greenhouse_effect - self.base_greenhouse_effect) * np.tanh(vapor_factor)
            else:
                greenhouse_factor = self.base_greenhouse_effect
            
            # Stefan-Boltzmann calculation with atmospheric greenhouse effect (year-based units)
            stefan_boltzmann = self.stefan_boltzmann_geological  # Already in year-based units
            effective_stefan = stefan_boltzmann * (1.0 - greenhouse_factor)
            
            # Radiative cooling with greenhouse effect providing the main balance
            cooling_efficiency = self.radiative_cooling_efficiency
            
            # Newton Cooling Law approximation (much more stable than T^4)
            # Instead of σT^4, use h(T - T_ambient) where h is effective heat transfer coefficient
            cooling_mask = T_valid > T_space
            power_per_area = np.zeros_like(T_valid)
            
            if np.any(cooling_mask):
                T_cooling = T_valid[cooling_mask]
                emissivity_cooling = emissivity[cooling_mask]
                
                # Newton cooling law: Q = h(T - T_ambient)
                # Effective heat transfer coefficient based on emissivity and Stefan-Boltzmann constant
                # h ≈ 4σεT₀³ where T₀ is reference temperature (around 300K)
                T_reference = 300.0  # Reference temperature for linearization (K)
                stefan_boltzmann = self.stefan_boltzmann_geological
                effective_stefan = stefan_boltzmann * (1.0 - greenhouse_factor)
                
                # Linearized heat transfer coefficient (much more stable)
                h_effective = 4.0 * effective_stefan * emissivity_cooling * (T_reference ** 3)
                
                # Scale for geological timescales 
                h_geological = h_effective / 1000.0  # Conservative scaling
                
                # Newton cooling: P = h(T - T_space)
                temp_difference = T_cooling - T_space
                power_per_area[cooling_mask] = h_geological * cooling_efficiency * temp_difference
            
            # Convert surface power (J/(year⋅m²)) to volumetric power density (W/m³)
            # Radiative cooling is a surface phenomenon, spread over surface layer
            surface_layer_thickness = self.cell_size * self.surface_radiation_depth_fraction
            volumetric_power_density = power_per_area / (surface_layer_thickness * self.seconds_per_year)  # W/m³
            
            # Convert to temperature source term: Q/(ρcp) in K/s, then to K/year
            cooling_source = -(
                volumetric_power_density / 
                (density_valid * specific_heat_valid)
            ) * self.seconds_per_year  # Convert K/s → K/year (negative for cooling)
            
            # Apply cooling source term
            cooling_y, cooling_x = radiative_coords[0][valid_idx], radiative_coords[1][valid_idx]
            source_term[cooling_y, cooling_x] += cooling_source
            
            # Track cooling power density for visualization (negative = heat loss)
            self.power_density[cooling_y, cooling_x] -= volumetric_power_density
            
            # Track total radiative output for debugging
            total_radiative_power = np.sum(volumetric_power_density * (self.cell_size ** 3))  # W
            self.thermal_fluxes['radiative_output'] = total_radiative_power
        
        return source_term
    
    def _calculate_atmospheric_heating_source(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Calculate atmospheric heating source term Q/(ρcp) in K/year"""
        # Atmospheric heating is now handled directly in solar heating source calculation
        # This method remains as a placeholder for future atmospheric-specific heating
        return np.zeros_like(self.temperature)

    
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
        
        # Core physics (every step)
        self.temperature, stability_factor = self._heat_diffusion()
        
        # Apply stability factor to the time step for this step
        effective_dt = self.dt * stability_factor
        self._last_stability_factor = stability_factor
        
        # Update center of mass and pressure (every step - needed for thermal calculations)
        self._calculate_center_of_mass()
        self._calculate_planetary_pressure()
        
        # Apply metamorphic processes (every step - fundamental)
        metamorphic_changes = self._apply_metamorphism()
        
        # Run geological processes based on performance configuration
        step_count = int(self.time / self.dt)
        
        # Unified density stratification (physics-correct vectorized for speed + realism)
        density_stratification_changes = False
        if step_count % self.step_interval_differentiation == 0:
            density_stratification_changes = self._apply_density_stratification_local_vectorized()
        
        # Gravitational collapse (vectorized for maximum speed)
        collapse_changes = False
        if step_count % self.step_interval_collapse == 0:
            collapse_changes = self._apply_gravitational_collapse_vectorized()
        
        # Air migration (vectorized for maximum speed)
        fluid_changes = False
        if step_count % self.step_interval_fluid == 0:
            fluid_changes = self._apply_fluid_dynamics_vectorized()
        
        # Weathering (configurable)
        weathering_changes = False
        if self.enable_weathering and step_count % 10 == 0:  # Every 10th step when enabled
            weathering_changes = self._apply_weathering()
        
        # Update material properties if material types changed
        if metamorphic_changes or density_stratification_changes or collapse_changes or fluid_changes or weathering_changes:
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
            # Calculate step number (approximate from time and dt)
            step_number = int(self.time / self.dt) if self.dt > 0 else 0
            
            non_space_mask = (self.material_types != MaterialType.SPACE)
            temp_celsius = self.temperature - 273.15
            avg_planet_temp = np.mean(temp_celsius[non_space_mask]) if np.any(non_space_mask) else 0.0
            min_temp = np.min(temp_celsius[non_space_mask]) if np.any(non_space_mask) else 0.0
            max_temp = np.max(temp_celsius[non_space_mask]) if np.any(non_space_mask) else 0.0
            
            # Clean, formatted header with step number and key info
            self.logger.debug("=" * 65)
            self.logger.debug(f"STEP {step_number:4d} | Time: {self.time:6.1f}y | Planet Avg: {avg_planet_temp:6.1f}°C")
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
            atmosphere_mask = air_mask | water_vapor_mask
            water_total_mask = water_mask | water_vapor_mask | ice_mask
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
        
        # Create solid and non-solid masks
        solid_mask = self._get_solid_mask()
        non_solid_mask = ~solid_mask & (self.material_types != MaterialType.SPACE)
        
        if not np.any(solid_mask) or not np.any(non_solid_mask):
            return False
        
        # Vectorized fall steps
        kernel = self._collapse_kernel_4 if self.neighbor_count == 4 else self._collapse_kernel_8
        
        for fall_step in range(self.max_fall_steps):
            step_changes = False
            
            # Find solid materials adjacent to non-solid materials (fast morphological operation)
            solid_near_cavities = solid_mask & ndimage.binary_dilation(non_solid_mask, structure=kernel)
            
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
            
            # Get neighbor offsets using the helper function (shuffled to avoid grid artifacts)
            neighbor_offsets = self._get_neighbors(self.neighbor_count, shuffle=True)
            
            # Vectorized neighbor checking
            for dy, dx in neighbor_offsets:
                if moves_made >= max_moves_per_step:
                    break
                
                # Calculate neighbor positions
                neighbor_y = solid_coords[0] + dy
                neighbor_x = solid_coords[1] + dx
                
                # Bounds check
                in_bounds = ((neighbor_y >= 0) & (neighbor_y < self.height) & 
                           (neighbor_x >= 0) & (neighbor_x < self.width))
                
                if not np.any(in_bounds):
                    continue
                
                valid_idx = np.where(in_bounds)[0]
                valid_solid_y = solid_coords[0][valid_idx]
                valid_solid_x = solid_coords[1][valid_idx]
                valid_neighbor_y = neighbor_y[valid_idx]
                valid_neighbor_x = neighbor_x[valid_idx]
                
                # Check if neighbors are non-solid and closer to center
                neighbor_distances = distances[valid_neighbor_y, valid_neighbor_x]
                solid_distances_valid = solid_distances[valid_idx]
                
                is_non_solid = non_solid_mask[valid_neighbor_y, valid_neighbor_x]
                is_closer = neighbor_distances < solid_distances_valid
                can_collapse = is_non_solid & is_closer
                
                if not np.any(can_collapse):
                    continue
                
                # Apply fall probability and limit moves
                collapse_idx = np.where(can_collapse)[0]
                fall_prob = self.gravitational_fall_probability if fall_step == 0 else self.gravitational_fall_probability_later
                
                # Randomly sample collapses
                random_mask = np.random.random(len(collapse_idx)) < fall_prob
                final_collapse_idx = collapse_idx[random_mask]
                
                if len(final_collapse_idx) == 0:
                    continue
                
                # Limit number of moves for stability
                final_collapse_idx = final_collapse_idx[:max_moves_per_step - moves_made]
                
                # Get final coordinates
                collapse_solid_y = valid_solid_y[final_collapse_idx]
                collapse_solid_x = valid_solid_x[final_collapse_idx]
                collapse_neighbor_y = valid_neighbor_y[final_collapse_idx]
                collapse_neighbor_x = valid_neighbor_x[final_collapse_idx]
                
                # Vectorized material swapping
                if len(final_collapse_idx) > 0:
                    # Store materials
                    solid_materials = self.material_types[collapse_solid_y, collapse_solid_x].copy()
                    cavity_materials = self.material_types[collapse_neighbor_y, collapse_neighbor_x].copy()
                    
                    # Swap materials
                    self.material_types[collapse_neighbor_y, collapse_neighbor_x] = solid_materials
                    self.material_types[collapse_solid_y, collapse_solid_x] = cavity_materials
                    
                    # Transfer temperatures (vectorized)
                    solid_temps = self.temperature[collapse_solid_y, collapse_solid_x].copy()
                    self.temperature[collapse_neighbor_y, collapse_neighbor_x] = solid_temps
                    self.temperature[collapse_solid_y, collapse_solid_x] = (solid_temps + 273.15) / 2
                    
                    moves_made += len(final_collapse_idx)
                    step_changes = True
                    changes_made = True
            
            # Update masks for next iteration if changes were made
            if step_changes:
                solid_mask = self._get_solid_mask()
                non_solid_mask = ~solid_mask & (self.material_types != MaterialType.SPACE)
            else:
                break  # No changes, stop iterating
        
        return changes_made
    
    def _apply_fluid_dynamics_vectorized(self):
        """Fast vectorized fluid dynamics using morphological operations"""
        changes_made = False
        
        if not np.any((self.material_types == MaterialType.AIR)):
            return False
        
        # Air migration using vectorized operations
        air_mask = (self.material_types == MaterialType.AIR)
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
            
            # Vectorized neighbor checking for air migration (shuffled to avoid grid artifacts)
            neighbor_offsets = self._get_neighbors(self.neighbor_count, shuffle=True)
            
            # Find best migration targets for each air cell
            for dy, dx in neighbor_offsets:
                neighbor_y = air_coords[0] + dy
                neighbor_x = air_coords[1] + dx
                
                # Bounds check
                in_bounds = ((neighbor_y >= 0) & (neighbor_y < self.height) & 
                           (neighbor_x >= 0) & (neighbor_x < self.width))
                
                if not np.any(in_bounds):
                    continue
                
                valid_idx = np.where(in_bounds)[0]
                valid_air_y = air_coords[0][valid_idx]
                valid_air_x = air_coords[1][valid_idx]
                valid_neighbor_y = neighbor_y[valid_idx]
                valid_neighbor_x = neighbor_x[valid_idx]
                
                # Check migration conditions
                neighbor_materials = self.material_types[valid_neighbor_y, valid_neighbor_x]
                neighbor_distances = distances[valid_neighbor_y, valid_neighbor_x]
                air_distances_valid = air_distances[valid_idx]
                
                # Air wants to move toward surface (away from center)
                not_space = (neighbor_materials != MaterialType.SPACE)
                toward_surface = (neighbor_distances > air_distances_valid)
                
                # Check material properties for migration (vectorized where possible)
                can_migrate = np.zeros(len(valid_idx), dtype=bool)
                for i, mat in enumerate(neighbor_materials):
                    if not_space[i] and toward_surface[i]:
                        props = self.material_db.get_properties(mat)
                        can_migrate[i] = not props.is_solid or props.porosity > 0.1
                
                if not np.any(can_migrate):
                    continue
                
                # Apply migration probability
                migrate_idx = np.where(can_migrate)[0]
                migration_prob = self.fluid_migration_probability
                
                random_mask = np.random.random(len(migrate_idx)) < migration_prob
                final_migrate_idx = migrate_idx[random_mask]
                
                if len(final_migrate_idx) == 0:
                    continue
                
                # Limit migrations for stability
                max_migrations = min(len(final_migrate_idx), 50)
                final_migrate_idx = final_migrate_idx[:max_migrations]
                
                # Get final coordinates
                migrate_air_y = valid_air_y[final_migrate_idx]
                migrate_air_x = valid_air_x[final_migrate_idx]
                migrate_neighbor_y = valid_neighbor_y[final_migrate_idx]
                migrate_neighbor_x = valid_neighbor_x[final_migrate_idx]
                
                # Vectorized material swapping
                if len(final_migrate_idx) > 0:
                    air_materials = self.material_types[migrate_air_y, migrate_air_x].copy()
                    neighbor_materials_swap = self.material_types[migrate_neighbor_y, migrate_neighbor_x].copy()
                    
                    self.material_types[migrate_neighbor_y, migrate_neighbor_x] = air_materials
                    self.material_types[migrate_air_y, migrate_air_x] = neighbor_materials_swap
                    changes_made = True
                    
                    # Limit to prevent too many changes per timestep
                    break
        
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
        is_liquid = ((sample_materials == MaterialType.WATER) | 
                     (sample_materials == MaterialType.MAGMA))
        is_hot_solid = (sample_temps > 1200.0)  # Hot solids can flow
        
        mobile_mask = is_gas | is_liquid | is_hot_solid
        if not np.any(mobile_mask):
            return False
        
        # Get mobile cells coordinates
        mobile_indices = np.where(mobile_mask)[0]
        mobile_y = sample_y[mobile_indices]
        mobile_x = sample_x[mobile_indices]
                        
        # Original neighbor checking logic - randomized order for each cell
        neighbor_offsets = self._get_neighbors(num_neighbors=self.neighbor_count, shuffle=True)
        
        # Process each neighbor direction (preserves original local physics)
        for dy, dx in neighbor_offsets:
            # Calculate neighbor positions for all mobile cells
            neighbor_y = mobile_y + dy
            neighbor_x = mobile_x + dx
            
            # Bounds check - vectorized
            in_bounds = ((neighbor_y >= 0) & (neighbor_y < self.height) & 
                        (neighbor_x >= 0) & (neighbor_x < self.width))
            
            if not np.any(in_bounds):
                continue
            
            # Filter to valid neighbors
            valid_indices = np.where(in_bounds)[0]
            valid_mobile_y = mobile_y[valid_indices]
            valid_mobile_x = mobile_x[valid_indices]
            valid_neighbor_y = neighbor_y[valid_indices]
            valid_neighbor_x = neighbor_x[valid_indices]
            
            # Check if neighbors are non-space materials - vectorized
            neighbor_materials = self.material_types[valid_neighbor_y, valid_neighbor_x]
            is_non_space = (neighbor_materials != MaterialType.SPACE)
            
            if not np.any(is_non_space):
                continue
            
            # Filter to non-space neighbors
            ns_indices = np.where(is_non_space)[0]
            final_mobile_y = valid_mobile_y[ns_indices]
            final_mobile_x = valid_mobile_x[ns_indices]
            final_neighbor_y = valid_neighbor_y[ns_indices]
            final_neighbor_x = valid_neighbor_x[ns_indices]
            
            # Vectorized density and distance calculations (original physics)
            mobile_densities = effective_density_grid[final_mobile_y, final_mobile_x]
            neighbor_densities = effective_density_grid[final_neighbor_y, final_neighbor_x]
            mobile_distances = distances[final_mobile_y, final_mobile_x]
            neighbor_distances = distances[final_neighbor_y, final_neighbor_x]
            
            # Original buoyancy physics - vectorized
            # Case 1: Mobile cell closer to center but less dense (should rise)
            case1 = (mobile_distances < neighbor_distances) & (mobile_densities < neighbor_densities)
            # Case 2: Mobile cell farther from center but more dense (should sink)  
            case2 = (mobile_distances > neighbor_distances) & (mobile_densities > neighbor_densities)
            should_swap = case1 | case2
            
            # Original density difference threshold - vectorized
            min_densities = np.minimum(mobile_densities, neighbor_densities)
            max_densities = np.maximum(mobile_densities, neighbor_densities)
            density_ratios = np.divide(max_densities, min_densities, out=np.ones_like(max_densities), where=(min_densities > 0))
            significant_diff = density_ratios >= self.density_ratio_threshold
            
            # Final swap condition (original logic)
            final_swap = should_swap & significant_diff
            
            if not np.any(final_swap):
                continue
            
            # Apply random probability (preserves stochastic nature)
            swap_indices = np.where(final_swap)[0]
            random_mask = np.random.random(len(swap_indices)) < self.density_swap_probability
            final_swap_indices = swap_indices[random_mask]
            
            if len(final_swap_indices) == 0:
                continue
            
            # Get cells to swap
            swap_mobile_y = final_mobile_y[final_swap_indices]
            swap_mobile_x = final_mobile_x[final_swap_indices]
            swap_neighbor_y = final_neighbor_y[final_swap_indices]
            swap_neighbor_x = final_neighbor_x[final_swap_indices]
            
            # Vectorized swapping (preserves original swap mechanics)
            if len(final_swap_indices) > 0:
                # Swap materials
                mobile_materials = self.material_types[swap_mobile_y, swap_mobile_x].copy()
                neighbor_materials_swap = self.material_types[swap_neighbor_y, swap_neighbor_x].copy()
                self.material_types[swap_mobile_y, swap_mobile_x] = neighbor_materials_swap
                self.material_types[swap_neighbor_y, swap_neighbor_x] = mobile_materials
                
                # Swap temperatures (convective heat transfer)
                mobile_temps = self.temperature[swap_mobile_y, swap_mobile_x].copy()
                neighbor_temps = self.temperature[swap_neighbor_y, swap_neighbor_x].copy()
                self.temperature[swap_mobile_y, swap_mobile_x] = neighbor_temps
                self.temperature[swap_neighbor_y, swap_neighbor_x] = mobile_temps
                
                changes_made = True
                break  # Only one swap per cell per iteration (original behavior)
        
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
    
    def _solve_radiative_cooling_analytical(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Solve radiative cooling using analytical solution - unconditionally stable!"""
        working_temp = temperature.copy()
        
        # Find surface cells that radiate to space
        space_mask = ~non_space_mask
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )
        
        # Surface cells are those adjacent to space or outer atmosphere
        surface_candidates = ndimage.binary_dilation(space_mask, structure=self._circular_kernel_3x3) & non_space_mask
        
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