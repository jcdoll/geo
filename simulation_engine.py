"""
Core simulation engine for 2D geological processes.
Handles heat transfer, pressure calculation, and rock state evolution.
"""

import numpy as np
from numba import jit
from typing import Tuple, Optional
try:
    from .materials import MaterialType, MaterialDatabase
except ImportError:
    from materials import MaterialType, MaterialDatabase
from scipy import ndimage
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

# JIT-compiled performance-critical functions
@jit(nopython=True)
def _jit_apply_convolution_diffusion(temperature, thermal_diffusivity, kernel, dt_factor):
    """JIT-compiled convolution for heat diffusion"""
    height, width = temperature.shape
    new_temp = temperature.copy()
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            if thermal_diffusivity[y, x] > 0:
                # Apply 3x3 convolution manually for speed
                laplacian = 0.0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        laplacian += temperature[y+ky, x+kx] * kernel[ky+1, kx+1]
                
                temp_change = thermal_diffusivity[y, x] * laplacian * dt_factor
                new_temp[y, x] += temp_change
    
    return new_temp

@jit(nopython=True)
def _jit_calculate_pressure(distances, rock_type_ids, planet_radius, cell_size, total_mass, gravity_constant):
    """JIT-compiled pressure calculation"""
    height, width = distances.shape
    pressure = np.zeros_like(distances)
    
    for y in range(height):
        for x in range(width):
            rock_id = rock_type_ids[y, x]
            if rock_id == 0:  # Space (assuming SPACE has ID 0)
                pressure[y, x] = 0.0
                continue
            
            distance = distances[y, x]
            distance_m = distance * cell_size
            
            if distance_m < cell_size:
                distance_m = cell_size
            
            # Simplified lithostatic pressure
            surface_distance = planet_radius
            depth = max(0.0, surface_distance - distance) * cell_size
            
            if depth > 0:
                avg_density = 3000.0  # kg/m³
                avg_g = 9.81  # m/s²
                pressure[y, x] = avg_density * avg_g * depth / 1e6  # MPa
            else:
                pressure[y, x] = 0.1  # Surface pressure
    
    return pressure

class GeologySimulation:
    """Main simulation engine for 2D geological processes"""
    
    def __init__(self, width: int, height: int, cell_size: float = 50.0, quality: int = 1):
        """
        Initialize simulation grid
        
        Args:
            width: Grid width in cells
            height: Grid height in cells  
            cell_size: Size of each cell in meters
            quality: Quality setting (1=high quality, 2=balanced, 3=fast)
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size  # meters per cell
        
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
        self.dt = 2.0  # Reduced from 10 years to 2 years for better stability
        
        # Unit conversion constants
        self.seconds_per_year = 365.25 * 24 * 3600
        self.stefan_boltzmann_geological = 5.67e-8 * self.seconds_per_year  # W/(m²⋅K⁴) → J/(year⋅m²⋅K⁴)
        self.gravity_constant = 6.67430e-11  # G in m³/(kg⋅s²)
        
        # Planetary parameters
        self.planet_radius_fraction = 0.8  # Fraction of grid width for initial planet
        self.planet_center = (width // 2, height // 2)
        self.total_mass = 0.0  # Will be calculated
        self.center_of_mass = (width / 2, height / 2)  # Will be calculated dynamically
        
        # Material database
        self.material_db = MaterialDatabase()
        
        # General physics settings (not performance-dependent)
        self.atmospheric_diffusivity_enhancement = 1.2  # Enhanced heat transfer in atmosphere
        self.interface_diffusivity_enhancement = 1.5    # Enhanced heat transfer at material interfaces
        self.surface_radiation_depth_fraction = 0.1     # Fraction of cell depth that participates in surface radiation
        self.radiative_cooling_efficiency = 0.2         # Cooling efficiency factor for Stefan-Boltzmann radiation
        
        # Temperature constants
        self.space_temperature = 2.7                    # Cosmic background temperature (K)
        self.reference_temperature = 273.15             # Reference temperature for thermal expansion (K)
        self.core_temperature = 400.0 + 273.15         # Initial planetary core temperature (K) - closer to equilibrium
        self.surface_temperature = 15.0 + 273.15       # Initial planetary surface temperature (K) - closer to equilibrium
        self.temperature_decay_constant = 2.0          # Temperature gradient decay factor - steeper gradient
        self.melting_temperature = 1200 + 273.15       # General melting temperature threshold (K)
        self.core_heating_depth_scale = 1.0            # Exponential scale factor for core heating vs depth
        
        # Pressure constants  
        self.surface_pressure = 0.1                     # Surface pressure (MPa)
        self.atmospheric_scale_height = 8400            # Atmospheric scale height (m)
        self.average_gravity = 9.81                     # Average gravitational acceleration (m/s²)
        self.average_solid_density = 3000               # Average solid rock density (kg/m³)
        self.average_fluid_density = 2000               # Average fluid density (kg/m³)
        
        # Solar and greenhouse constants (balanced for stable temperatures)
        self.solar_constant = 1361                      # Solar constant (W/m²)
        self.planetary_distance_factor = 0.01          # Distance factor for solar intensity - increased to balance cooling
        self.atmospheric_absorption = 0.25             # Atmospheric absorption fraction - realistic 25%
        self.base_greenhouse_effect = 0.5              # Base greenhouse effect fraction - increased to retain heat
        self.max_greenhouse_effect = 0.8               # Maximum greenhouse effect fraction
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
        self.max_time_series = 200  # Keep last 200 data points
        
        # Performance optimization caches
        self._material_props_cache = {}  # Cache for material property lookups
        self._neighbor_cache = {}        # Cache for neighbor offset calculations
        self._distance_cache = None      # Cache for distance calculations
        self._properties_dirty = True    # Flag to track when material properties need updating
        
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
        """Fast explicit heat diffusion with intelligent stability handling"""
        # Only process non-space cells
        non_space_mask = (self.material_types != MaterialType.SPACE)
        
        # Get thermal diffusivity for all cells (α = k / (ρ * cp))
        valid_thermal = (self.density > 0) & (self.specific_heat > 0) & (self.thermal_conductivity > 0)
        thermal_diffusivity = np.zeros_like(self.thermal_conductivity)
        thermal_diffusivity[valid_thermal] = (
            self.thermal_conductivity[valid_thermal] / 
            (self.density[valid_thermal] * self.specific_heat[valid_thermal])
        )
        
        # Enhanced diffusion at material interfaces (simplified)
        if hasattr(self, 'interface_diffusivity_enhancement'):
            neighbors = self._get_neighbors(4, shuffle=False)
            for dy, dx in neighbors:
                shifted_materials = np.roll(np.roll(self.material_types, dy, axis=0), dx, axis=1)
                interface_mask = (self.material_types != shifted_materials) & non_space_mask
                
                if np.any(interface_mask):
                    enhancement_mask = interface_mask & valid_thermal
                    if np.any(enhancement_mask):
                        thermal_diffusivity[enhancement_mask] *= self.interface_diffusivity_enhancement
        
        # Reset power density for this timestep
        self.power_density.fill(0.0)
        
        # Pre-compute diffusion kernel (cached) - more isotropic for circular approximation
        if not hasattr(self, '_diffusion_kernel'):
            # Improved circular approximation with normalized weights
            # Diagonal neighbors weighted by 1/√2 ≈ 0.707 for distance correction
            diagonal_weight = 0.707
            cardinal_weight = 1.0
            total_neighbor_weight = 4 * cardinal_weight + 4 * diagonal_weight
            center_weight = -total_neighbor_weight
            
            self._diffusion_kernel = np.array([
                [diagonal_weight, cardinal_weight, diagonal_weight],
                [cardinal_weight, center_weight, cardinal_weight], 
                [diagonal_weight, cardinal_weight, diagonal_weight]
            ]) / 8.0
        
        # Apply diffusion using fast convolution
        temp_laplacian = ndimage.convolve(self.temperature, self._diffusion_kernel, mode='constant', cval=0)
        
        # Time step and stability calculation
        dt_seconds = self.dt * self.seconds_per_year
        dx_squared = self.cell_size ** 2
        
        # Calculate diffusion coefficient and check stability
        diffusion_coefficient = thermal_diffusivity * dt_seconds / dx_squared
        max_coeff = np.max(diffusion_coefficient[non_space_mask]) if np.any(non_space_mask) else 0.0
        
        # Use more conservative stability limit to reduce oscillations
        max_stable_coeff = 0.1  # Reduced from 0.25 for better stability
        stability_factor = 1.0
        
        if max_coeff > max_stable_coeff:
            # Multiple sub-steps for stability (more efficient than matrix solve)
            stability_factor = max_stable_coeff / max_coeff
            n_substeps = max(1, int(np.ceil(1.0 / stability_factor)))
            sub_dt = dt_seconds / n_substeps
            
            new_temp = self.temperature.copy()
            for substep in range(n_substeps):
                # Sub-step diffusion
                sub_diffusion_coeff = thermal_diffusivity * sub_dt / dx_squared
                temp_change = sub_diffusion_coeff * temp_laplacian
                new_temp[non_space_mask] += temp_change[non_space_mask]
                
                # Update laplacian for next substep if needed
                if substep < n_substeps - 1:
                    temp_laplacian = ndimage.convolve(new_temp, self._diffusion_kernel, mode='constant', cval=0)
            
            # Use average effective time step for reporting
            stability_factor = 1.0 / n_substeps
        else:
            # Single step - fast path
            new_temp = self.temperature.copy()
            temp_change = diffusion_coefficient * temp_laplacian
            new_temp[non_space_mask] += temp_change[non_space_mask]
        
        # Store debugging info
        self._max_thermal_diffusivity = np.max(thermal_diffusivity[non_space_mask]) if np.any(non_space_mask) else 0.0
        
        # Calculate power density from final temperature change
        final_temp_change = new_temp - self.temperature
        mass_valid = valid_thermal & non_space_mask
        if np.any(mass_valid):
            power_from_diffusion = np.zeros_like(self.power_density)
            power_from_diffusion[mass_valid] = (
                self.density[mass_valid] * self.specific_heat[mass_valid] * 
                final_temp_change[mass_valid] / dt_seconds
            )
            self.power_density += power_from_diffusion
        
        # Apply other heat sources/sinks
        self._apply_solar_heating(new_temp, non_space_mask)
        self._apply_radiative_cooling(new_temp, non_space_mask)
        
        # Ensure space stays at cosmic background temperature
        new_temp[~non_space_mask] = self.space_temperature
        
        # Safety check: prevent temperatures below absolute zero
        new_temp = np.maximum(new_temp, 0.1)
        
        return new_temp, stability_factor
    
    def _apply_radiative_cooling(self, new_temp: np.ndarray, non_space_mask: np.ndarray):
        """Apply Stefan-Boltzmann cooling through transparent atmosphere model"""
        space_mask = ~non_space_mask
        
        # Find outer atmosphere - atmospheric materials connected to space (not interior air pockets)
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )
        
        # Find all atmosphere connected to space using connected components
        # This algorithm handles arbitrary atmosphere thickness in O(n) time
        
        # Label all connected atmosphere regions (using circular kernel to reduce artifacts)
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
            return
        
        # Apply Stefan-Boltzmann cooling
        T_radiating = self.temperature[radiative_mask]
        T_space = self.space_temperature
        
        # Stefan-Boltzmann law (vectorized calculation)
        valid_cooling = (T_radiating > T_space) & (self.density[radiative_mask] > 0) & (self.specific_heat[radiative_mask] > 0)
        
        if np.any(valid_cooling):
            # Get indices of valid cooling cells
            cooling_indices = np.where(radiative_mask)
            valid_subset = valid_cooling
            
            T_valid = T_radiating[valid_subset]
            density_valid = self.density[cooling_indices][valid_subset]
            specific_heat_valid = self.specific_heat[cooling_indices][valid_subset]
            
            # Get emissivity from material properties
            material_types_valid = self.material_types[cooling_indices][valid_subset]
            emissivity = np.array([self.material_db.get_properties(mat_type).emissivity 
                                 for mat_type in material_types_valid])
            
            # Stefan-Boltzmann calculation with atmospheric greenhouse effect (vectorized)
            stefan_boltzmann = 5.67e-8
            
            # Calculate dynamic greenhouse effect based on water vapor content
            # Find all atmospheric water vapor
            water_vapor_mask = (self.material_types == MaterialType.WATER_VAPOR)
            total_water_vapor_mass = np.sum(self.density[water_vapor_mask]) if np.any(water_vapor_mask) else 0.0
            
            # Scale greenhouse effect by water vapor content (logarithmic to prevent runaway)
            # Logarithmic scaling to prevent runaway greenhouse
            if total_water_vapor_mass > 0:
                vapor_factor = np.log1p(total_water_vapor_mass / self.greenhouse_vapor_scaling) / 10.0  # Dampened scaling
                greenhouse_factor = self.base_greenhouse_effect + (self.max_greenhouse_effect - self.base_greenhouse_effect) * np.tanh(vapor_factor)
            else:
                greenhouse_factor = self.base_greenhouse_effect
            
            effective_stefan = stefan_boltzmann * (1.0 - greenhouse_factor)
            
            # Moderate radiative cooling with greenhouse effect providing the main balance
            cooling_efficiency = self.radiative_cooling_efficiency
            power_per_area = emissivity * effective_stefan * cooling_efficiency * (T_valid**4 - T_space**4)
            
            # Temperature change calculation (vectorized)
            dt_seconds = self.dt * self.seconds_per_year
            energy_loss = power_per_area * dt_seconds * (self.cell_size ** 2)
            mass = density_valid * (self.cell_size ** 3)
            temp_change = -energy_loss / (mass * specific_heat_valid)
            
            # Calculate power density from radiative cooling (W/m³)
            # Radiative cooling is a surface phenomenon, not volumetric
            # Track as effective power density over a thin surface layer
            surface_layer_thickness = self.cell_size * self.surface_radiation_depth_fraction
            power_density_cooling = power_per_area / surface_layer_thickness
            
            # Add cooling power to power density tracking (negative = heat loss)
            cooling_y, cooling_x = cooling_indices
            valid_y, valid_x = cooling_y[valid_subset], cooling_x[valid_subset]
            self.power_density[valid_y, valid_x] -= power_density_cooling
            
            # Prevent cooling below space temperature
            max_cooling = T_valid - T_space
            temp_change = np.maximum(temp_change, -max_cooling)
            
            # Apply cooling to the new temperature array
            new_temp[valid_y, valid_x] += temp_change
    
    def _apply_solar_heating(self, new_temp: np.ndarray, non_space_mask: np.ndarray):
        """Apply solar heating with equatorial maximum and polar minimum"""
        # Find surface cells that can receive solar radiation
        space_mask = ~non_space_mask
        
        # Surface cells are those adjacent to space or outer atmosphere (using circular kernel)
        surface_candidates = ndimage.binary_dilation(space_mask, structure=self._circular_kernel_3x3) & non_space_mask
        
        if not np.any(surface_candidates):
            return
        
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
        # At equator (lat=0): cos(0) = 1.0 (maximum)
        # At poles (lat=π/2): cos(π/2) = 0.0 (minimum)
        latitude_radians = normalized_latitude * (np.pi / 2)  # 0 to π/2
        solar_intensity_factor = np.cos(latitude_radians)
        
        # Broadcast to full grid size
        solar_intensity_factor = np.broadcast_to(solar_intensity_factor, (self.height, self.width))
        
        # Solar constant and planet-specific factors  
        effective_solar_intensity = (self.solar_constant * self.planetary_distance_factor * 
                                    (1.0 - self.atmospheric_absorption))
        
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
        
        # Solar flux reaching surface (after atmospheric absorption and albedo)
        surface_solar_flux = (effective_solar_constant * self.planetary_distance_factor * 
                             (1.0 - self.atmospheric_absorption))  # W/m²
        
        # Solar flux absorbed by atmosphere  
        atmospheric_solar_flux = (effective_solar_constant * self.planetary_distance_factor * 
                                 self.atmospheric_absorption)  # W/m²
        
        # 1. Heat the atmosphere with absorbed solar energy
        atmosphere_mask = (
            (self.material_types == MaterialType.AIR) |
            (self.material_types == MaterialType.WATER_VAPOR)
        )
        atmo_coords = np.where(atmosphere_mask)
        
        if len(atmo_coords[0]) > 0:
            atmo_y, atmo_x = atmo_coords
            atmo_solar_intensity = solar_intensity_factor[atmo_y, atmo_x]
            
            # BUGFIX: Distribute atmospheric solar energy among ALL atmospheric cells
            # instead of applying full flux to each cell
            total_atmo_cells = len(atmo_coords[0])
            total_weighted_intensity = np.sum(atmo_solar_intensity)
            
            if total_weighted_intensity > 0:
                # Each cell gets a fraction of the total atmospheric energy based on its solar intensity
                atmo_energy_fraction = atmo_solar_intensity / total_weighted_intensity
                atmo_solar_heating = atmospheric_solar_flux * atmo_energy_fraction  # W/m² distributed
            else:
                atmo_solar_heating = np.zeros_like(atmo_solar_intensity)
            
            # Convert to temperature change for atmosphere
            dt_seconds = self.dt * self.seconds_per_year
            cell_area = self.cell_size ** 2
            atmo_energy_input = atmo_solar_heating * dt_seconds * cell_area  # J
            
            atmo_density = self.density[atmo_coords]
            atmo_specific_heat = self.specific_heat[atmo_coords]
            cell_volume = self.cell_size ** 3
            atmo_mass = atmo_density * cell_volume
            
            valid_atmo_heating = (atmo_density > 0) & (atmo_specific_heat > 0)
            if np.any(valid_atmo_heating):
                atmo_temp_increase = np.zeros_like(atmo_solar_heating)
                atmo_temp_increase[valid_atmo_heating] = (
                    atmo_energy_input[valid_atmo_heating] / 
                    (atmo_mass[valid_atmo_heating] * atmo_specific_heat[valid_atmo_heating])
                )
                
                # Apply atmospheric heating
                valid_atmo_y = atmo_y[valid_atmo_heating]
                valid_atmo_x = atmo_x[valid_atmo_heating]
                new_temp[valid_atmo_y, valid_atmo_x] += atmo_temp_increase[valid_atmo_heating]
                
                # Track atmospheric solar power density (positive = heat input)
                # Atmosphere absorbs solar radiation throughout its volume
                atmo_power_density = atmo_solar_heating[valid_atmo_heating] / self.cell_size  # W/m³
                self.power_density[valid_atmo_y, valid_atmo_x] += atmo_power_density
        
        # 2. Heat the surface with transmitted solar energy
        surface_coords = np.where(surface_candidates)
        if len(surface_coords[0]) == 0:
            return
        
        # Get solar intensity for surface cells
        surface_y, surface_x = surface_coords
        surface_solar_intensity = solar_intensity_factor[surface_y, surface_x]
        surface_flux = surface_solar_flux * surface_solar_intensity  # W/m²
        
        # Convert to temperature change
        dt_seconds = self.dt * self.seconds_per_year
        cell_area = self.cell_size ** 2  # m²
        energy_input = surface_flux * dt_seconds * cell_area  # J
        
        # Calculate mass and specific heat for surface cells
        surface_density = self.density[surface_coords]
        surface_specific_heat = self.specific_heat[surface_coords]
        cell_volume = self.cell_size ** 3  # m³
        cell_mass = surface_density * cell_volume  # kg
        
        # Temperature change: ΔT = Energy / (mass × specific_heat)
        valid_heating = (surface_density > 0) & (surface_specific_heat > 0)
        
        if np.any(valid_heating):
            temp_increase = np.zeros_like(surface_flux)
            temp_increase[valid_heating] = (
                energy_input[valid_heating] / 
                (cell_mass[valid_heating] * surface_specific_heat[valid_heating])
            )
            
            # Apply solar heating to surface cells
            heating_y, heating_x = surface_coords
            valid_y, valid_x = heating_y[valid_heating], heating_x[valid_heating]
            new_temp[valid_y, valid_x] += temp_increase[valid_heating]
            
            # Track solar power density (positive = heat input)
            # Solar heating is surface phenomenon - don't convert to volumetric for power tracking
            # Instead, track as effective power density over a thin surface layer
            surface_layer_thickness = self.cell_size * self.surface_radiation_depth_fraction
            solar_power_density = surface_flux / surface_layer_thickness  # W/m³ (concentrated in thin layer)
            self.power_density[valid_y, valid_x] += solar_power_density[valid_heating]

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
            self.total_mass = total_mass

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
    
    def _apply_internal_heat_generation(self):
        """Apply internal heat generation from radioactive decay and other sources"""
        # Get reusable arrays
        distances = self._get_distances_from_center()
        solid_mask = self._get_solid_mask()
        
        # Heat generation rate based on depth (more heat from radioactive decay in deep materials)
        planet_radius = self._get_planet_radius()
        relative_depth = np.clip(1.0 - distances / planet_radius, 0.0, 1.0)
        
        # Heat balance analysis:
        # Target: ~25% magma core radius at 1B years (reasonable for early Earth)
        # Stefan-Boltzmann cooling ∝ T⁴, so need balanced internal heating
        # Earth's heat flow ~0.06 W/m² average, ~0.1 W/m² from radioactivity
        
        # Physics-based radioactive heating (W/m³)
        # Crustal rocks: ~1-3 µW/m³ from K, U, Th decay
        # Core: much higher due to pressure concentration and primordial heat
        # Drastically reduced to prevent runaway heating
        crustal_heating_rate = 0.1e-6 * relative_depth**2  # W/m³ - reduced by 100x
        core_heating_rate = 2e-6 * np.exp(self.core_heating_depth_scale * relative_depth)  # W/m³ - reduced by 100x and very gentle exponential

        # Convert power density to temperature change: ΔT = (Power × dt) / (ρ × cp × volume)
        # Only apply to solid materials (they have meaningful density and specific_heat)
        cell_volume = self.cell_size ** 3  # m³ per cell
        dt_seconds = self.dt * self.seconds_per_year

        # Calculate temperature increase from radioactive heating
        temp_increase = np.zeros_like(self.temperature)
        valid_heating = solid_mask & (self.density > 0) & (self.specific_heat > 0)

        if np.any(valid_heating):
            total_power_density = (crustal_heating_rate + core_heating_rate)[valid_heating]  # W/m³
            mass_per_cell = (self.density * cell_volume)[valid_heating]  # kg
            specific_heat_values = self.specific_heat[valid_heating]  # J/(kg·K)
            
            # ΔT = (Power × time) / (mass × specific_heat)
            energy_per_cell = total_power_density * cell_volume * dt_seconds  # J
            temp_increase[valid_heating] = energy_per_cell / (mass_per_cell * specific_heat_values)  # K

        # Apply heating
        self.temperature += temp_increase

        # Add radioactive heating to power density tracking (positive = heat generation)
        self.power_density[valid_heating] += total_power_density
    
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
        
        # Save state for potential reversal
        self._save_state()
        
        # Core physics (every step)
        self.temperature, stability_factor = self._heat_diffusion()
        
        # Apply stability factor to the time step for this step
        effective_dt = self.dt * stability_factor
        self._last_stability_factor = stability_factor
        self._apply_internal_heat_generation()
        
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
            'effective_dt': getattr(self, '_last_stability_factor', 1.0) * self.dt,
            'stability_factor': getattr(self, '_last_stability_factor', 1.0),
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