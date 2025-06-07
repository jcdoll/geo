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
    
    def __init__(self, width: int, height: int, cell_size: float = 50.0):
        """
        Initialize simulation grid
        
        Args:
            width: Grid width in cells
            height: Grid height in cells  
            cell_size: Size of each cell in meters
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size  # meters per cell
        
        # Core simulation grids
        self.material_types = np.full((height, width), MaterialType.GRANITE, dtype=object)
        self.temperature = np.zeros((height, width), dtype=np.float64)
        self.pressure = np.zeros((height, width), dtype=np.float64)
        self.pressure_offset = np.zeros((height, width), dtype=np.float64)  # User-applied pressure changes
        self.age = np.zeros((height, width), dtype=np.float64)
        
        # Derived properties (computed from material types)
        self.density = np.zeros((height, width), dtype=np.float64)
        self.thermal_conductivity = np.zeros((height, width), dtype=np.float64)
        self.specific_heat = np.zeros((height, width), dtype=np.float64)
        
        # Simulation parameters
        self.time = 0.0
        self.dt = 50.0  # years per time step (reduced for gradual ice melting)
        
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
        
        # History for time reversal
        self.max_history = 1000
        self.history = []
        self.history_step = 0
        
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
    
    def _setup_planetary_conditions(self):
        """Set up initial planetary conditions with emergent circular shape"""
        # Initialize everything as space
        self.material_types.fill(MaterialType.SPACE)
        self.temperature.fill(2.7)  # Space temperature (~3K cosmic background radiation)
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
                    core_temp = 1800.0 + 273.15  # K - hot core for activity
                    surface_temp = -30.0 + 273.15  # K - cold surface to preserve ice
                    
                    # Use exponential decay with gentler gradient for more extensive hot zone
                    decay_constant = 1.5  # Gentler decay = larger hot zone
                    temp_gradient = (core_temp - surface_temp) * np.exp(-decay_constant * relative_depth)
                    self.temperature[y, x] = surface_temp + temp_gradient
                    
                    # Add some randomness for more interesting geology (further reduced to preserve ice)
                    self.temperature[y, x] += np.random.normal(0, 5)
                    
                    # "Dirty iceball" formation - include ice pockets in outer regions
                    if relative_depth > 0.6:  # Outer 40% of planet - more ice
                        material_types = [
                            MaterialType.GRANITE, MaterialType.BASALT, MaterialType.SANDSTONE, 
                            MaterialType.LIMESTONE, MaterialType.SHALE, MaterialType.ICE, MaterialType.ICE, MaterialType.ICE  # More ice in outer regions
                        ]
                    elif relative_depth > 0.3:  # Middle regions - some ice
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
                    if self.temperature[y, x] > 1200 + 273.15:  # Hot enough to melt
                        self.material_types[y, x] = MaterialType.MAGMA
                        
                    # Add some surface variation (not perfectly circular)
                    if relative_depth > 0.85:  # Near surface
                        # Add some randomness to make surface irregular
                        noise = np.random.random() * 0.1
                        if relative_depth + noise > 1.0:
                            # Sometimes extend into space or create atmosphere
                            if np.random.random() < 0.3:  # 30% chance of atmosphere
                                self.material_types[y, x] = MaterialType.AIR
                                self.temperature[y, x] = surface_temp  # Already in K
        
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
    
    def _heat_diffusion(self) -> np.ndarray:
        """Heat diffusion using scipy convolution for efficient computation"""
        # Start with current temperature
        new_temp = self.temperature.copy()
        
        # Skip cells that are space
        non_space_mask = (self.material_types != MaterialType.SPACE)
        
        # Create thermal diffusivity array (vectorized)
        valid_thermal = (self.density > 0) & (self.specific_heat > 0) & (self.thermal_conductivity > 0)
        thermal_diffusivity = np.zeros_like(self.temperature)
        thermal_diffusivity[valid_thermal] = (
            self.thermal_conductivity[valid_thermal] / 
            (self.density[valid_thermal] * self.specific_heat[valid_thermal])
        )
        
        # Convolution kernel for 8-neighbor heat diffusion with distance weighting
        # Weight matrix: diagonals get 1/√2, cardinals get 1.0, center gets negative sum
        kernel = np.array([
            [0.707, 1.0, 0.707],
            [1.0,  -6.83, 1.0],   # -6.83 ≈ -(4*1.0 + 4*0.707)
            [0.707, 1.0, 0.707]
        ]) / 8.0  # Normalize
        
        # Apply diffusion using convolution (much faster than nested loops)
        temp_laplacian = ndimage.convolve(self.temperature, kernel, mode='constant', cval=0)
        
        # Calculate temperature changes (vectorized)
        scaling_factor = self.dt * self.seconds_per_year * 0.0001 / (self.cell_size ** 2)
        temp_change = thermal_diffusivity * temp_laplacian * scaling_factor
        
        # Apply changes only to non-space cells
        new_temp[non_space_mask] += temp_change[non_space_mask]
        
        # Stefan-Boltzmann cooling to space (vectorized for boundary cells)
        self._apply_radiative_cooling(new_temp, non_space_mask)
        
        # Ensure space stays at cosmic background temperature
        new_temp[~non_space_mask] = 2.7  # Kelvin
        
        # Safety check: prevent temperatures below absolute zero
        new_temp = np.maximum(new_temp, 0.1)
        
        return new_temp
    
    def _apply_radiative_cooling(self, new_temp: np.ndarray, non_space_mask: np.ndarray):
        """Apply Stefan-Boltzmann cooling to cells adjacent to space"""
        # Find cells adjacent to space using morphological operations (fast)
        space_mask = ~non_space_mask
        
        # Dilate space mask to find cells adjacent to space
        adjacent_to_space = ndimage.binary_dilation(space_mask, structure=np.ones((3,3))) & non_space_mask
        
        if not np.any(adjacent_to_space):
            return
        
        # Apply Stefan-Boltzmann cooling (vectorized)
        T_surface = self.temperature[adjacent_to_space]
        T_space = 2.7
        
        # Stefan-Boltzmann law (vectorized calculation)
        valid_cooling = (T_surface > T_space) & (self.density[adjacent_to_space] > 0) & (self.specific_heat[adjacent_to_space] > 0)
        
        if np.any(valid_cooling):
            # Get indices of valid cooling cells
            cooling_indices = np.where(adjacent_to_space)
            valid_subset = valid_cooling
            
            T_valid = T_surface[valid_subset]
            density_valid = self.density[cooling_indices][valid_subset]
            specific_heat_valid = self.specific_heat[cooling_indices][valid_subset]
            
            # Stefan-Boltzmann calculation (vectorized)
            stefan_boltzmann = 5.67e-8
            emissivity = 0.9
            power_per_area = emissivity * stefan_boltzmann * (T_valid**4 - T_space**4)
            
            # Temperature change calculation (vectorized)
            energy_loss = power_per_area * self.dt * self.seconds_per_year * (self.cell_size ** 2)
            mass = density_valid * (self.cell_size ** 3)
            temp_change = -energy_loss / (mass * specific_heat_valid)
            
            # Prevent cooling below space temperature
            max_cooling = T_valid - T_space
            temp_change = np.maximum(temp_change, -max_cooling)
            
            # Apply cooling to the new temperature array
            cooling_y, cooling_x = cooling_indices
            valid_y, valid_x = cooling_y[valid_subset], cooling_x[valid_subset]
            new_temp[valid_y, valid_x] += temp_change

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
        """Calculate pressure using gravitational model appropriate for a planet"""
        # Reset all pressures
        self.pressure.fill(0.0)
        
        # Get distance array for all cells
        distances = self._get_distances_from_center()
        
        for y in range(self.height):
            for x in range(self.width):
                if self.material_types[y, x] == MaterialType.SPACE:
                    self.pressure[y, x] = 0.0  # Vacuum
                    continue
                
                # Distance from center of mass
                distance = distances[y, x]
                distance_m = distance * self.cell_size  # Convert to meters
                
                if distance_m < self.cell_size:  # Avoid division by zero at center
                    distance_m = self.cell_size
                
                # Gravitational acceleration at this point
                g_local = self.gravity_constant * self.total_mass / (distance_m ** 2)
                
                # For atmospheric pressure, use hydrostatic equilibrium
                if self.material_types[y, x] == MaterialType.AIR:
                    # Simple atmospheric model - pressure decreases with height
                    # Height above surface (approximated)
                    surface_distance = self._get_planet_radius()
                    height_above_surface = max(0, distance - surface_distance) * self.cell_size
                    
                    # Exponential atmosphere
                    scale_height = 8400  # meters
                    surface_pressure = 0.1  # MPa (Earth sea level)
                    self.pressure[y, x] = surface_pressure * np.exp(-height_above_surface / scale_height)
                else:
                    # For solid matter, use simplified lithostatic pressure
                    # This is an approximation - should integrate along gravitational field lines
                    surface_distance = self._get_planet_radius()
                    depth = max(0, surface_distance - distance) * self.cell_size  # meters below surface
                    
                    if depth > 0:
                        # Simplified: assume average density and g
                        avg_density = 3000  # kg/m³ typical crustal density
                        avg_g = 9.81  # m/s² simplified
                        self.pressure[y, x] = avg_density * avg_g * depth / 1e6  # Convert to MPa
                    else:
                        self.pressure[y, x] = 0.1  # Surface pressure
        
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
    
    def _apply_gravitational_differentiation(self):
        """Apply gravitational sorting using efficient operations"""
        mobile_threshold = 700 + 273.15  # K - materials become mobile when hot/plastic
        
        # Use fixed geometric center instead of dynamic center of mass for more circular results
        fixed_center_x = self.width / 2.0
        fixed_center_y = self.height / 2.0
        distances = self._get_distances_from_center(fixed_center_x, fixed_center_y)
        mobile_mask = self._get_mobile_mask(mobile_threshold)
        
        # Count mobile cells for stats
        mobile_cells = np.sum(mobile_mask)
        if mobile_cells == 0:
            self._last_differentiation_stats = {'mobile_cells': 0, 'swaps': 0, 'changes_made': False, 'swap_rate': '0/0'}
            return False
        
        # Pre-calculate temperature factors for all cells (vectorized)
        temp_factors = self._calculate_temperature_factors(mobile_threshold)
        
        swap_count = 0
        changes_made = False
        
        # Process only mobile cells (much smaller loop) - randomize order to reduce grid artifacts
        mobile_coords = np.where(mobile_mask)
        cell_indices = np.arange(len(mobile_coords[0]))
        np.random.shuffle(cell_indices)  # Random processing order
        
        for i in cell_indices:
            y, x = mobile_coords[0][i], mobile_coords[1][i]
            current_density = self.density[y, x]
            current_distance = distances[y, x]
            current_temp_factor = temp_factors[y, x]
            
            # Get randomized neighbors to avoid grid artifacts
            neighbor_offsets = self._get_neighbors(8, shuffle=True)  # 8 neighbors, randomized
            
            # Check neighbors
            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and
                    mobile_mask[ny, nx]):  # Only consider mobile neighbors
                    
                    neighbor_density = self.density[ny, nx]
                    neighbor_distance = distances[ny, nx]
                    
                    # Check if swap is gravitationally favorable
                    should_swap = ((neighbor_distance < current_distance and neighbor_density < current_density) or
                                 (neighbor_distance > current_distance and neighbor_density > current_density))
                    
                    if should_swap:
                        # Calculate swap probability with circular bias
                        density_diff = abs(current_density - neighbor_density)
                        
                        # Add circular bias - prefer swaps that improve circularity
                        distance_diff = abs(neighbor_distance - current_distance)
                        circular_bias = min(2.0, distance_diff / 5.0)  # Bonus for large distance differences
                        
                        swap_probability = min(0.5, (density_diff / 1000.0) * current_temp_factor * circular_bias)
                        
                        if np.random.random() < swap_probability:
                            # Swap material types
                            self.material_types[y, x], self.material_types[ny, nx] = self.material_types[ny, nx], self.material_types[y, x]
                            changes_made = True
                            swap_count += 1
                            break  # Only one swap per cell per timestep
        
        # Store stats for debugging
        self._last_differentiation_stats = {
            'mobile_cells': int(mobile_cells),
            'swaps': swap_count,
            'changes_made': changes_made,
            'swap_rate': f"{swap_count}/{mobile_cells}" if mobile_cells > 0 else "0/0"
        }
        
        return changes_made
    
    def _apply_gravitational_collapse(self):
        """Apply gravitational collapse - materials fall into air cavities (fast, multi-step falling)"""
        changes_made = False
        
        # Get distance array for gravitational direction
        fixed_center_x = self.width / 2.0
        fixed_center_y = self.height / 2.0
        distances = self._get_distances_from_center(fixed_center_x, fixed_center_y)
        
        # Multiple passes to allow materials to fall multiple steps per frame
        max_fall_steps = 5  # Allow materials to fall up to 5 cells per timestep (more realistic)
        
        for fall_step in range(max_fall_steps):
            step_changes = False
            
            # Find all solid materials (materials that other materials can't fall through)
            solid_material_mask = self._get_solid_material_mask()
            solid_coords = np.where(solid_material_mask)
            
            if len(solid_coords[0]) == 0:
                break
            
            # Randomize processing order to avoid artifacts
            cell_indices = np.arange(len(solid_coords[0]))
            np.random.shuffle(cell_indices)
            
            for i in cell_indices:
                y, x = solid_coords[0][i], solid_coords[1][i]
                
                # Skip if this material was already moved in this step
                if self.material_types[y, x] == MaterialType.AIR:
                    continue
                    
                current_distance = distances[y, x]
                
                # Get randomized neighbors to avoid grid artifacts
                neighbor_offsets = self._get_neighbors(8, shuffle=True)  # 8 neighbors, randomized
                
                best_collapse_target = None
                best_distance = current_distance
                
                for dy, dx in neighbor_offsets:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.height and 0 <= nx < self.width):
                        neighbor_distance = distances[ny, nx]
                        neighbor_material = self.material_types[ny, nx]
                        
                        # Material can collapse into non-solid materials that are closer to center
                        neighbor_props = self.material_db.get_properties(neighbor_material)
                        if (not neighbor_props.is_solid and 
                            neighbor_distance < current_distance):
                            
                            # Prefer the cavity closest to center for maximum gravitational effect
                            if neighbor_distance < best_distance:
                                best_collapse_target = (ny, nx)
                                best_distance = neighbor_distance
                
                # High probability for immediate fall (gravity acts quickly on unsupported materials)
                fall_probability = 0.9 if fall_step == 0 else 0.7  # High initial fall, slightly lower for cascading
                
                if best_collapse_target and np.random.random() < fall_probability:
                    ny, nx = best_collapse_target
                    
                    # Material falls into non-solid material
                    displaced_material = self.material_types[ny, nx]
                    self.material_types[ny, nx] = self.material_types[y, x]
                    self.material_types[y, x] = displaced_material  # Displaced material moves up
                    
                    # Also transfer temperature (material carries its heat)
                    self.temperature[ny, nx] = self.temperature[y, x]
                    self.temperature[y, x] = (self.temperature[y, x] + 273.15) / 2  # Air is cooler
                    
                    step_changes = True
                    changes_made = True
            
            # If no changes in this step, stop the cascading process
            if not step_changes:
                break
        
        return changes_made
    
    def _apply_fluid_dynamics(self):
        """Apply air migration and cavity filling (water vaporization now handled by metamorphic system)"""
        changes_made = False
        
        # Note: Water vaporization is now handled by the metamorphic system in _apply_metamorphism()
        
        # Phase 1: Air migration toward surface (buoyancy through porous materials)
        air_coords = np.where(self.material_types == MaterialType.AIR)
        distances = self._get_distances_from_center()
        
        for i in range(len(air_coords[0])):
            y, x = air_coords[0][i], air_coords[1][i]
            current_distance = distances[y, x]
            
            # Check neighbors for migration opportunities (randomized order)
            neighbors = self._get_neighbors(8, shuffle=True)  # 8 neighbors, randomized
            
            best_neighbor = None
            best_distance = current_distance
            
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and
                    self.material_types[ny, nx] != MaterialType.SPACE):
                    
                    neighbor_distance = distances[ny, nx]
                    neighbor_material = self.material_types[ny, nx]
                    
                    # Air wants to move toward surface (larger distance from center)
                    # Can migrate through porous materials or displace other materials
                    neighbor_props = self.material_db.get_properties(neighbor_material)
                    can_migrate = (
                        not neighbor_props.is_solid or  # Move through/displace other materials
                        (neighbor_distance > current_distance and  # Moving toward surface
                         neighbor_props.porosity > 0.1)  # Through porous material
                    )
                    
                    if can_migrate and neighbor_distance > best_distance:
                        best_neighbor = (ny, nx)
                        best_distance = neighbor_distance
            
            # Migrate air toward surface with some probability
            if best_neighbor and np.random.random() < 0.3:  # 30% migration chance per timestep
                ny, nx = best_neighbor
                old_material = self.material_types[ny, nx]
                
                # Swap positions
                self.material_types[y, x] = old_material
                self.material_types[ny, nx] = MaterialType.AIR
                changes_made = True
        
        # Phase 2: Cavity filling - materials gradually flow into non-solid spaces
        # This simulates geological settling and compaction over time
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                current_props = self.material_db.get_properties(self.material_types[y, x])
                if not current_props.is_solid:
                    # Look for solid material neighbors that could "fall" into this cavity
                    neighbors = self._get_neighbors(4, shuffle=True)  # 4 cardinal neighbors, randomized
                    
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.height and 0 <= nx < self.width):
                            neighbor_material = self.material_types[ny, nx]
                            
                            # Solid materials can gradually fill cavities (very slow process)
                            neighbor_props = self.material_db.get_properties(neighbor_material)
                            if (neighbor_props.is_solid and neighbor_material != MaterialType.SPACE and
                                np.random.random() < 0.05):  # 5% chance - slow geological process
                                
                                # Material flows into cavity
                                displaced_material = self.material_types[y, x]
                                self.material_types[y, x] = neighbor_material
                                self.material_types[ny, nx] = displaced_material  # Displaced material moves to where material came from
                                changes_made = True
                                break
        
        return changes_made
    
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
            neighbors = self._get_neighbors(8, shuffle=False)
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
        
        # Much more conservative internal heating to prevent runaway magma production
        # Core heat: remnant from formation + radioactive decay (very deep only)
        core_heating = 0.05 * np.exp(4.0 * relative_depth)  # K per timestep - very concentrated in deep core
        # Crustal heat: radioactive decay (K, U, Th in granites) - minimal
        crustal_heating = 0.02 * relative_depth**2  # K per timestep - quadratic falloff, minimal at surface
        
        total_heating = (core_heating + crustal_heating) * self.dt / 2000.0  # Much reduced scaling
        
        # Apply heating only to solid materials
        heat_addition = np.where(solid_mask, total_heating, 0.0)
        self.temperature += heat_addition
    
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
            'time': self.time
        }
        self.history.append(state)
    
    def step_forward(self, dt: Optional[float] = None):
        """Advance simulation by one time step - optimized version"""
        if dt is not None:
            self.dt = dt
        
        # Save state for potential reversal
        self._save_state()
        
        # Use efficient heat diffusion
        self.temperature = self._heat_diffusion()
        
        # Add internal heat generation
        self._apply_internal_heat_generation()
        
        # Update center of mass and pressure
        self._calculate_center_of_mass()
        self._calculate_planetary_pressure()
        
        # Apply metamorphic processes
        metamorphic_changes = self._apply_metamorphism()
        
        # Apply gravitational differentiation
        differentiation_changes = self._apply_gravitational_differentiation()
        
        # Apply gravitational collapse
        collapse_changes = self._apply_gravitational_collapse()
        
        # Apply water vaporization and air migration
        fluid_changes = self._apply_fluid_dynamics()
        
        # Apply surface weathering processes
        weathering_changes = self._apply_weathering()
        
        # Update material properties if material types changed
        if metamorphic_changes or differentiation_changes or collapse_changes or fluid_changes or weathering_changes:
            self._update_material_properties()
        
        # Update age
        self.age += self.dt
        self.time += self.dt
        
        # Final safety check: ensure SPACE cells stay as SPACE and at cosmic background temp
        space_mask = (self.material_types == MaterialType.SPACE)
        self.temperature[space_mask] = 2.7  # Kelvin
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
    
    def get_visualization_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get data for visualization (material colors, temperature, pressure)"""
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
        
        return colors, self.temperature, self.pressure
    
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