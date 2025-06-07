"""
Core simulation engine for 2D geological processes.
Handles heat transfer, pressure calculation, and rock state evolution.
"""

import numpy as np
from numba import jit
from typing import Tuple, Optional
from rock_types import RockType, RockDatabase

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
        self.rock_types = np.full((height, width), RockType.GRANITE, dtype=object)
        self.temperature = np.zeros((height, width), dtype=np.float64)
        self.pressure = np.zeros((height, width), dtype=np.float64)
        self.pressure_offset = np.zeros((height, width), dtype=np.float64)  # User-applied pressure changes
        self.age = np.zeros((height, width), dtype=np.float64)
        
        # Derived properties (computed from rock types)
        self.density = np.zeros((height, width), dtype=np.float64)
        self.thermal_conductivity = np.zeros((height, width), dtype=np.float64)
        self.specific_heat = np.zeros((height, width), dtype=np.float64)
        
        # Simulation parameters
        self.time = 0.0
        self.dt = 1000.0  # years per time step
        
        # Unit conversion constants
        self.seconds_per_year = 365.25 * 24 * 3600
        self.stefan_boltzmann_geological = 5.67e-8 * self.seconds_per_year  # W/(m²⋅K⁴) → J/(year⋅m²⋅K⁴)
        self.gravity_constant = 6.67430e-11  # G in m³/(kg⋅s²)
        
        # Planetary parameters
        self.planet_radius_fraction = 0.8  # Fraction of grid width for initial planet
        self.planet_center = (width // 2, height // 2)
        self.total_mass = 0.0  # Will be calculated
        self.center_of_mass = (width / 2, height / 2)  # Will be calculated dynamically
        
        # Rock database
        self.rock_db = RockDatabase()
        
        # History for time reversal
        self.max_history = 1000
        self.history = []
        self.history_step = 0
        
        # Initialize as an emergent planet
        self._setup_planetary_conditions()
        self._update_material_properties()
        self._calculate_center_of_mass()
    
    def _setup_planetary_conditions(self):
        """Set up initial planetary conditions with emergent circular shape"""
        # Initialize everything as space
        self.rock_types.fill(RockType.SPACE)
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
                    # Within planet - set rock type and temperature based on distance from center
                    relative_depth = distance / planet_radius  # 0 at center, 1 at surface
                    
                    # Temperature: hotter planet for more diffusion and segregation (in Kelvin)
                    core_temp = 1800.0 + 273.15  # K - much hotter core for more activity
                    surface_temp = 25.0 + 273.15  # K - warmer surface
                    
                    # Use exponential decay with gentler gradient for more extensive hot zone
                    decay_constant = 1.5  # Gentler decay = larger hot zone
                    temp_gradient = (core_temp - surface_temp) * np.exp(-decay_constant * relative_depth)
                    self.temperature[y, x] = surface_temp + temp_gradient
                    
                    # Add some randomness for more interesting geology (reduced to prevent surface melting)
                    self.temperature[y, x] += np.random.normal(0, 15)
                    
                    # "Dirty iceball" formation - include ice pockets in outer regions
                    if relative_depth > 0.6:  # Outer 40% of planet - more ice
                        rock_types = [
                            RockType.GRANITE, RockType.BASALT, RockType.SANDSTONE, 
                            RockType.LIMESTONE, RockType.SHALE, RockType.ICE, RockType.ICE, RockType.ICE  # More ice in outer regions
                        ]
                    elif relative_depth > 0.3:  # Middle regions - some ice
                        rock_types = [
                            RockType.GRANITE, RockType.BASALT, RockType.GNEISS, 
                            RockType.SANDSTONE, RockType.LIMESTONE, RockType.SHALE,
                            RockType.MARBLE, RockType.QUARTZITE, RockType.ICE  # Some ice
                        ]
                    else:  # Inner core - mostly rocks, minimal ice
                        rock_types = [
                            RockType.GRANITE, RockType.BASALT, RockType.GNEISS, 
                            RockType.SANDSTONE, RockType.LIMESTONE, RockType.SHALE,
                            RockType.MARBLE, RockType.QUARTZITE, RockType.SCHIST,
                            RockType.SLATE, RockType.ANDESITE
                        ]
                    
                    self.rock_types[y, x] = np.random.choice(rock_types)
                    
                    # Convert hot material to magma based on temperature
                    if self.temperature[y, x] > 1200 + 273.15:  # Hot enough to melt
                        self.rock_types[y, x] = RockType.MAGMA
                        
                    # Add some surface variation (not perfectly circular)
                    if relative_depth > 0.85:  # Near surface
                        # Add some randomness to make surface irregular
                        noise = np.random.random() * 0.1
                        if relative_depth + noise > 1.0:
                            # Sometimes extend into space or create atmosphere
                            if np.random.random() < 0.3:  # 30% chance of atmosphere
                                self.rock_types[y, x] = RockType.AIR
                                self.temperature[y, x] = surface_temp  # Already in K
        
        # Calculate initial pressure using gravitational model
        self._calculate_planetary_pressure()
    
    def _update_material_properties(self):
        """Update material property grids based on current rock types"""
        for y in range(self.height):
            for x in range(self.width):
                rock_type = self.rock_types[y, x]
                props = self.rock_db.get_properties(rock_type)
                
                self.density[y, x] = props.density
                self.thermal_conductivity[y, x] = props.thermal_conductivity
                self.specific_heat[y, x] = props.specific_heat
    
    def _calculate_center_of_mass(self):
        """Calculate the center of mass of all matter in the simulation"""
        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        for y in range(self.height):
            for x in range(self.width):
                if self.rock_types[y, x] != RockType.SPACE:
                    # Mass of this cell
                    cell_volume = self.cell_size ** 3  # m³ (treating as 3D cube)
                    cell_mass = self.density[y, x] * cell_volume  # kg
                    
                    total_mass += cell_mass
                    weighted_x += cell_mass * x
                    weighted_y += cell_mass * y
        
        if total_mass > 0:
            self.center_of_mass = (weighted_x / total_mass, weighted_y / total_mass)
            self.total_mass = total_mass
        else:
            # Fallback if no matter exists
            self.center_of_mass = (self.width / 2, self.height / 2)
            self.total_mass = 0.0
    
    def _calculate_planetary_pressure(self):
        """Calculate pressure using gravitational model appropriate for a planet"""
        # Reset all pressures
        self.pressure.fill(0.0)
        
        # Get distance array for all cells
        distances = self._get_distances_from_center()
        
        for y in range(self.height):
            for x in range(self.width):
                if self.rock_types[y, x] == RockType.SPACE:
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
                if self.rock_types[y, x] == RockType.AIR:
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
    
    def _heat_diffusion_step_planetary(self) -> np.ndarray:
        """Fast, stable heat diffusion with simplified planetary boundary conditions"""
        new_temp = self.temperature.copy()
        
        # Simple diffusion with cooling to space - much faster and stable
        for y in range(self.height):
            for x in range(self.width):
                if self.rock_types[y, x] == RockType.SPACE:
                    continue
                
                # Conductive heat transfer with neighbors
                heat_change = 0.0
                neighbors = [(y, x+1), (y, x-1), (y+1, x), (y-1, x)]
                
                for ny, nx in neighbors:
                    if self.rock_types[ny, nx] == RockType.SPACE:
                        # Stefan-Boltzmann radiative cooling to space (proper physics)
                        T_surface = self.temperature[y, x]  # Already in Kelvin
                        T_space = 2.7  # Cosmic background radiation
                        if T_surface > T_space and self.density[y, x] > 0 and self.specific_heat[y, x] > 0:
                            # Stefan-Boltzmann: power ∝ T⁴ - T_space⁴
                            stefan_boltzmann = 5.67e-8  # W/(m²⋅K⁴)
                            emissivity = 0.9
                            power_per_area = emissivity * stefan_boltzmann * (T_surface**4 - T_space**4)  # W/m²
                            
                            # Convert to temperature change using proper thermodynamics
                            # Energy loss = power × time × surface_area
                            # Temperature change = energy_loss / (mass × specific_heat)
                            # mass = density × volume = density × cell_size³
                            # surface_area = cell_size² (for one face exposed to space)
                            
                            energy_loss_per_timestep = power_per_area * self.dt * self.seconds_per_year * (self.cell_size ** 2)  # Joules
                            mass = self.density[y, x] * (self.cell_size ** 3)  # kg
                            temp_change = -energy_loss_per_timestep / (mass * self.specific_heat[y, x])  # K
                            
                            # Prevent cooling below space temperature (can't be colder than space)
                            max_cooling = T_surface - T_space
                            temp_change = max(temp_change, -max_cooling)
                            
                            heat_change += temp_change
                    else:
                        # Simple conductive diffusion
                        if (self.density[y, x] > 0 and self.specific_heat[y, x] > 0 and
                            self.thermal_conductivity[y, x] > 0):
                            
                            temp_diff = self.temperature[ny, nx] - self.temperature[y, x]
                            
                            # Simplified diffusion rate (dimensionally correct)
                            alpha = (self.thermal_conductivity[y, x] / 
                                   (self.density[y, x] * self.specific_heat[y, x]))
                            
                            # Scale to geological time with stability factor
                            diffusion_rate = alpha * temp_diff / (self.cell_size ** 2)
                            temp_change = diffusion_rate * self.dt * self.seconds_per_year * 0.001  # Much smaller for gradual changes
                            
                            heat_change += temp_change
                
                new_temp[y, x] += heat_change
        
        # Ensure space stays at cosmic background temperature
        new_temp[self.rock_types == RockType.SPACE] = 2.7  # Kelvin
        
        # Safety check: prevent any temperature from going below absolute zero
        new_temp = np.maximum(new_temp, 0.1)  # Minimum 0.1K to avoid numerical issues
        
        return new_temp
    
    def _apply_metamorphism(self):
        """Apply metamorphic transitions based on P-T conditions"""
        changes_made = False
        
        for y in range(self.height):
            for x in range(self.width):
                current_rock = self.rock_types[y, x]
                
                # Skip SPACE cells - they should never change
                if current_rock == RockType.SPACE:
                    continue
                    
                temp = self.temperature[y, x]
                pressure = self.pressure[y, x]
                
                # Check for ice melting to water first
                if current_rock == RockType.ICE and temp >= 0 + 273.15:  # 0°C melting point
                    self.rock_types[y, x] = RockType.WATER
                    changes_made = True
                    continue
                
                # Check for rock melting to magma
                if (self.rock_db.should_melt(current_rock, temp) and 
                    current_rock not in [RockType.MAGMA, RockType.ICE, RockType.WATER, RockType.AIR, RockType.SPACE]):
                    self.rock_types[y, x] = RockType.MAGMA
                    changes_made = True
                    continue
                
                # Check for metamorphic transitions
                new_rock = self.rock_db.get_metamorphic_product(current_rock, temp, pressure)
                if new_rock and new_rock != current_rock:
                    self.rock_types[y, x] = new_rock
                    changes_made = True
                
                # Handle magma cooling
                if current_rock == RockType.MAGMA and temp < 800 + 273.15:  # 800°C in Kelvin
                    # Determine composition based on location (simplified)
                    composition = "felsic" if y < self.height * 0.5 else "mafic"
                    new_rock = self.rock_db.get_cooling_product(temp, pressure, composition)
                    self.rock_types[y, x] = new_rock
                    changes_made = True
        
        return changes_made
    
    def _get_distances_from_center(self, center_x: float = None, center_y: float = None) -> np.ndarray:
        """Get distance array from center point (or center of mass if not specified)"""
        if center_x is None or center_y is None:
            center_x, center_y = self.center_of_mass
        
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        return np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    def _get_mobile_mask(self, temperature_threshold: float = None) -> np.ndarray:
        """Get mask for cells that are mobile (hot and not space)"""
        if temperature_threshold is None:
            temperature_threshold = 700 + 273.15  # Default mobile threshold
        
        return ((self.temperature >= temperature_threshold) & 
                (self.rock_types != RockType.SPACE))
    
    def _get_solid_mask(self) -> np.ndarray:
        """Get mask for cells that contain solid matter (not space)"""
        return (self.rock_types != RockType.SPACE)
    
    def _get_planet_radius(self) -> float:
        """Get planet radius in cells"""
        return min(self.width, self.height) * self.planet_radius_fraction / 2
    
    def _get_randomized_neighbors(self) -> list:
        """Get 8-neighbor offsets in randomized order to avoid grid artifacts"""
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        np.random.shuffle(neighbors)
        return neighbors
    
    def _calculate_temperature_factors(self, threshold: float, scale: float = 200.0, max_factor: float = 10.0) -> np.ndarray:
        """Calculate exponential temperature factors for mobility"""
        temp_excess = np.maximum(0, self.temperature - threshold)
        return np.minimum(max_factor, np.exp(temp_excess / scale))
    
    def _apply_gravitational_differentiation(self):
        """Apply gravitational sorting using vectorized operations - much faster"""
        mobile_threshold = 700 + 273.15  # K - rocks become mobile when hot/plastic
        
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
            neighbor_offsets = self._get_randomized_neighbors()
            
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
                            # Swap rock types
                            self.rock_types[y, x], self.rock_types[ny, nx] = self.rock_types[ny, nx], self.rock_types[y, x]
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
        """Apply gravitational collapse - rocks fall into air cavities (no structural support)"""
        changes_made = False
        
        # Get distance array for gravitational direction
        fixed_center_x = self.width / 2.0
        fixed_center_y = self.height / 2.0
        distances = self._get_distances_from_center(fixed_center_x, fixed_center_y)
        
        # Find all solid rocks (not air, water, space, or magma which flows differently)
        solid_rock_mask = ~np.isin(self.rock_types, [RockType.AIR, RockType.WATER, RockType.SPACE, RockType.MAGMA])
        solid_coords = np.where(solid_rock_mask)
        
        # Randomize processing order to avoid artifacts
        cell_indices = np.arange(len(solid_coords[0]))
        np.random.shuffle(cell_indices)
        
        for i in cell_indices:
            y, x = solid_coords[0][i], solid_coords[1][i]
            current_distance = distances[y, x]
            
            # Get randomized neighbors to avoid grid artifacts
            neighbor_offsets = self._get_randomized_neighbors()
            
            best_collapse_target = None
            best_distance = current_distance
            
            for dy, dx in neighbor_offsets:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width):
                    neighbor_distance = distances[ny, nx]
                    neighbor_rock = self.rock_types[ny, nx]
                    
                    # Rock can collapse into air cavities that are closer to center
                    if (neighbor_rock == RockType.AIR and 
                        neighbor_distance < current_distance):
                        
                        # Prefer the cavity closest to center for maximum gravitational effect
                        if neighbor_distance < best_distance:
                            best_collapse_target = (ny, nx)
                            best_distance = neighbor_distance
            
            # Collapse into air cavity with high probability (gravity is strong)
            if best_collapse_target and np.random.random() < 0.7:  # 70% collapse chance - gravity is relentless
                ny, nx = best_collapse_target
                
                # Rock falls into air cavity
                self.rock_types[ny, nx] = self.rock_types[y, x]
                self.rock_types[y, x] = RockType.AIR  # Leaves air behind
                
                # Also transfer temperature (rock carries its heat)
                self.temperature[ny, nx] = self.temperature[y, x]
                self.temperature[y, x] = (self.temperature[y, x] + 273.15) / 2  # Air is cooler
                
                changes_made = True
        
        return changes_made
    
    def _apply_fluid_dynamics(self):
        """Apply water vaporization, air migration, and cavity filling"""
        changes_made = False
        vaporization_temp = 100 + 273.15  # K - water boiling point
        
        # Phase 1: Water vaporization
        water_mask = (self.rock_types == RockType.WATER)
        hot_water_mask = water_mask & (self.temperature >= vaporization_temp)
        
        if np.any(hot_water_mask):
            # Convert hot water to air (steam)
            self.rock_types[hot_water_mask] = RockType.AIR
            changes_made = True
        
        # Phase 2: Air migration toward surface (buoyancy through porous materials)
        air_coords = np.where(self.rock_types == RockType.AIR)
        distances = self._get_distances_from_center()
        
        for i in range(len(air_coords[0])):
            y, x = air_coords[0][i], air_coords[1][i]
            current_distance = distances[y, x]
            
            # Check neighbors for migration opportunities
            neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1),  # cardinal directions
                        (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)]  # diagonals
            
            best_neighbor = None
            best_distance = current_distance
            
            for ny, nx in neighbors:
                if (0 <= ny < self.height and 0 <= nx < self.width and
                    self.rock_types[ny, nx] != RockType.SPACE):
                    
                    neighbor_distance = distances[ny, nx]
                    neighbor_rock = self.rock_types[ny, nx]
                    
                    # Air wants to move toward surface (larger distance from center)
                    # Can migrate through porous materials or displace other fluids
                    can_migrate = (
                        neighbor_rock == RockType.WATER or  # Displace water
                        neighbor_rock == RockType.AIR or    # Move through air
                        (neighbor_distance > current_distance and  # Moving toward surface
                         self._get_porosity(neighbor_rock) > 0.1)  # Through porous rock
                    )
                    
                    if can_migrate and neighbor_distance > best_distance:
                        best_neighbor = (ny, nx)
                        best_distance = neighbor_distance
            
            # Migrate air toward surface with some probability
            if best_neighbor and np.random.random() < 0.3:  # 30% migration chance per timestep
                ny, nx = best_neighbor
                old_rock = self.rock_types[ny, nx]
                
                # Swap positions
                self.rock_types[y, x] = old_rock
                self.rock_types[ny, nx] = RockType.AIR
                changes_made = True
        
        # Phase 3: Cavity filling - rocks gradually flow into air-filled spaces
        # This simulates geological settling and compaction over time
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.rock_types[y, x] == RockType.AIR:
                    # Look for solid rock neighbors that could "fall" into this cavity
                    neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                    
                    for ny, nx in neighbors:
                        if (0 <= ny < self.height and 0 <= nx < self.width):
                            neighbor_rock = self.rock_types[ny, nx]
                            
                            # Solid rocks can gradually fill cavities (very slow process)
                            if (neighbor_rock not in [RockType.SPACE, RockType.AIR, RockType.WATER] and
                                np.random.random() < 0.05):  # 5% chance - slow geological process
                                
                                # Rock flows into cavity
                                self.rock_types[y, x] = neighbor_rock
                                self.rock_types[ny, nx] = RockType.AIR  # Leaves air behind
                                changes_made = True
                                break
        
        return changes_made
    
    def _get_porosity(self, rock_type):
        """Get porosity of a rock type for fluid migration calculations"""
        if rock_type == RockType.SPACE:
            return 0.0
        props = self.rock_db.get_properties(rock_type)
        return props.porosity
    
    def _apply_internal_heat_generation(self):
        """Apply internal heat generation from radioactive decay and other sources"""
        # Get reusable arrays
        distances = self._get_distances_from_center()
        solid_mask = self._get_solid_mask()
        
        # Heat generation rate based on depth (more heat from radioactive decay in deep rocks)
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
            'rock_types': self.rock_types.copy(),
            'temperature': self.temperature.copy(),
            'pressure': self.pressure.copy(),
            'pressure_offset': self.pressure_offset.copy(),
            'age': self.age.copy(),
            'time': self.time
        }
        self.history.append(state)
    
    def step_forward(self, dt: Optional[float] = None):
        """Advance simulation by one time step"""
        if dt is not None:
            self.dt = dt
        
        # Save state for potential reversal
        self._save_state()
        
        # Heat diffusion with planetary boundary conditions
        self.temperature = self._heat_diffusion_step_planetary()
        
        # Add internal heat generation (radioactive decay, tidal heating, etc.)
        self._apply_internal_heat_generation()
        
        # Update center of mass and pressure
        self._calculate_center_of_mass()
        self._calculate_planetary_pressure()
        
        # Apply metamorphic processes
        metamorphic_changes = self._apply_metamorphism()
        
        # Apply gravitational differentiation (density sorting)
        differentiation_changes = self._apply_gravitational_differentiation()
        
        # Apply gravitational collapse (rocks falling into air cavities)
        collapse_changes = self._apply_gravitational_collapse()
        
        # Apply water vaporization and air migration
        fluid_changes = self._apply_fluid_dynamics()
        
        # Update material properties if rock types changed
        if metamorphic_changes or differentiation_changes or collapse_changes or fluid_changes:
            self._update_material_properties()
        
        # Update age
        self.age += self.dt
        self.time += self.dt
        
        # Final safety check: ensure SPACE cells stay as SPACE and at cosmic background temp
        space_mask = (self.rock_types == RockType.SPACE)
        self.temperature[space_mask] = 2.7  # Kelvin
        self.pressure[space_mask] = 0.0
    
    def step_backward(self):
        """Reverse simulation by one time step"""
        if len(self.history) > 0:
            state = self.history.pop()
            self.rock_types = state['rock_types']
            self.temperature = state['temperature']
            self.pressure = state['pressure']
            self.pressure_offset = state['pressure_offset']
            self.age = state['age']
            self.time = state['time']
            
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
        
        # Apply heat with much higher base temperature
        base_temp_increase = 800.0  # K - much more effective heating
        temp_addition = intensity * base_temp_increase
        
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
    
    def get_visualization_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get data for visualization (rock colors, temperature, pressure)"""
        # Create color array from rock types
        colors = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            for x in range(self.width):
                rock_type = self.rock_types[y, x]
                props = self.rock_db.get_properties(rock_type)
                colors[y, x] = props.color_rgb
        
        return colors, self.temperature, self.pressure
    
    def get_stats(self) -> dict:
        """Get simulation statistics"""
        # Convert to string values for unique counting
        rock_strings = np.array([rock.value for rock in self.rock_types.flatten()])
        unique_rocks, counts = np.unique(rock_strings, return_counts=True)
        rock_percentages = {rock: count/len(self.rock_types.flatten())*100 
                          for rock, count in zip(unique_rocks, counts)}
        
        stats = {
            'time': self.time,
            'dt': self.dt,
            'avg_temperature': np.mean(self.temperature) - 273.15,  # Convert to Celsius for display
            'max_temperature': np.max(self.temperature) - 273.15,   # Convert to Celsius for display
            'avg_pressure': np.mean(self.pressure),
            'max_pressure': np.max(self.pressure),
            'rock_composition': rock_percentages,
            'history_length': len(self.history)
        }
        
        # Add differentiation stats if available
        if hasattr(self, '_last_differentiation_stats'):
            stats.update(self._last_differentiation_stats)
        
        return stats 