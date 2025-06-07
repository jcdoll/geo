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
    
    def __init__(self, width: int, height: int, cell_size: float = 1000.0):
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
        self.age = np.zeros((height, width), dtype=np.float64)
        
        # Derived properties (computed from rock types)
        self.density = np.zeros((height, width), dtype=np.float64)
        self.thermal_conductivity = np.zeros((height, width), dtype=np.float64)
        self.specific_heat = np.zeros((height, width), dtype=np.float64)
        
        # Simulation parameters
        self.time = 0.0
        self.dt = 1000.0  # years per time step
        self.gravity = 9.81  # m/s²
        
        # Rock database
        self.rock_db = RockDatabase()
        
        # History for time reversal
        self.max_history = 1000
        self.history = []
        self.history_step = 0
        
        # Initialize with realistic conditions
        self._setup_initial_conditions()
        self._update_material_properties()
    
    def _setup_initial_conditions(self):
        """Set up realistic initial Earth conditions"""
        # Surface temperature gradient (15°C at surface, geothermal gradient)
        surface_temp = 15.0  # °C
        geothermal_gradient = 25.0  # °C per km
        
        for y in range(self.height):
            depth_km = y * self.cell_size / 1000.0
            temp = surface_temp + geothermal_gradient * depth_km
            
            # Add some randomness for more realistic conditions
            temp += np.random.normal(0, 10, self.width)
            
            self.temperature[y, :] = temp
            
            # Set initial rock types based on depth
            if y < self.height * 0.1:  # Top 10% - sedimentary
                self.rock_types[y, :] = np.random.choice([
                    RockType.SANDSTONE, RockType.LIMESTONE, RockType.SHALE
                ], size=self.width)
            elif y < self.height * 0.3:  # Upper crust - granite
                self.rock_types[y, :] = RockType.GRANITE
            else:  # Lower crust and mantle - basalt
                self.rock_types[y, :] = RockType.BASALT
        
        # Calculate initial pressure (lithostatic)
        self._calculate_pressure()
    
    def _update_material_properties(self):
        """Update material property grids based on current rock types"""
        for y in range(self.height):
            for x in range(self.width):
                rock_type = self.rock_types[y, x]
                props = self.rock_db.get_properties(rock_type)
                
                self.density[y, x] = props.density
                self.thermal_conductivity[y, x] = props.thermal_conductivity
                self.specific_heat[y, x] = props.specific_heat
    
    def _calculate_pressure(self):
        """Calculate lithostatic pressure from overlying rock"""
        self.pressure[0, :] = 0.1  # Atmospheric pressure in MPa
        
        for y in range(1, self.height):
            # Pressure increases with depth due to weight of overlying rock
            depth_pressure = self.density[y-1, :] * self.gravity * self.cell_size / 1e6  # Convert to MPa
            self.pressure[y, :] = self.pressure[y-1, :] + depth_pressure
    
    @staticmethod
    @jit(nopython=True)
    def _heat_diffusion_step(temperature: np.ndarray, thermal_conductivity: np.ndarray, 
                           specific_heat: np.ndarray, density: np.ndarray, 
                           dt: float, cell_size: float) -> np.ndarray:
        """Perform one step of heat diffusion using finite differences"""
        new_temp = temperature.copy()
        height, width = temperature.shape
        
        # Heat diffusion coefficient: k/(ρ*c)
        alpha = thermal_conductivity / (density * specific_heat)
        
        for y in range(1, height-1):
            for x in range(1, width-1):
                # Second derivatives (Laplacian)
                d2T_dx2 = (temperature[y, x+1] - 2*temperature[y, x] + temperature[y, x-1]) / (cell_size**2)
                d2T_dy2 = (temperature[y+1, x] - 2*temperature[y, x] + temperature[y-1, x]) / (cell_size**2)
                
                # Heat equation: dT/dt = α * ∇²T
                dT_dt = alpha[y, x] * (d2T_dx2 + d2T_dy2)
                new_temp[y, x] += dT_dt * dt * 365.25 * 24 * 3600  # Convert years to seconds
        
        return new_temp
    
    def _apply_metamorphism(self):
        """Apply metamorphic transitions based on P-T conditions"""
        changes_made = False
        
        for y in range(self.height):
            for x in range(self.width):
                current_rock = self.rock_types[y, x]
                temp = self.temperature[y, x]
                pressure = self.pressure[y, x]
                
                # Check for melting first
                if self.rock_db.should_melt(current_rock, temp) and current_rock != RockType.MAGMA:
                    self.rock_types[y, x] = RockType.MAGMA
                    changes_made = True
                    continue
                
                # Check for metamorphic transitions
                new_rock = self.rock_db.get_metamorphic_product(current_rock, temp, pressure)
                if new_rock and new_rock != current_rock:
                    self.rock_types[y, x] = new_rock
                    changes_made = True
                
                # Handle magma cooling
                if current_rock == RockType.MAGMA and temp < 800:
                    # Determine composition based on location (simplified)
                    composition = "felsic" if y < self.height * 0.5 else "mafic"
                    new_rock = self.rock_db.get_cooling_product(temp, pressure, composition)
                    self.rock_types[y, x] = new_rock
                    changes_made = True
        
        return changes_made
    
    def _save_state(self):
        """Save current state for time reversal"""
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        
        state = {
            'rock_types': self.rock_types.copy(),
            'temperature': self.temperature.copy(),
            'pressure': self.pressure.copy(),
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
        
        # Heat diffusion
        self.temperature = GeologySimulation._heat_diffusion_step(
            self.temperature, self.thermal_conductivity, self.specific_heat, 
            self.density, self.dt, self.cell_size
        )
        
        # Update pressure
        self._calculate_pressure()
        
        # Apply metamorphic processes
        changes_made = self._apply_metamorphism()
        
        # Update material properties if rock types changed
        if changes_made:
            self._update_material_properties()
        
        # Update age
        self.age += self.dt
        self.time += self.dt
    
    def step_backward(self):
        """Reverse simulation by one time step"""
        if len(self.history) > 0:
            state = self.history.pop()
            self.rock_types = state['rock_types']
            self.temperature = state['temperature']
            self.pressure = state['pressure']
            self.age = state['age']
            self.time = state['time']
            
            self._update_material_properties()
            return True
        return False
    
    def add_heat_source(self, x: int, y: int, radius: int, temperature: float):
        """Add a localized heat source (e.g., magma intrusion)"""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        self.temperature[ny, nx] = max(self.temperature[ny, nx], temperature)
    
    def apply_tectonic_stress(self, x: int, y: int, radius: int, pressure_increase: float):
        """Apply tectonic stress to increase pressure locally"""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        self.pressure[ny, nx] += pressure_increase
    
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
        
        return {
            'time': self.time,
            'dt': self.dt,
            'avg_temperature': np.mean(self.temperature),
            'max_temperature': np.max(self.temperature),
            'avg_pressure': np.mean(self.pressure),
            'max_pressure': np.max(self.pressure),
            'rock_composition': rock_percentages,
            'history_length': len(self.history)
        } 