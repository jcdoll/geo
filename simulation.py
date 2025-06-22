"""
Main simulation loop for flux-based geological simulation.

This module orchestrates the time integration using operator splitting
for clarity and modularity.
"""

import numpy as np
from typing import Optional, Dict, Any
import time

from state import FluxState
from transport import FluxTransport
from physics import FluxPhysics
from materials import MaterialDatabase, MaterialType


class FluxSimulation:
    """Main simulation class that orchestrates all physics modules."""
    
    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 50.0,
        setup_planet: bool = True,
    ):
        """
        Initialize flux-based simulation.
        
        Args:
            nx: Grid cells in x direction
            ny: Grid cells in y direction
            dx: Cell size in meters
            setup_planet: Whether to create initial planet setup
        """
        # Core components
        self.state = FluxState(nx, ny, dx)
        self.transport = FluxTransport(self.state)
        self.physics = FluxPhysics(self.state)
        self.material_db = MaterialDatabase()
        
        # Simulation parameters
        self.paused = False
        self.step_count = 0
        self.solar_angle = 0.0
        self.solar_rotation_rate = 2 * np.pi / (24 * 3600)  # rad/s (24 hour day)
        
        # Performance tracking
        self.last_step_time = 0.0
        self.fps = 0.0
        
        # Physical constants
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        self.solar_constant = 1361.0  # W/m²
        self.t_space = 2.7  # K (cosmic background)
        
        # Initialize planet if requested
        if setup_planet:
            self.setup_initial_planet()
            
    def setup_initial_planet(self):
        """Create a simple initial planet configuration."""
        nx, ny = self.state.nx, self.state.ny
        
        # Create circular planet
        cx, cy = nx // 2, ny // 2
        radius = min(nx, ny) // 3
        
        y_grid, x_grid = np.ogrid[:ny, :nx]
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        # Rock core
        rock_mask = dist_from_center < radius
        self.state.vol_frac[MaterialType.SPACE][rock_mask] = 0.0
        self.state.vol_frac[MaterialType.ROCK][rock_mask] = 1.0
        
        # Thin atmosphere
        atmos_mask = (dist_from_center >= radius) & (dist_from_center < radius + 10)
        self.state.vol_frac[MaterialType.SPACE][atmos_mask] = 0.0
        self.state.vol_frac[MaterialType.AIR][atmos_mask] = 1.0
        
        # Add some water on surface
        water_mask = (dist_from_center >= radius - 5) & (dist_from_center < radius) & (y_grid < cy)
        self.state.vol_frac[MaterialType.ROCK][water_mask] = 0.5
        self.state.vol_frac[MaterialType.WATER][water_mask] = 0.5
        
        # Set initial temperature
        self.state.temperature.fill(288.0)  # 15°C
        
        # Add some heat in core
        core_mask = dist_from_center < radius // 2
        self.state.temperature[core_mask] = 1000.0  # Hot core
        
        # Normalize and update properties
        self.state.normalize_volume_fractions()
        self.state.update_mixture_properties(self.material_db)
        
    def reset(self):
        """Reset simulation to initial state."""
        self.__init__(self.state.nx, self.state.ny, self.state.dx, setup_planet=True)
        
    def step_forward(self):
        """Execute one simulation timestep."""
        if self.paused:
            return
            
        start_time = time.time()
        
        # Compute adaptive timestep
        dt = self.physics.apply_cfl_limit()
        self.state.dt = dt
        
        # Operator splitting approach
        self.timestep(dt)
        
        # Update simulation time
        self.state.time += dt
        self.step_count += 1
        self.solar_angle += dt * self.solar_rotation_rate
        
        # Track performance
        self.last_step_time = time.time() - start_time
        if self.last_step_time > 0:
            self.fps = 1.0 / self.last_step_time
            
    def timestep(self, dt: float):
        """
        Execute one timestep using operator splitting.
        
        Args:
            dt: Time step in seconds
        """
        # 1. Self-gravity
        gx, gy = self.physics.solve_gravity()
        
        # 2. Pressure
        pressure = self.physics.solve_pressure(gx, gy)
        
        # 3. Momentum update (pressure gradients + gravity)
        self.physics.update_momentum(pressure, gx, gy, dt)
        
        # 4. Advection (flux-based transport)
        self.transport.advect_materials(dt)
        
        # 5. Thermal diffusion
        self.transport.diffuse_heat(dt)
        
        # 6. Solar heating
        self.apply_solar_heating(dt)
        
        # 7. Radiative cooling
        self.apply_radiative_cooling(dt)
        
        # 8. Phase transitions
        self.apply_phase_transitions(dt)
        
        # 9. Update mixture properties
        self.state.update_mixture_properties(self.material_db)
        
    def apply_solar_heating(self, dt: float):
        """
        Apply solar heating using ray marching.
        
        Simplified version - full DDA implementation would be imported
        from ca/heat_transfer.py
        """
        # Simple top-down heating for now
        solar_flux = self.solar_constant * 1e-5  # Reduced for simulation
        
        # Apply to top row with absorption
        for i in range(self.state.nx):
            # Skip space cells
            if self.state.density[0, i] < 0.1:
                continue
                
            # Material absorption
            absorbed = solar_flux * (1.0 - self.state.emissivity[0, i])
            
            # Convert to temperature change
            dT = absorbed * dt / (self.state.density[0, i] * self.state.specific_heat[0, i])
            self.state.temperature[0, i] += dT
            
    def apply_radiative_cooling(self, dt: float):
        """Apply Stefan-Boltzmann radiative cooling."""
        # Skip space cells
        mask = self.state.density > 0.1
        
        # Stefan-Boltzmann cooling
        cooling = self.state.emissivity * self.stefan_boltzmann * (
            self.state.temperature**4 - self.t_space**4
        )
        
        # Apply greenhouse effect for atmospheric cells
        vapor_frac = self.state.vol_frac[MaterialType.WATER_VAPOR]
        greenhouse = 0.1 + 0.5 * np.tanh(np.log(1 + vapor_frac * 10) / 10)
        cooling *= (1 - greenhouse)
        
        # Apply cooling
        dT = -cooling * dt / (self.state.density * self.state.specific_heat + 1e-10)
        self.state.temperature[mask] += dT[mask]
        
    def apply_phase_transitions(self, dt: float):
        """
        Apply material phase transitions based on T-P conditions.
        
        Vectorized implementation for performance.
        """
        T = self.state.temperature
        P = self.state.pressure
        
        # Water -> Ice (freezing)
        freeze_mask = (T < 273.15) & (self.state.vol_frac[MaterialType.WATER] > 0)
        if np.any(freeze_mask):
            rate = 0.1 * dt  # 10% per second
            amount = self.state.vol_frac[MaterialType.WATER] * np.minimum(rate, 1.0)
            
            self.state.vol_frac[MaterialType.WATER] -= amount * freeze_mask
            self.state.vol_frac[MaterialType.ICE] += amount * freeze_mask
            # Latent heat
            self.state.heat_source += amount * freeze_mask * 3.34e5
            
        # Ice -> Water (melting)
        melt_mask = (T > 273.15) & (self.state.vol_frac[MaterialType.ICE] > 0)
        if np.any(melt_mask):
            rate = 0.1 * dt
            amount = self.state.vol_frac[MaterialType.ICE] * np.minimum(rate, 1.0)
            
            self.state.vol_frac[MaterialType.ICE] -= amount * melt_mask
            self.state.vol_frac[MaterialType.WATER] += amount * melt_mask
            # Latent heat
            self.state.heat_source -= amount * melt_mask * 3.34e5
            
        # Apply heat sources
        if np.any(self.state.heat_source != 0):
            dT = self.state.heat_source * dt / (
                self.state.density * self.state.specific_heat + 1e-10
            )
            self.state.temperature += dT
            self.state.heat_source.fill(0.0)
            
        # Normalize after transitions
        self.state.normalize_volume_fractions()
        
    def get_info(self) -> Dict[str, Any]:
        """Get simulation information for display."""
        return {
            'time': self.state.time,
            'timestep': self.state.dt,
            'step_count': self.step_count,
            'fps': self.fps,
            'total_mass': self.state.get_total_mass(),
            'total_energy': self.state.get_total_energy(),
            'material_inventory': self.state.get_material_inventory(),
            'avg_temperature': np.mean(self.state.temperature),
            'max_temperature': np.max(self.state.temperature),
            'min_temperature': np.min(self.state.temperature),
        }