"""
Main simulation loop for flux-based geological simulation.

This module orchestrates the time integration using operator splitting
for clarity and modularity.
"""

import numpy as np
from typing import Optional, Dict, Any
import time

from state import FluxState
from physics import FluxPhysics
from materials import MaterialDatabase, MaterialType
from scenarios import setup_scenario
from gravity_solver import GravitySolver
from pressure_solver import PressureSolver
from heat_transfer import HeatTransfer
from transport import FluxTransport


class FluxSimulation:
    """Main simulation class that orchestrates all physics modules."""
    
    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 50.0,
        scenario: Optional[str] = None,
    ):
        """
        Initialize flux-based simulation.
        
        Args:
            nx: Grid cells in x direction
            ny: Grid cells in y direction
            dx: Cell size in meters
            scenario: Name of scenario to load (or None for default)
        """
        # Core components
        self.state = FluxState(nx, ny, dx)
        
        # Transport module
        self.transport = FluxTransport(self.state)
            
        # Physics modules
        self.gravity_solver = GravitySolver(self.state)
        self.pressure_solver = PressureSolver(self.state)
        self.heat_transfer = HeatTransfer(self.state)
        self.physics = FluxPhysics(self.state)
        self.material_db = MaterialDatabase()
        
        # Simulation parameters
        self.paused = True  # Start paused so user can see initial state
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
        
        # Store scenario for reset
        self.scenario = scenario
        
        # Initialize scenario
        if scenario is not False:  # Allow False to skip setup
            setup_scenario(scenario, self.state, self.material_db)
            
        
    def reset(self, scenario: Optional[str] = None):
        """Reset simulation to initial state."""
        # Use provided scenario or fall back to stored one
        scenario_to_use = scenario if scenario is not None else self.scenario
        
        # Store current dimensions
        nx, ny, dx = self.state.nx, self.state.ny, self.state.dx
        
        # Reinitialize
        self.__init__(nx, ny, dx, scenario=scenario_to_use)
        
    def step_forward(self):
        """Execute one simulation timestep."""
        if self.paused:
            return
            
        start_time = time.time()
        
        # Compute adaptive timestep
        t0 = time.perf_counter()
        dt = self.physics.apply_cfl_limit()
        self.state.dt = dt
        if hasattr(self, 'step_timings'):
            self.step_timings['cfl_calculation'] = time.perf_counter() - t0
        
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
        # Initialize timing if needed
        if not hasattr(self, 'step_timings'):
            self.step_timings = {}
            
        import time
        
        # 1. Self-gravity
        t0 = time.perf_counter()
        gx, gy = self.gravity_solver.solve_gravity()
        self.step_timings['gravity'] = time.perf_counter() - t0
        
        # 2. Momentum update (projection enforces incompressibility)
        t0 = time.perf_counter()
        self.physics.update_momentum(gx, gy, dt)
        self.step_timings['momentum'] = time.perf_counter() - t0
        
        # 4. Advection (flux-based transport)
        t0 = time.perf_counter()
        self.transport.advect_materials_vectorized(dt)
        self.step_timings['advection'] = time.perf_counter() - t0
        
        # 5. Heat transfer (diffusion + radiation)
        t0 = time.perf_counter()
        self.heat_transfer.solve_heat_equation(dt)
        self.step_timings['heat_transfer'] = time.perf_counter() - t0
        
        # 6. Solar heating
        t0 = time.perf_counter()
        self.apply_solar_heating(dt)
        self.step_timings['solar_heating'] = time.perf_counter() - t0
        
        # 7. Phase transitions
        t0 = time.perf_counter()
        self.apply_phase_transitions(dt)
        self.step_timings['phase_transitions'] = time.perf_counter() - t0
        
        # 8. Update mixture properties
        t0 = time.perf_counter()
        self.state.update_mixture_properties(self.material_db)
        self.step_timings['update_properties'] = time.perf_counter() - t0
        
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
            
    def apply_phase_transitions(self, dt: float):
        """
        Apply material phase transitions based on T-P conditions.
        
        Vectorized implementation for performance.
        """
        T = self.state.temperature
        P = self.state.pressure
        
        # Water -> Ice (freezing)
        freeze_mask = (T < 273.15) & (self.state.vol_frac[MaterialType.WATER.value] > 0)
        if np.any(freeze_mask):
            rate = 0.1 * dt  # 10% per second
            amount = self.state.vol_frac[MaterialType.WATER.value] * np.minimum(rate, 1.0)
            
            self.state.vol_frac[MaterialType.WATER.value] -= amount * freeze_mask
            self.state.vol_frac[MaterialType.ICE.value] += amount * freeze_mask
            # Latent heat
            self.state.heat_source += amount * freeze_mask * 3.34e5
            
        # Ice -> Water (melting)
        melt_mask = (T > 273.15) & (self.state.vol_frac[MaterialType.ICE.value] > 0)
        if np.any(melt_mask):
            rate = 0.1 * dt
            amount = self.state.vol_frac[MaterialType.ICE.value] * np.minimum(rate, 1.0)
            
            self.state.vol_frac[MaterialType.ICE.value] -= amount * melt_mask
            self.state.vol_frac[MaterialType.WATER.value] += amount * melt_mask
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