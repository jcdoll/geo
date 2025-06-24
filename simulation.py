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
from multigrid_dispatcher import SmootherType
from heat_transfer import HeatTransfer
from heat_transfer_multigrid import HeatTransferMultigrid
from heat_transfer_vectorized_stable import HeatTransferVectorized
from transport import FluxTransport
from transport_vectorized import FluxTransportVectorized
from solar_heating_vectorized import apply_solar_heating_vectorized
from solar_heating_proper_safe import apply_solar_heating_proper
from atmospheric_processes import AtmosphericProcesses
from phase_transitions import PhaseTransitionSystem


class FluxSimulation:
    """Main simulation class that orchestrates all physics modules."""
    
    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 50.0,
        scenario: Optional[str] = None,
        use_multigrid_heat: bool = False,
        cell_depth: Optional[float] = None,
        smoother_type: SmootherType = SmootherType.JACOBI,
    ):
        """
        Initialize flux-based simulation.
        
        Args:
            nx: Grid cells in x direction
            ny: Grid cells in y direction
            dx: Cell size in meters
            scenario: Name of scenario to load (or None for default)
            use_multigrid_heat: Use multigrid solver for heat transfer (default: False, uses ADI)
            cell_depth: Cell depth in meters (defaults to domain width for cubic simulation)
            smoother_type: Which smoother to use for pressure solver (default: JACOBI)
        """
        # Core components
        self.state = FluxState(nx, ny, dx, cell_depth=cell_depth)
        
        # Transport module (vectorized)
        self.transport = FluxTransportVectorized(self.state)
            
        # Physics modules
        self.gravity_solver = GravitySolver(self.state)
        self.pressure_solver = PressureSolver(self.state, smoother_type)
        
        # Choose heat transfer solver
        self.use_multigrid_heat = use_multigrid_heat
        if use_multigrid_heat:
            self.heat_transfer = HeatTransferMultigrid(self.state)
        else:
            # Use vectorized heat transfer for better performance
            self.heat_transfer = HeatTransferVectorized(self.state)
            
        self.physics = FluxPhysics(self.state)
        self.material_db = MaterialDatabase()
        
        # Initialize atmospheric processes module
        self.atmospheric_processes = AtmosphericProcesses(self.state)
        
        # Initialize phase transition system
        self.phase_transitions = PhaseTransitionSystem(self.state, self.material_db)
        
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
        
        # Debug flags to disable individual physics processes
        self.enable_gravity = True
        self.enable_momentum = True
        self.enable_advection = True
        self.enable_heat_transfer = True
        self.enable_solar_heating = True
        self.enable_phase_transitions = True
        self.enable_atmospheric = True
        
        # Initialize scenario
        if scenario is not False:  # Allow False to skip setup
            setup_scenario(scenario, self.state, self.material_db)
            
            # Solve initial gravity, pressure, and velocity fields
            self._solve_initial_state()
            
        
    def reset(self, scenario: Optional[str] = None):
        """Reset simulation to initial state."""
        # Use provided scenario or fall back to stored one
        scenario_to_use = scenario if scenario is not None else self.scenario
        
        # Store current dimensions and settings
        nx, ny, dx = self.state.nx, self.state.ny, self.state.dx
        use_multigrid = self.use_multigrid_heat
        smoother = self.pressure_solver.smoother_type
        
        # Reinitialize
        self.__init__(nx, ny, dx, scenario=scenario_to_use, use_multigrid_heat=use_multigrid, smoother_type=smoother)
        
    def _solve_initial_state(self):
        """Solve gravity and pressure fields for initial state."""
        # Solve gravity field
        gx, gy = self.gravity_solver.solve_gravity()
        self.state.gravity_x[:] = gx
        self.state.gravity_y[:] = gy
        
        # For initial state, we just need gravity solved
        # Velocity starts at zero which is appropriate for initial conditions
        # Pressure will be computed during first timestep
        
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
        
        # Reset power density tracking for this timestep
        self.state.power_density.fill(0.0)
        
        # 1. Self-gravity
        t0 = time.perf_counter()
        if self.enable_gravity:
            gx, gy = self.gravity_solver.solve_gravity()
        else:
            gx, gy = self.state.gravity_x, self.state.gravity_y  # Use existing values
        self.step_timings['gravity'] = time.perf_counter() - t0
        
        # 2. Momentum update (projection enforces incompressibility)
        t0 = time.perf_counter()
        if self.enable_momentum:
            self.physics.update_momentum(gx, gy, dt)
        self.step_timings['momentum'] = time.perf_counter() - t0
        
        # 4. Advection (flux-based transport)
        t0 = time.perf_counter()
        if self.enable_advection:
            self.transport.advect_materials_vectorized(dt)
        self.step_timings['advection'] = time.perf_counter() - t0
        
        # 5. Heat transfer (diffusion + radiation)
        t0 = time.perf_counter()
        if self.enable_heat_transfer:
            self.heat_transfer.solve_heat_equation(dt)
        self.step_timings['heat_transfer'] = time.perf_counter() - t0
        
        # 6. Solar heating (using proper material absorption)
        t0 = time.perf_counter()
        if self.enable_solar_heating:
            apply_solar_heating_proper(self.state, dt, self.solar_angle, self.solar_constant)
        self.step_timings['solar_heating'] = time.perf_counter() - t0
        
        # 7. Phase transitions (general system handles all material transitions)
        t0 = time.perf_counter()
        if self.enable_phase_transitions:
            self.phase_transitions.apply_transitions(dt)
        self.step_timings['phase_transitions'] = time.perf_counter() - t0
        
        # 8. Atmospheric processes (convection only - phase transitions handled above)
        t0 = time.perf_counter()
        if self.enable_atmospheric:
            self.atmospheric_processes.apply_convection()
        self.step_timings['atmospheric_processes'] = time.perf_counter() - t0
        
        # 9. Update mixture properties
        t0 = time.perf_counter()
        self.state.update_mixture_properties(self.material_db)
        self.step_timings['update_properties'] = time.perf_counter() - t0
        
        # 10. Final safety check - ensure no NaN/inf values propagated
        if not np.all(np.isfinite(self.state.temperature)):
            nan_count = np.sum(~np.isfinite(self.state.temperature))
            print(f"Warning: {nan_count} NaN/inf temperatures after timestep, clamping to valid range")
            self.state.temperature = np.nan_to_num(self.state.temperature, 
                                                   nan=300.0, 
                                                   posinf=5000.0, 
                                                   neginf=self.t_space)
        
        # Clamp power density to reasonable range
        max_power = 1e6  # 1 MW/m³ max heating
        min_power = -1e5  # 100 kW/m³ max cooling
        self.state.power_density = np.clip(self.state.power_density, min_power, max_power)
        
    def apply_solar_heating(self, dt: float):
        """
        Apply solar heating using DDA ray marching.
        
        Traces rays from the sun through the atmosphere and deposits
        energy based on material absorption coefficients.
        """
        # Solar direction (angle from vertical)
        sun_x = np.sin(self.solar_angle)
        sun_y = np.cos(self.solar_angle)
        
        # Determine which boundary rays enter from
        if sun_y > 0:  # Sun above horizon
            # Spawn rays from top boundary
            for i in range(self.state.nx):
                self._trace_solar_ray(i, 0, sun_x, sun_y, dt)
        else:  # Sun below horizon (night)
            return  # No solar heating at night
            
        # Also spawn rays from side boundaries if sun is at an angle
        if abs(sun_x) > 0.1:
            if sun_x > 0:  # Sun from right
                for j in range(self.state.ny):
                    self._trace_solar_ray(self.state.nx-1, j, sun_x, sun_y, dt)
            else:  # Sun from left
                for j in range(self.state.ny):
                    self._trace_solar_ray(0, j, sun_x, sun_y, dt)
                    
    def _trace_solar_ray(self, start_x: int, start_y: int, 
                        sun_x: float, sun_y: float, dt: float):
        """
        Trace a single solar ray using DDA algorithm.
        
        Args:
            start_x, start_y: Starting position
            sun_x, sun_y: Sun direction (normalized)
            dt: Time step
        """
        # Initial intensity
        intensity = self.solar_constant
        
        # Ray direction (opposite of sun direction - rays go from sun to ground)
        dx = -sun_x
        dy = -sun_y
        
        # Current position
        x, y = float(start_x), float(start_y)
        
        # DDA parameters
        if abs(dx) > abs(dy):
            # X-major
            step_x = 1.0 if dx > 0 else -1.0
            step_y = dy / abs(dx)
        else:
            # Y-major
            step_y = 1.0 if dy > 0 else -1.0
            step_x = dx / abs(dy)
            
        # Material absorption coefficients from PHYSICS_FLUX.md
        absorption_coeffs = {
            MaterialType.AIR: 0.001,
            MaterialType.WATER_VAPOR: 0.005,
            MaterialType.WATER: 0.02,
            MaterialType.ICE: 0.01,
            MaterialType.SPACE: 0.0,
            # All rocks/solids are opaque
            MaterialType.ROCK: 1.0,
            MaterialType.SAND: 1.0,
            MaterialType.URANIUM: 1.0,
            MaterialType.MAGMA: 1.0,
        }
        
        # March along ray
        while 0 <= x < self.state.nx and 0 <= y < self.state.ny and intensity > 1e-6:
            ix, iy = int(x), int(y)
            
            # Skip space cells
            if self.state.density[iy, ix] < 0.1:
                x += step_x
                y += step_y
                continue
                
            # Compute absorption from material mixture
            total_absorption = 0.0
            weighted_albedo = 0.0
            for mat_type in MaterialType:
                vol_frac = self.state.vol_frac[mat_type.value, iy, ix]
                if vol_frac > 0:
                    total_absorption += vol_frac * absorption_coeffs.get(mat_type, 0.1)
                    # Get material albedo
                    mat_props = self.material_db.get_properties(mat_type)
                    weighted_albedo += vol_frac * mat_props.albedo
                    
            # Apply material albedo (1 - albedo = fraction absorbed)
            effective_absorption = total_absorption * (1.0 - weighted_albedo)
            
            # Energy absorbed in this cell
            absorbed = intensity * effective_absorption
            
            # Convert to volumetric power density (W/m³)
            # Power absorbed per unit area, divided by cell depth
            volumetric_power = absorbed / self.state.cell_depth
            
            # Update power density tracking
            self.state.power_density[iy, ix] += volumetric_power
            
            # Convert to temperature change
            if self.state.density[iy, ix] > 0 and self.state.specific_heat[iy, ix] > 0:
                dT = volumetric_power * dt / (self.state.density[iy, ix] * 
                                             self.state.specific_heat[iy, ix])
                self.state.temperature[iy, ix] += dT
                
            # Attenuate ray
            intensity *= (1.0 - effective_absorption)
            
            # Stop if we hit opaque material
            if effective_absorption > 0.99:
                break
                
            # Step to next cell
            x += step_x
            y += step_y
            
        
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