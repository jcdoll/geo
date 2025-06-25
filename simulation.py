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
from multigrid import SmootherType
from heat_transfer import HeatTransfer
from transport import FluxTransport
from solar_heating import apply_solar_heating
from atmospheric_processes import AtmosphericProcesses
from phase_transitions import PhaseTransitionSystem
from gravity_solver import SolverMethod

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
        init_gravity_ramp: bool = True,
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
            init_gravity_ramp: Use gradual gravity ramping during initialization (default: True)
        """
        # Core components
        self.state = FluxState(nx, ny, dx, cell_depth=cell_depth)
        
        # Transport module (vectorized)
        self.transport = FluxTransport(self.state)
            
        # Physics modules
        # Initialize gravity solver with DFT method (no boundary artifacts)
        self.gravity_solver = GravitySolver(self.state, method=SolverMethod.DFT)
        self.pressure_solver = PressureSolver(self.state, smoother_type)
        
        # Choose heat transfer solver
        self.use_multigrid_heat = use_multigrid_heat
        solver_method = "multigrid" if use_multigrid_heat else "adi"
        self.heat_transfer = HeatTransfer(self.state, solver_method=solver_method, simulation=self)
            
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
        
        # Store scenario and init params for reset
        self.scenario = scenario
        self.init_gravity_ramp = init_gravity_ramp
        
        # Debug flags to disable individual physics processes
        self.enable_gravity = True
        self.enable_momentum = True
        self.enable_advection = True
        self.enable_heat_transfer = True
        self.enable_uranium_heating = True  # Separate from heat transfer
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
        
        # Preserve physics enable/disable flags
        saved_flags = {
            'enable_gravity': self.enable_gravity,
            'enable_momentum': self.enable_momentum,
            'enable_advection': self.enable_advection,
            'enable_heat_transfer': self.enable_heat_transfer,
            'enable_uranium_heating': self.enable_uranium_heating,
            'enable_solar_heating': self.enable_solar_heating,
            'enable_phase_transitions': self.enable_phase_transitions,
            'enable_atmospheric': self.enable_atmospheric,
        }
        
        # Reinitialize
        self.__init__(nx, ny, dx, scenario=scenario_to_use, use_multigrid_heat=use_multigrid, smoother_type=smoother)
        
        # Restore physics flags
        for flag_name, flag_value in saved_flags.items():
            setattr(self, flag_name, flag_value)
        
    def _solve_initial_state(self):
        """Solve gravity, pressure, and velocity fields for initial state.
        
        CRITICAL: Pressure is ONLY EVER calculated through velocity projection method.
        
        NEVER NEVER NEVER calculate pressure by:
        - Integration (e.g., P = P₀ + ∫ρg dy) - THIS IS WRONG
        - Hydrostatic approximation - THIS IS WRONG
        - Any direct solving - THIS IS WRONG
        
        The velocity projection method in update_momentum() is the ONLY correct way.
        It works even with zero initial velocity - the projection enforces incompressibility
        and produces the correct pressure field.
        """
        # Run timestep with dt=0 and initialization flag
        # This will:
        # 1. Calculate gravity field
        # 2. Establish pressure equilibrium (with optional ramping)
        # 3. Calculate initial power density from all heat sources
        self.timestep(0.0, is_initialization=True)
        
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
            
    def timestep(self, dt: float, is_initialization: bool = False):
        """
        Execute one timestep using operator splitting.
        
        Args:
            dt: Time step in seconds (0 for initialization)
            is_initialization: True when called from _solve_initial_state
        """
        # Initialize timing if needed
        if not hasattr(self, 'step_timings'):
            self.step_timings = {}
        
        # Reset power density tracking for this timestep
        self.state.power_density.fill(0.0)
        
        # 1. Self-gravity
        t0 = time.perf_counter()
        if self.enable_gravity:
            gx, gy = self.gravity_solver.solve_gravity()
            self.state.gravity_x[:] = gx
            self.state.gravity_y[:] = gy
        else:
            gx, gy = self.state.gravity_x, self.state.gravity_y  # Use existing values
        self.step_timings['gravity'] = time.perf_counter() - t0
        
        # 2. Momentum update (projection enforces incompressibility)
        t0 = time.perf_counter()
        if self.enable_momentum:
            if is_initialization and self.init_gravity_ramp:
                # Gradually establish pressure equilibrium for initialization
                n_init_steps = 10
                dt_init = 0.001  # Very small timestep for gentle initialization
                
                for i in range(n_init_steps):
                    # Gradually ramp up the gravity force to avoid shock
                    ramp_factor = (i + 1) / n_init_steps
                    gx_ramped = gx * ramp_factor
                    gy_ramped = gy * ramp_factor
                    
                    # Update momentum with ramped gravity
                    self.physics.update_momentum(gx_ramped, gy_ramped, dt_init)
                    
                    # Also update face coefficients each step to ensure consistency
                    self.state.update_face_coefficients()
            elif dt > 0:
                # Normal momentum update
                self.physics.update_momentum(gx, gy, dt)
        self.step_timings['momentum'] = time.perf_counter() - t0
        
        # 4. Advection (flux-based transport)
        t0 = time.perf_counter()
        if self.enable_advection and dt > 0:
            self.transport.advect_materials_vectorized(dt)
        self.step_timings['advection'] = time.perf_counter() - t0
        
        # 5. Heat transfer (diffusion + radiation)
        t0 = time.perf_counter()
        if self.enable_heat_transfer:
            self.heat_transfer.solve_heat_equation(dt)
        self.step_timings['heat_transfer'] = time.perf_counter() - t0
        
        # 5b. Uranium heating (separate from general heat transfer)
        t0 = time.perf_counter()
        # Always calculate uranium heating for power density display,
        # but only apply temperature changes if enabled
        self.heat_transfer.apply_heat_generation(dt, apply_heating=self.enable_uranium_heating)
        self.step_timings['uranium_heating'] = time.perf_counter() - t0
        
        # 6. Solar heating (using proper material absorption)
        t0 = time.perf_counter()
        if self.enable_solar_heating:
            apply_solar_heating(self.state, dt, self.solar_angle, self.solar_constant)
        self.step_timings['solar_heating'] = time.perf_counter() - t0
        
        # 7. Phase transitions (general system handles all material transitions)
        t0 = time.perf_counter()
        if self.enable_phase_transitions and dt > 0:
            self.phase_transitions.apply_transitions(dt)
        self.step_timings['phase_transitions'] = time.perf_counter() - t0
        
        # 8. Atmospheric processes (convection only - phase transitions handled above)
        t0 = time.perf_counter()
        if self.enable_atmospheric and dt > 0:
            self.atmospheric_processes.apply_convection()
        self.step_timings['atmospheric_processes'] = time.perf_counter() - t0
        
        # 9. Update mixture properties
        t0 = time.perf_counter()
        if dt > 0 or is_initialization:
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
        
        # No artificial limits on power density - let physics determine the values

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