"""
Physics solver integration for flux-based simulation.

This module integrates existing multigrid solvers (gravity, pressure) with
the flux framework and handles momentum updates.
"""

import numpy as np
from typing import Tuple, Optional
from state import FluxState

# Import existing solvers from ca/ directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ca'))

try:
    from ca.gravity_solver import solve_self_gravity
    from ca.pressure_solver import solve_poisson_variable_multigrid
except ImportError:
    # Placeholder implementations for now
    def solve_self_gravity(density, dx, G=6.67430e-11):
        """Placeholder gravity solver."""
        ny, nx = density.shape
        gx = np.zeros_like(density)
        gy = np.ones_like(density) * -9.81  # Default downward gravity
        return gx, gy
        
    def solve_poisson_variable_multigrid(rhs, dx, bc='dirichlet'):
        """Placeholder pressure solver."""
        return np.zeros_like(rhs)


class FluxPhysics:
    """Integrates physics solvers with flux framework."""
    
    def __init__(self, state: FluxState):
        """
        Initialize physics integration.
        
        Args:
            state: FluxState instance to operate on
        """
        self.state = state
        self.G = 6.67430e-11  # Gravitational constant
        
    def solve_gravity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for gravitational field using existing multigrid solver.
        
        Returns:
            (gx, gy): Gravitational acceleration components
        """
        # Use continuous density field
        gx, gy = solve_self_gravity(self.state.density, self.state.dx, self.G)
        
        # Store in state
        self.state.gravity_x = gx
        self.state.gravity_y = gy
        
        return gx, gy
        
    def solve_pressure(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """
        Solve for pressure field using multigrid solver.
        
        Critical: Use same discretization as gradient operator to ensure
        force balance at equilibrium.
        
        Args:
            gx, gy: Gravitational field components
            
        Returns:
            pressure: Pressure field
        """
        # Build RHS: ∇²P = ∇·(ρg) = ρ(∇·g) + g·(∇ρ)
        rhs = self._build_pressure_rhs(gx, gy)
        
        # Solve Poisson equation
        pressure = solve_poisson_variable_multigrid(rhs, self.state.dx)
        
        # Apply boundary conditions
        # Set pressure to zero in vacuum/space
        vacuum_threshold = 0.1  # kg/m³
        pressure[self.state.density < vacuum_threshold] = 0.0
        
        # Store in state
        self.state.pressure = pressure
        
        return pressure
        
    def _build_pressure_rhs(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """
        Build right-hand side for pressure Poisson equation.
        
        For self-gravity: ∇²P = -4πGρ² + g·∇ρ
        """
        rho = self.state.density
        dx = self.state.dx
        
        # Compute density gradients (central differences)
        drho_dx = np.zeros_like(rho)
        drho_dy = np.zeros_like(rho)
        
        # X-gradient
        drho_dx[:, 1:-1] = (rho[:, 2:] - rho[:, :-2]) / (2 * dx)
        drho_dx[:, 0] = (rho[:, 1] - rho[:, 0]) / dx
        drho_dx[:, -1] = (rho[:, -1] - rho[:, -2]) / dx
        
        # Y-gradient
        drho_dy[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2 * dx)
        drho_dy[0, :] = (rho[1, :] - rho[0, :]) / dx
        drho_dy[-1, :] = (rho[-1, :] - rho[-2, :]) / dx
        
        # Compute divergence of gravity (for self-gravity: ∇·g = -4πGρ)
        div_g = -4 * np.pi * self.G * rho
        
        # Build RHS
        rhs = rho * div_g + gx * drho_dx + gy * drho_dy
        
        return rhs
        
    def update_momentum(self, pressure: np.ndarray, gx: np.ndarray, gy: np.ndarray, dt: float):
        """
        Update velocities from pressure gradients and gravity.
        
        Critical: Use same gradient stencil as Poisson solver!
        
        Args:
            pressure: Pressure field
            gx, gy: Gravitational field
            dt: Time step
        """
        dx = self.state.dx
        rho = self.state.density + 1e-10  # Avoid division by zero
        
        # Pressure gradients (using same stencil as solver)
        dpdx = np.zeros_like(pressure)
        dpdy = np.zeros_like(pressure)
        
        # Central differences matching the solver
        dpdx[:, 1:-1] = (pressure[:, 2:] - pressure[:, :-2]) / (2 * dx)
        dpdy[1:-1, :] = (pressure[2:, :] - pressure[:-2, :]) / (2 * dx)
        
        # One-sided at boundaries
        dpdx[:, 0] = (pressure[:, 1] - pressure[:, 0]) / dx
        dpdx[:, -1] = (pressure[:, -1] - pressure[:, -2]) / dx
        dpdy[0, :] = (pressure[1, :] - pressure[0, :]) / dx
        dpdy[-1, :] = (pressure[-1, :] - pressure[-2, :]) / dx
        
        # Update velocities: dv/dt = -∇P/ρ + g
        self.state.velocity_x += dt * (-dpdx / rho + gx)
        self.state.velocity_y += dt * (-dpdy / rho + gy)
        
        # Apply viscous damping
        self.apply_viscous_damping(dt)
        
    def apply_viscous_damping(self, dt: float):
        """
        Apply material-dependent viscous damping.
        
        Args:
            dt: Time step
        """
        # Viscosity provides resistance to motion
        # v_new = v_old * (1 - viscosity * dt)
        damping = 1.0 - self.state.viscosity * dt
        damping = np.clip(damping, 0.0, 1.0)  # Ensure stability
        
        self.state.velocity_x *= damping
        self.state.velocity_y *= damping
        
    def apply_cfl_limit(self) -> float:
        """
        Compute CFL-limited timestep.
        
        Returns:
            dt: Maximum stable timestep
        """
        dx = self.state.dx
        
        # Maximum velocity
        vmax = np.maximum(
            np.abs(self.state.velocity_x).max(),
            np.abs(self.state.velocity_y).max()
        )
        
        # Thermal diffusion limit
        alpha_max = np.max(self.state.thermal_conductivity / 
                          (self.state.density * self.state.specific_heat + 1e-10))
        
        # CFL conditions
        if vmax > 0:
            dt_advection = 0.5 * dx / vmax
        else:
            dt_advection = 1.0
            
        if alpha_max > 0:
            dt_diffusion = 0.25 * dx * dx / alpha_max
        else:
            dt_diffusion = 1.0
            
        # Take minimum with safety factor
        dt = 0.8 * min(dt_advection, dt_diffusion)
        
        # Clamp to reasonable range
        dt = np.clip(dt, 0.001, 1.0)
        
        return dt