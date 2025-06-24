"""
Physics solver integration for flux-based simulation.

This module handles momentum updates and viscous damping.
Gravity and pressure solving are handled by dedicated solver modules.
"""

import numpy as np
from typing import Tuple, Optional
from state import FluxState


class FluxPhysics:
    """Handles momentum physics for flux simulation."""
    
    def __init__(self, state: FluxState):
        """
        Initialize physics integration.
        
        Args:
            state: FluxState instance to operate on
        """
        self.state = state
        self.G = 6.67430e-11  # Gravitational constant
        self.pressure_solver = None  # Lazy init
        
    def update_momentum(self, gx: np.ndarray, gy: np.ndarray, dt: float):
        """Update momentum using MAC projection scheme.

        1. Predictor: v* = v + dt*(advection + gravity + viscosity)
        2. Projection: solve for φ and correct velocities to be divergence-free
        3. Pressure update: P = P + φ
        """

        st = self.state

        # Update face coefficients (β = 1/ρ with harmonic averaging)
        st.update_face_coefficients()
        
        # Convert cell velocities to face velocities for MAC grid
        st.update_face_velocities_from_cell()

        # ------------------------------------------------------------------
        # PREDICTOR STAGE: Advance face velocities with explicit forces
        # ------------------------------------------------------------------
        
        # 1. Convective acceleration on cell centers (then interpolate to faces)
        ax_conv, ay_conv = self._compute_convective_acceleration()
        
        # 2. Apply convective and gravity forces
        st.velocity_x += dt * (ax_conv + gx)
        st.velocity_y += dt * (ay_conv + gy)
        
        # 3. Apply viscous damping
        self.apply_viscous_damping(dt)
        
        # 4. Update face velocities with v* for projection
        st.update_face_velocities_from_cell()
        
        # Zero velocities in space regions before projection
        space_mask = st.density < 1.0
        st.velocity_x[space_mask] = 0.0
        st.velocity_y[space_mask] = 0.0

        # ------------------------------------------------------------------
        # PROJECTION STAGE: Make velocity field divergence-free
        # ------------------------------------------------------------------
        if self.pressure_solver is None:
            from pressure_solver import PressureSolver
            self.pressure_solver = PressureSolver(st)
        
        phi = self.pressure_solver.project_velocity(dt, gx=gx, gy=gy, bc_type="neumann")
        
        # Note: project_velocity updates face velocities and then 
        # calls update_cell_velocities_from_face() internally
        
    def apply_viscous_damping(self, dt: float):
        """
        Apply material-dependent viscous damping.
        
        This is a simplified implementation that applies a damping factor
        rather than full Laplacian diffusion (ν∇²v). For geological timescales
        and high-viscosity materials, this approximation is adequate.
        
        Args:
            dt: Time step
        """
        # Viscosity provides resistance to motion
        # v_new = v_old * (1 - viscosity * dt / dx²)
        # This is a first-order approximation to viscous diffusion
        dx2 = self.state.dx * self.state.dx
        damping = 1.0 - self.state.viscosity * dt / dx2
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
        
        # Gravity wave speed (for incompressible flow)
        g = 9.81  # m/s^2
        c_grav = np.sqrt(g * dx)  # Local gravity wave speed
        
        # Thermal diffusion limit
        # Only consider cells with significant mass (not space)
        mass_mask = self.state.density > 1.0  # kg/m³
        if np.any(mass_mask):
            alpha_masked = self.state.thermal_conductivity[mass_mask] / \
                          (self.state.density[mass_mask] * self.state.specific_heat[mass_mask] + 1e-10)
            alpha_max = np.max(alpha_masked)
        else:
            alpha_max = 0.0
        
        # CFL conditions
        # Use maximum of advection velocity and gravity wave speed
        c_max = max(vmax, c_grav)
        if c_max > 0:
            dt_advection = 0.5 * dx / c_max
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_convective_acceleration(self):
        """Compute convective acceleration −(v·∇)v using first-order upwind.

        Returns
        -------
        ax, ay : ndarray
            Accelerations to be added to dv/dt.
        """
        st = self.state
        dx = st.dx

        vx = st.velocity_x
        vy = st.velocity_y

        # Compute gradients with upwind scheme (vectorised)
        dvx_dx = np.zeros_like(vx)
        dvx_dy = np.zeros_like(vx)
        dvy_dx = np.zeros_like(vy)
        dvy_dy = np.zeros_like(vy)

        # X-derivatives
        dvx_dx[:, 1:] = (vx[:, 1:] - vx[:, :-1]) / dx
        dvy_dx[:, 1:] = (vy[:, 1:] - vy[:, :-1]) / dx

        # Y-derivatives
        dvx_dy[1:, :] = (vx[1:, :] - vx[:-1, :]) / dx
        dvy_dy[1:, :] = (vy[1:, :] - vy[:-1, :]) / dx

        ax = -(vx * dvx_dx + vy * dvx_dy)
        ay = -(vx * dvy_dx + vy * dvy_dy)

        return ax, ay