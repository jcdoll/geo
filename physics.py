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
        
        # Check for NaN in accelerations
        if np.any(np.isnan(ax_conv)) or np.any(np.isnan(ay_conv)):
            print(f"WARNING: NaN in convective acceleration!")
            print(f"  ax_conv has {np.sum(np.isnan(ax_conv))} NaN values")
            print(f"  ay_conv has {np.sum(np.isnan(ay_conv))} NaN values")
        
        # Check for NaN in gravity
        if np.any(np.isnan(gx)) or np.any(np.isnan(gy)):
            print(f"WARNING: NaN in gravity field!")
            print(f"  gx has {np.sum(np.isnan(gx))} NaN values")
            print(f"  gy has {np.sum(np.isnan(gy))} NaN values")
        
        # 2. Apply convective and gravity forces
        st.velocity_x += dt * (ax_conv + gx)
        st.velocity_y += dt * (ay_conv + gy)
        
        # Debug: Check if gravity is being applied
        if np.any(gy > 0):
            max_gy = np.max(gy)
            max_dvy = dt * max_gy
            if max_dvy > 0.1:  # Significant velocity change
                print(f"DEBUG: Applied gravity gy_max={max_gy:.2f}, dt={dt:.3f}, max velocity change={max_dvy:.2f} m/s")
        
        # 3. Apply viscous damping
        self.apply_viscous_damping(dt)
        
        # 4. Update face velocities with v* for projection
        st.update_face_velocities_from_cell()
        
        # Debug check before projection
        if np.any(np.isnan(st.velocity_x)) or np.any(np.isnan(st.velocity_y)):
            print(f"WARNING: NaN in velocities before projection!")
            print(f"  vx: min={np.nanmin(st.velocity_x):.2e}, max={np.nanmax(st.velocity_x):.2e}")
            print(f"  vy: min={np.nanmin(st.velocity_y):.2e}, max={np.nanmax(st.velocity_y):.2e}")
            print(f"  dt={dt:.3e}, max(gx)={np.max(np.abs(gx)):.2e}, max(gy)={np.max(np.abs(gy)):.2e}")
            print(f"  max(ax_conv)={np.max(np.abs(ax_conv)):.2e}, max(ay_conv)={np.max(np.abs(ay_conv)):.2e}")
        
        # Don't zero velocities in space - let materials fall through space!
        # Space should have very low viscosity and allow free movement

        # ------------------------------------------------------------------
        # PROJECTION STAGE: Make velocity field divergence-free
        # ------------------------------------------------------------------
        if self.pressure_solver is None:
            from pressure_solver import PressureSolver
            self.pressure_solver = PressureSolver(st)
        
        phi = self.pressure_solver.project_velocity(dt, gx=gx, gy=gy, bc_type="neumann")
        
        # Note: project_velocity updates face velocities and then 
        # calls update_cell_velocities_from_face() internally
        
        # Debug: Check velocities after projection
        if np.any(gy > 0):
            max_vy_after = np.max(np.abs(st.velocity_y))
            if max_vy_after < 0.1:
                print(f"DEBUG: Velocities zeroed by pressure projection! max_vy={max_vy_after:.4f}")
        
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
            # Use float64 to avoid overflow in intermediate calculations
            k = self.state.thermal_conductivity[mass_mask].astype(np.float64)
            rho = self.state.density[mass_mask].astype(np.float64)
            cp = self.state.specific_heat[mass_mask].astype(np.float64)
            alpha_masked = k / (rho * cp + 1e-10)
            alpha_max = float(np.max(alpha_masked))
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
        
        # Check for NaN velocities before computation
        if np.any(np.isnan(vx)) or np.any(np.isnan(vy)):
            print("WARNING: NaN velocities detected in convective acceleration")
            print(f"  vx: min={np.nanmin(vx):.2e}, max={np.nanmax(vx):.2e}, nan_count={np.sum(np.isnan(vx))}")
            print(f"  vy: min={np.nanmin(vy):.2e}, max={np.nanmax(vy):.2e}, nan_count={np.sum(np.isnan(vy))}")
            # Return zero acceleration to prevent further propagation
            return np.zeros_like(vx), np.zeros_like(vy)

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