"""
Physics solver integration for flux-based simulation.

This module handles momentum updates and viscous damping.
Gravity and pressure solving are handled by dedicated solver modules.
"""

import numpy as np
from typing import Tuple, Optional
from state import FluxState
from pressure_solver import PressureSolver
from pressure_solver_allspeed import AllSpeedPressureSolver
from pressure_solver_lowmach import LowMachPressureSolver


class FluxPhysics:
    """Handles momentum physics for flux simulation."""
    
    def __init__(self, state: FluxState, implicit_gravity: bool = True, 
                 solver_type: str = "lowmach"):
        """
        Initialize physics integration.
        
        Args:
            state: FluxState instance to operate on
            implicit_gravity: Use implicit gravity treatment for stability (default: True)
            solver_type: Pressure solver type ("standard", "allspeed", "lowmach")
        """
        self.state = state
        self.G = 6.67430e-11  # Gravitational constant
        self.pressure_solver = None  # Lazy init
        self.implicit_gravity = implicit_gravity
        self.solver_type = solver_type
        
        # For backward compatibility
        self.use_allspeed = (solver_type == "allspeed")
        
    def update_momentum(self, gx: np.ndarray, gy: np.ndarray, dt: float):
        """Update momentum using MAC projection scheme.

        Explicit gravity mode:
        1. Predictor: v* = v + dt*(advection + gravity + viscosity)
        2. Projection: solve for φ and correct velocities to be divergence-free
        3. Pressure update: P = P + φ
        
        Implicit gravity mode (default):
        1. Predictor: v* = v + dt*(advection + viscosity)
        2. Projection with gravity: solve for φ with gravity source term
        3. Apply gravity correction: v = v* + dt*g - dt*β∇φ
        4. Pressure update: P = P + φ
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
        
        # 2. Apply convective forces
        st.velocity_x += dt * ax_conv
        st.velocity_y += dt * ay_conv
        
        # Apply gravity explicitly only if not using implicit gravity
        if not self.implicit_gravity:
            st.velocity_x += dt * gx
            st.velocity_y += dt * gy
        
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
            if self.solver_type == "lowmach":
                self.pressure_solver = LowMachPressureSolver(st)
            elif self.solver_type == "allspeed":
                self.pressure_solver = AllSpeedPressureSolver(st)
            else:
                self.pressure_solver = PressureSolver(st)
        
        # Pass gravity fields when using implicit gravity
        if self.implicit_gravity:
            phi = self.pressure_solver.project_velocity(dt, gx=gx, gy=gy, 
                                                      bc_type="neumann", 
                                                      implicit_gravity=True)
            # Note: Gravity is handled inside project_velocity for implicit scheme
            # The projection solver accounts for gravity in the source term
        else:
            # Standard projection without gravity source term
            phi = self.pressure_solver.project_velocity(dt, bc_type="neumann", 
                                                      implicit_gravity=False)
        
        # Print convergence info if monitoring is enabled
        if self.pressure_solver.enable_monitoring and self.pressure_solver.last_iterations > 0:
            print(f"  Pressure solver: {self.pressure_solver.last_iterations} iterations, "
                  f"residual={self.pressure_solver.last_residual:.2e}")
        
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
        st = self.state
        dx2 = st.dx * st.dx
        
        # Standard material viscosity
        damping = 1.0 - st.viscosity * dt / dx2
        damping = np.clip(damping, 0.0, 1.0)  # Ensure stability
        
        # When using all-speed method, add extra damping in low-density regions
        if self.use_allspeed:
            # In space/low-density regions, add artificial viscosity
            # This represents molecular viscosity in rarified gases
            space_mask = st.density < 1.0  # kg/m³
            if np.any(space_mask):
                # Strong damping in space to prevent velocity runaway
                space_damping = np.exp(-dt / 0.1)  # 0.1 second damping time
                damping[space_mask] = np.minimum(damping[space_mask], space_damping)
        
        st.velocity_x *= damping
        st.velocity_y *= damping
        
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
        
        # Advection CFL limit
        if vmax > 0:
            dt_advection = 0.5 * dx / vmax
        else:
            dt_advection = float('inf')
        
        # Gravity acceleration limit
        gmax = max(np.max(np.abs(self.state.gravity_x)), 
                   np.max(np.abs(self.state.gravity_y)))
        
        if self.implicit_gravity:
            # Implicit gravity is more stable but still has limits
            # Allow larger velocity changes but not infinite
            if gmax > 0:
                # Allow larger velocity change for implicit method
                max_dv = 2.0 * dx  # Allow up to 2 cell sizes per timestep
                dt_gravity = max_dv / gmax
            else:
                dt_gravity = float('inf')
        else:
            # Explicit gravity needs strict limits
            if gmax > 0:
                # Limit velocity change to fraction of dx per timestep
                max_dv = 0.1 * dx  # Allow at most 10% of cell size velocity change
                dt_gravity = max_dv / gmax
            else:
                dt_gravity = float('inf')
        
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
            if alpha_max > 0:
                dt_diffusion = 0.25 * dx * dx / alpha_max
            else:
                dt_diffusion = float('inf')
        else:
            dt_diffusion = float('inf')
        
        # Take minimum of all limits with safety factor
        dt = 0.8 * min(dt_advection, dt_gravity, dt_diffusion)
        
        # Clamp to reasonable range
        # For implicit gravity, use more conservative max timestep
        if self.implicit_gravity:
            max_dt = 1000.0  # 1000 seconds max for stability
        else:
            max_dt = 0.1 * 365.25 * 24 * 3600  # 0.1 year for explicit
        dt = np.clip(dt, 0.001, max_dt)
        
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
    
    def enable_solver_monitoring(self, enable: bool = True):
        """Enable or disable convergence monitoring for the pressure solver."""
        if self.pressure_solver:
            self.pressure_solver.enable_monitoring = enable
    
    def get_solver_stats(self):
        """Get convergence statistics from the pressure solver."""
        if self.pressure_solver:
            return self.pressure_solver.get_convergence_stats()
        return None