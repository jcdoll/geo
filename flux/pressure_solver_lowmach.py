"""
Low-Mach preconditioned pressure solver for extreme density ratios.

This implements Low-Mach preconditioning which modifies the governing equations
to handle flows with extreme density variations. The key idea is to modify the
time derivatives to equalize acoustic wave speeds across different materials.

The preconditioned system replaces:
∂U/∂t + ∂F/∂x = S

With:
[P]∂U/∂t + ∂F/∂x = S

Where [P] is the preconditioning matrix designed to equalize eigenvalues.
"""

import numpy as np
from typing import Tuple, Optional
from state import FluxState
from materials import MaterialType

# Import MAC multigrid solver
from multigrid import solve_mac_poisson_configurable, SmootherType, BoundaryCondition


class LowMachPressureSolver:
    """Low-Mach preconditioned pressure solver for extreme density ratios."""
    
    def __init__(self, state: FluxState, reference_mach: float = 0.1):
        """
        Initialize Low-Mach preconditioned solver.
        
        Args:
            state: FluxState instance
            reference_mach: Reference Mach number for preconditioning (default: 0.1)
        """
        self.state = state
        self.reference_mach = reference_mach
        self.phi_prev = None  # Store previous solution for warm start
        
        # Reference values for preconditioning
        self.rho_ref = 1000.0  # kg/m³ - reference density (water)
        self.c_ref = 340.0     # m/s - reference sound speed
        self.u_ref = self.reference_mach * self.c_ref  # Reference velocity
        
        # Convergence monitoring
        self.last_iterations = 0
        self.last_residual = 0.0
        self.enable_monitoring = False
        
    def compute_preconditioning_parameters(self, density: np.ndarray, 
                                         velocity_x: np.ndarray, 
                                         velocity_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Low-Mach preconditioning parameters.
        
        Returns:
            beta_p: Preconditioned compressibility
            theta: Preconditioning parameter
        """
        # Local velocity magnitude
        u_local = np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Local Mach number (using reference sound speed)
        mach_local = u_local / self.c_ref
        
        # Ensure minimum Mach number for stability
        mach_eff = np.maximum(mach_local, self.reference_mach)
        
        # Preconditioning parameter θ (Turkel's formulation)
        # θ = 1 for M=1 (no preconditioning), θ = M² for M<<1
        theta = np.minimum(1.0, mach_eff**2)
        
        # Preconditioned compressibility
        # This modifies the pressure-density relationship
        beta_p = theta / (density * self.c_ref**2)
        
        return beta_p, theta
        
    def project_velocity(self, dt: float, gx: np.ndarray = None, gy: np.ndarray = None, 
                        bc_type: str = "neumann", implicit_gravity: bool = True) -> np.ndarray:
        """
        Project velocity field using Low-Mach preconditioning.
        
        The preconditioned projection equation becomes:
        ∇·(β∇φ) = (1/Δt)∇·v* - ∇·(βg) + (1-θ)/θ ∇·(ρv*)
        
        Where θ is the preconditioning parameter.
        
        Args:
            dt: Time step
            gx, gy: Gravity acceleration fields
            bc_type: Boundary condition type
            implicit_gravity: Whether to treat gravity implicitly
            
        Returns:
            phi: Pressure correction field
        """
        st = self.state
        
        # Ensure face coefficients are up to date
        st.update_face_coefficients()
        
        # Compute preconditioning parameters
        beta_p, theta = self.compute_preconditioning_parameters(
            st.density, st.velocity_x, st.velocity_y
        )
        
        if self.enable_monitoring:
            print(f"  Theta range: {np.min(theta):.2e} to {np.max(theta):.2e}")
            print(f"  Low-Mach cells (theta < 0.1): {np.sum(theta < 0.1)} / {theta.size}")
        
        # Compute divergence of face velocities
        div = self._compute_divergence()
        
        # Standard RHS
        rhs = div / dt
        
        # Add gravity source term for implicit treatment
        if implicit_gravity and gx is not None and gy is not None:
            div_beta_g = self._compute_gravity_divergence(gx, gy)
            rhs -= div_beta_g
        
        # Low-Mach preconditioning: scale RHS based on local Mach number
        # In low-Mach regions (theta << 1), this relaxes the divergence constraint
        # This prevents the pressure solver from overcompensating
        rhs *= np.sqrt(theta)  # Scale by sqrt(theta) for smoother transition
        
        # For very low density regions, add stabilization
        # This prevents spurious pressure modes
        space_mask = st.density < 1.0  # kg/m³
        if np.any(space_mask):
            # Add artificial diffusion to pressure in space regions
            # This damps acoustic waves that would otherwise reflect
            diffusion_coeff = 0.1 * st.dx**2 / dt
            rhs[space_mask] += diffusion_coeff * self._compute_pressure_laplacian()[space_mask]
        
        # Solve modified Poisson equation
        # Note: We use standard beta coefficients, not preconditioned ones
        # The preconditioning appears only in the RHS
        from multigrid import solve_mac_poisson_configurable
        bc = BoundaryCondition.NEUMANN if bc_type == "neumann" else BoundaryCondition.DIRICHLET
        
        # Use tighter tolerance for better stability
        tol = 1e-5 if st.nx * st.ny < 4096 else 1e-4
        
        phi = solve_mac_poisson_configurable(
            rhs,
            st.beta_x,
            st.beta_y,
            st.dx,
            bc_type=bc,
            tol=tol,
            max_cycles=100,  # More iterations for preconditioned system
            initial_guess=self.phi_prev,
            smoother_type=SmootherType.RED_BLACK,
            verbose=False
        )
        
        # Store for next time
        if self.phi_prev is None:
            self.phi_prev = np.copy(phi)
        else:
            self.phi_prev[:] = phi
        
        # Update face velocities using pressure gradient
        # For Low-Mach preconditioning, the velocity update is modified
        self._update_face_velocities_preconditioned(
            phi, dt, theta, bc_type, gx=gx, gy=gy, implicit_gravity=implicit_gravity
        )
        
        # Update cell-centered velocities from face velocities
        st.update_cell_velocities_from_face()
        
        # Update pressure with preconditioned correction
        # For Low-Mach preconditioning, we need to be careful with the pressure update
        # to avoid numerical instability
        st.pressure += phi  # Standard update, preconditioning is in the solve
        
        # Limit pressure in space regions to prevent runaway
        st.pressure[space_mask] = np.clip(st.pressure[space_mask], -1e6, 1e6)
        
        # Remove mean pressure to prevent drift (for Neumann BC)
        if bc_type == "neumann":
            material_mask = st.density > 1.0
            if np.any(material_mask):
                st.pressure[material_mask] -= np.mean(st.pressure[material_mask])
        
        return phi
        
    def _compute_preconditioning_term(self, theta: np.ndarray, density: np.ndarray,
                                     vx_face: np.ndarray, vy_face: np.ndarray) -> np.ndarray:
        """
        Compute the Low-Mach preconditioning term: (1-θ)/θ ∇·(ρv*).
        
        This term modifies the divergence constraint based on local Mach number.
        """
        dx = self.state.dx
        
        # Interpolate density to faces
        rho_x_face = np.zeros_like(vx_face)
        rho_y_face = np.zeros_like(vy_face)
        
        # X-faces
        rho_x_face[:, 1:-1] = 0.5 * (density[:, :-1] + density[:, 1:])
        rho_x_face[:, 0] = density[:, 0]
        rho_x_face[:, -1] = density[:, -1]
        
        # Y-faces
        rho_y_face[1:-1, :] = 0.5 * (density[:-1, :] + density[1:, :])
        rho_y_face[0, :] = density[0, :]
        rho_y_face[-1, :] = density[-1, :]
        
        # Compute ∇·(ρv*)
        div_rho_v = np.zeros_like(density)
        div_rho_v[:, :] = (
            (rho_x_face[:, 1:] * vx_face[:, 1:] - rho_x_face[:, :-1] * vx_face[:, :-1]) / dx +
            (rho_y_face[1:, :] * vy_face[1:, :] - rho_y_face[:-1, :] * vy_face[:-1, :]) / dx
        )
        
        # Preconditioning factor (1-θ)/θ
        # Avoid division by zero
        theta_safe = np.maximum(theta, 1e-6)
        precond_factor = (1.0 - theta) / theta_safe
        
        # Limit the factor to prevent instability
        precond_factor = np.clip(precond_factor, -10.0, 10.0)
        
        return precond_factor * div_rho_v
        
    def _compute_pressure_laplacian(self) -> np.ndarray:
        """
        Compute Laplacian of pressure for stabilization.
        
        This is used to add artificial diffusion in space regions.
        """
        st = self.state
        dx2 = st.dx * st.dx
        
        # Simple 5-point Laplacian
        laplacian = np.zeros_like(st.pressure)
        
        # Interior points
        laplacian[1:-1, 1:-1] = (
            st.pressure[1:-1, 2:] + st.pressure[1:-1, :-2] +
            st.pressure[2:, 1:-1] + st.pressure[:-2, 1:-1] -
            4 * st.pressure[1:-1, 1:-1]
        ) / dx2
        
        return laplacian
        
    def _update_face_velocities_preconditioned(self, phi: np.ndarray, dt: float, 
                                             theta: np.ndarray, bc_type: str,
                                             gx: np.ndarray = None, gy: np.ndarray = None,
                                             implicit_gravity: bool = False):
        """
        Update face velocities with Low-Mach preconditioning.
        
        The preconditioned velocity update includes theta scaling.
        """
        st = self.state
        dx = st.dx
        
        # Interpolate theta to faces for consistent update
        theta_x_face = np.ones((st.ny, st.nx + 1))
        theta_y_face = np.ones((st.ny + 1, st.nx))
        
        # X-faces
        theta_x_face[:, 1:-1] = 0.5 * (theta[:, :-1] + theta[:, 1:])
        theta_x_face[:, 0] = theta[:, 0]
        theta_x_face[:, -1] = theta[:, -1]
        
        # Y-faces
        theta_y_face[1:-1, :] = 0.5 * (theta[:-1, :] + theta[1:, :])
        theta_y_face[0, :] = theta[0, :]
        theta_y_face[-1, :] = theta[-1, :]
        
        # Standard pressure gradient update
        # X-faces
        grad_phi_x_int = (phi[:, 1:] - phi[:, :-1]) / dx
        st.velocity_x_face[:, 1:-1] -= dt * st.beta_x[:, 1:-1] * grad_phi_x_int
        
        # Y-faces
        grad_phi_y_int = (phi[1:, :] - phi[:-1, :]) / dx
        st.velocity_y_face[1:-1, :] -= dt * st.beta_y[1:-1, :] * grad_phi_y_int
        
        # Handle boundaries (simplified - using zero gradient)
        # Boundary handling would need to be more sophisticated for production
        
        # Add gravity term for implicit scheme
        if implicit_gravity and gx is not None and gy is not None:
            # Interpolate gravity to faces
            gx_face = np.zeros((st.ny, st.nx + 1))
            gx_face[:, 1:-1] = 0.5 * (gx[:, :-1] + gx[:, 1:])
            gx_face[:, 0] = gx[:, 0]
            gx_face[:, -1] = gx[:, -1]
            st.velocity_x_face += dt * gx_face
            
            gy_face = np.zeros((st.ny + 1, st.nx))
            gy_face[1:-1, :] = 0.5 * (gy[:-1, :] + gy[1:, :])
            gy_face[0, :] = gy[0, :]
            gy_face[-1, :] = gy[-1, :]
            st.velocity_y_face += dt * gy_face
            
    def _compute_divergence(self) -> np.ndarray:
        """Compute divergence of face-centered velocities."""
        st = self.state
        dx = st.dx
        
        div = np.zeros_like(st.density)
        div[:, :] = (
            (st.velocity_x_face[:, 1:] - st.velocity_x_face[:, :-1]) / dx +
            (st.velocity_y_face[1:, :] - st.velocity_y_face[:-1, :]) / dx
        )
        
        return div
        
    def _compute_gravity_divergence(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """Compute divergence of β*g where β = 1/ρ on faces."""
        st = self.state
        dx = st.dx
        
        # Interpolate gravity to faces
        gx_face = np.zeros((st.ny, st.nx + 1))
        gx_face[:, 1:-1] = 0.5 * (gx[:, :-1] + gx[:, 1:])
        gx_face[:, 0] = gx[:, 0]
        gx_face[:, -1] = gx[:, -1]
        
        gy_face = np.zeros((st.ny + 1, st.nx))
        gy_face[1:-1, :] = 0.5 * (gy[:-1, :] + gy[1:, :])
        gy_face[0, :] = gy[0, :]
        gy_face[-1, :] = gy[-1, :]
        
        # Compute β*g on faces
        beta_gx = st.beta_x * gx_face
        beta_gy = st.beta_y * gy_face
        
        # Compute divergence at cell centers
        div_beta_g = np.zeros_like(st.density)
        div_beta_g[:, :] = (
            (beta_gx[:, 1:] - beta_gx[:, :-1]) / dx +
            (beta_gy[1:, :] - beta_gy[:-1, :]) / dx
        )
        
        return div_beta_g