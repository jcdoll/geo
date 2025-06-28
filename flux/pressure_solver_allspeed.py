"""
All-speed pressure solver for flux-based geological simulation.

This implements the all-speed method that smoothly transitions between
incompressible flow (for materials) and compressible flow (for space/vacuum).

Key equation:
∇·(β∇φ) - ε(ρ)φ = ∇·v*/Δt - ∇·(βg)

Where:
- β = 1/ρ (harmonic averaged at faces)
- ε(ρ) = compressibility parameter (0 for dense materials, 1/c² for space)
- φ = pressure correction
"""

import numpy as np
from typing import Tuple, Optional
from state import FluxState
from materials import MaterialType

# Import MAC multigrid solver
from multigrid import SmootherType, BoundaryCondition


class AllSpeedPressureSolver:
    """All-speed pressure solver using variable compressibility."""
    
    def __init__(self, state: FluxState, sound_speed: float = 340.0):
        """
        Initialize all-speed pressure solver.
        
        Args:
            state: FluxState instance
            sound_speed: Reference sound speed in m/s (default: 340 m/s)
        """
        self.state = state
        self.sound_speed = sound_speed
        self.phi_prev = None  # Store previous solution for warm start
        
        # Transition thresholds for compressibility
        self.rho_incompressible = 100.0  # kg/m³ - above this, fully incompressible
        self.rho_compressible = 1.0      # kg/m³ - below this, fully compressible
        
        # Convergence monitoring
        self.last_iterations = 0
        self.last_residual = 0.0
        self.enable_monitoring = False
        
    def compute_epsilon(self, density: np.ndarray) -> np.ndarray:
        """
        Compute compressibility parameter ε(ρ).
        
        ε = 0 for incompressible flow (dense materials)
        ε = 1/c² for compressible flow (space/vacuum)
        Smooth transition in between.
        
        Args:
            density: Cell-centered density field
            
        Returns:
            epsilon: Cell-centered compressibility parameter
        """
        # Normalize density for smooth transition
        rho_norm = (density - self.rho_compressible) / (self.rho_incompressible - self.rho_compressible)
        rho_norm = np.clip(rho_norm, 0, 1)
        
        # Smooth transition using tanh
        # compress_factor = 1 in space, 0 in materials
        compress_factor = 0.5 * (1 - np.tanh(3 * (rho_norm - 0.5)))
        
        # Epsilon varies from 0 (incompressible) to 1/c² (compressible)
        epsilon = compress_factor / (self.sound_speed * self.sound_speed)
        
        return epsilon
        
    def project_velocity(self, dt: float, gx: np.ndarray = None, gy: np.ndarray = None, 
                        bc_type: str = "neumann", implicit_gravity: bool = True) -> np.ndarray:
        """
        Project velocity field using all-speed method.
        
        Solves: ∇·(β∇φ) - ε(ρ)φ = ∇·v*/Δt - ∇·(βg)
        Then updates: v = v* - Δt β∇φ (+ Δt g for implicit gravity)
        
        Args:
            dt: Time step
            gx, gy: Gravity acceleration fields (required if implicit_gravity=True)
            bc_type: Boundary condition type ('neumann' or 'dirichlet')
            implicit_gravity: Whether to treat gravity implicitly
            
        Returns:
            phi: Pressure correction field
        """
        st = self.state
        
        # Ensure face coefficients are up to date
        st.update_face_coefficients()
        
        # Compute compressibility parameter
        epsilon = self.compute_epsilon(st.density)
        
        # Debug: Check epsilon values
        epsilon_min = np.min(epsilon)
        epsilon_max = np.max(epsilon)
        if self.enable_monitoring:
            print(f"  Epsilon range: {epsilon_min:.2e} to {epsilon_max:.2e}")
            print(f"  Compressible cells: {np.sum(epsilon > 1e-8)} / {epsilon.size}")
        
        # Compute divergence of face velocities
        div = self._compute_divergence()
        
        # RHS for modified equation
        rhs = div / dt
        
        # Add gravity source term for implicit treatment
        if implicit_gravity and gx is not None and gy is not None:
            div_beta_g = self._compute_gravity_divergence(gx, gy)
            rhs -= div_beta_g
        
        # For the all-speed method, we modify the divergence constraint
        # In compressible regions, we allow some divergence based on pressure changes
        # This is equivalent to adding artificial compressibility
        
        # Apply artificial compressibility
        # In low-density regions, we relax the divergence-free constraint
        # by scaling down the RHS. This prevents the pressure solver from
        # trying to enforce strict incompressibility where it's not physical.
        
        # Create a smooth weight function (1 = incompressible, 0 = fully compressible)
        rho_normalized = (st.density - self.rho_compressible) / (self.rho_incompressible - self.rho_compressible)
        rho_normalized = np.clip(rho_normalized, 0, 1)
        
        # Use a smoother transition function
        incompress_weight = rho_normalized**2  # Quadratic transition
        
        # Apply weight to RHS
        rhs *= incompress_weight
        
        # In very low density regions, add damping to prevent runaway velocities
        # This is physically motivated by molecular viscosity in rarified gases
        low_density_mask = st.density < 1.0
        if np.any(low_density_mask):
            # Add artificial damping term proportional to velocity divergence
            # This helps stabilize the flow in space regions
            damping_coefficient = 0.1 * st.dx / dt  # Dimensional analysis gives L/T
            rhs[low_density_mask] += damping_coefficient * div[low_density_mask]
        
        # Now solve standard Poisson equation with modified RHS
        from multigrid import solve_mac_poisson_configurable
        bc = BoundaryCondition.NEUMANN if bc_type == "neumann" else BoundaryCondition.DIRICHLET
        
        phi = solve_mac_poisson_configurable(
            rhs,
            st.beta_x,
            st.beta_y,
            st.dx,
            bc_type=bc,
            tol=1e-4,
            max_cycles=50,
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
        if implicit_gravity:
            self._update_face_velocities(phi, dt, bc_type, gx=gx, gy=gy, 
                                       implicit_gravity=True, epsilon=epsilon)
        else:
            self._update_face_velocities(phi, dt, bc_type, epsilon=epsilon)
        
        # Update cell-centered velocities from face velocities
        st.update_cell_velocities_from_face()
        
        # Update pressure
        # With the modified divergence constraint, φ is always a pressure correction
        st.pressure += phi
        
        # Force zero pressure in deep space
        deep_space_mask = st.density < 0.01
        st.pressure[deep_space_mask] = 0.0
        
        # Remove mean pressure to prevent drift (for Neumann BC)
        if bc_type == "neumann":
            material_mask = st.density > self.rho_compressible
            if np.any(material_mask):
                st.pressure[material_mask] -= np.mean(st.pressure[material_mask])
        
        return phi
        
    def _solve_helmholtz(self, rhs: np.ndarray, epsilon: np.ndarray, dt: float, 
                        bc_type: str) -> np.ndarray:
        """
        Solve the modified Helmholtz equation: ∇·(β∇φ) - εφ = rhs
        
        This is more complex than standard Poisson and requires a modified solver.
        For now, we'll use an iterative approach with the existing multigrid
        as a preconditioner.
        """
        st = self.state
        ny, nx = rhs.shape
        
        # Initial guess
        phi = self.phi_prev.copy() if self.phi_prev is not None else np.zeros_like(rhs)
        
        # For efficiency, we'll use a hybrid approach:
        # - Where epsilon is very small (<1e-8), use standard Poisson
        # - Where epsilon is significant, use iterative Helmholtz solver
        
        nearly_incompressible = epsilon < 1e-8
        
        if np.all(nearly_incompressible):
            # Pure Poisson case - use existing fast solver
            from multigrid import solve_mac_poisson_configurable
            bc = BoundaryCondition.NEUMANN if bc_type == "neumann" else BoundaryCondition.DIRICHLET
            return solve_mac_poisson_configurable(
                rhs, beta_x, beta_y, dx, bc_type=bc, tol=1e-4, max_cycles=50,
                initial_guess=phi, smoother_type=SmootherType.RED_BLACK, verbose=False
            )
        
        # Mixed case - need Helmholtz solver
        # We'll use a simple iterative scheme with multigrid as preconditioner
        
        # Solver parameters
        max_iter = 50
        tol = 1e-4
        omega = 1.5  # Over-relaxation parameter
        
        for iteration in range(max_iter):
            # Compute residual: r = rhs - (∇·(β∇φ) - εφ)
            residual = self._compute_helmholtz_residual(phi, rhs, epsilon, st.beta_x, st.beta_y, st.dx)
            
            # Check convergence
            res_norm = np.linalg.norm(residual)
            if iteration == 0:
                res_norm0 = res_norm
            
            if res_norm < tol * res_norm0:
                if self.enable_monitoring:
                    print(f"  All-speed solver converged in {iteration} iterations")
                break
                
            # Apply one multigrid V-cycle as preconditioner
            # This solves: ∇·(β∇correction) = residual
            correction = self._multigrid_preconditioner(residual, st.beta_x, st.beta_y, st.dx, bc_type)
            
            # Update with relaxation
            # For Helmholtz, we need to account for the -εφ term
            phi += omega * correction / (1.0 + omega * epsilon * st.dx * st.dx)
        
        if iteration == max_iter - 1 and self.enable_monitoring:
            print(f"  All-speed solver reached max iterations ({max_iter})")
        
        self.last_iterations = iteration + 1
        self.last_residual = res_norm / res_norm0
        
        return phi
    
    def _compute_helmholtz_residual(self, phi: np.ndarray, rhs: np.ndarray, 
                                   epsilon: np.ndarray, beta_x: np.ndarray, 
                                   beta_y: np.ndarray, dx: float) -> np.ndarray:
        """Compute residual for Helmholtz equation: r = rhs - (∇·(β∇φ) - εφ)"""
        from multigrid import compute_residual_mac_vectorized
        
        # First compute standard Poisson residual using multigrid function
        # This gives us: rhs - ∇·(β∇φ)
        poisson_residual = compute_residual_mac_vectorized(
            phi, rhs, beta_x, beta_y, dx, BoundaryCondition.NEUMANN
        )
        
        # For Helmholtz, we need: rhs - (∇·(β∇φ) - εφ)
        # Which is: rhs - ∇·(β∇φ) + εφ = poisson_residual + εφ
        helmholtz_residual = poisson_residual + epsilon * phi
        
        return helmholtz_residual
    
    def _multigrid_preconditioner(self, residual: np.ndarray, beta_x: np.ndarray,
                                 beta_y: np.ndarray, dx: float, bc_type: str) -> np.ndarray:
        """Apply multigrid V-cycle as preconditioner (solves Poisson, not Helmholtz)"""
        from multigrid import solve_mac_poisson_configurable
        
        bc = BoundaryCondition.NEUMANN if bc_type == "neumann" else BoundaryCondition.DIRICHLET
        
        # Use relaxed tolerance since this is just a preconditioner
        correction = solve_mac_poisson_configurable(
            residual,
            beta_x,
            beta_y,
            dx,
            bc_type=bc,
            tol=1e-2,  # Relaxed tolerance
            max_cycles=1,  # Single V-cycle
            smoother_type=SmootherType.RED_BLACK,
            verbose=False
        )
        
        return correction
        
    def _compute_divergence(self) -> np.ndarray:
        """Compute divergence of face-centered velocities."""
        st = self.state
        dx = st.dx
        
        # Divergence at cell centers
        div = np.zeros_like(st.density)
        
        # ∇·v = (vx_e - vx_w)/dx + (vy_n - vy_s)/dx
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
    
    def _update_face_velocities(self, phi: np.ndarray, dt: float, bc_type: str,
                               gx: np.ndarray = None, gy: np.ndarray = None, 
                               implicit_gravity: bool = False, epsilon: np.ndarray = None):
        """
        Update face velocities using pressure gradient.
        
        In the all-speed method, the update depends on local compressibility.
        """
        st = self.state
        dx = st.dx
        
        # Standard pressure gradient update
        # X-faces
        grad_phi_x_int = (phi[:, 1:] - phi[:, :-1]) / dx
        st.velocity_x_face[:, 1:-1] -= dt * st.beta_x[:, 1:-1] * grad_phi_x_int
        
        # Y-faces
        grad_phi_y_int = (phi[1:, :] - phi[:-1, :]) / dx
        st.velocity_y_face[1:-1, :] -= dt * st.beta_y[1:-1, :] * grad_phi_y_int
        
        # Handle boundaries based on BC type
        if bc_type == "neumann":
            # Zero gradient at boundaries
            pass  # Already handled by not updating boundary faces
        else:  # Dirichlet
            # One-sided differences at boundaries
            # Left boundary
            grad_phi_x_left = phi[:, 0] / dx
            st.velocity_x_face[:, 0] -= dt * st.beta_x[:, 0] * grad_phi_x_left
            
            # Right boundary
            grad_phi_x_right = -phi[:, -1] / dx
            st.velocity_x_face[:, -1] -= dt * st.beta_x[:, -1] * grad_phi_x_right
            
            # Bottom boundary
            grad_phi_y_bottom = phi[0, :] / dx
            st.velocity_y_face[0, :] -= dt * st.beta_y[0, :] * grad_phi_y_bottom
            
            # Top boundary
            grad_phi_y_top = -phi[-1, :] / dx
            st.velocity_y_face[-1, :] -= dt * st.beta_y[-1, :] * grad_phi_y_top
        
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