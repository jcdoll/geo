"""
Pressure solver for flux-based geological simulation.

This implements the pressure-projection method for incompressible flow:
1. Given intermediate velocity v* (after applying forces)
2. Find pressure φ such that ∇·(β∇φ) = ∇·v*/Δt where β = 1/ρ
3. Update velocity: v_new = v* - Δt β∇φ
4. Result is divergence-free: ∇·v_new = 0

Key features:
- Variable density handled via face-centered β = 1/ρ coefficients
- MAC staggered grid for stability
- Harmonic averaging at material interfaces
"""

import numpy as np
from typing import Tuple, Optional
from state import FluxState
from materials import MaterialType

# Import MAC multigrid solver with configurable smoother
from multigrid import solve_mac_poisson_configurable, SmootherType, BoundaryCondition


class PressureSolver:
    """Multigrid pressure solver for projection method."""
    
    def __init__(self, state: FluxState, smoother_type: SmootherType = SmootherType.JACOBI):
        """
        Initialize pressure solver.
        
        Args:
            state: FluxState instance
            smoother_type: Which smoother to use (RED_BLACK or JACOBI)
        """
        self.state = state
        self.phi_prev = None  # Store previous solution for warm start
        self.smoother_type = smoother_type
        
        # Convergence monitoring
        self.last_iterations = 0
        self.last_residual = 0.0
        self.convergence_history = []
        self.enable_monitoring = False
        
    def project_velocity(self, dt: float, gx: np.ndarray = None, gy: np.ndarray = None, 
                        bc_type: str = "neumann", implicit_gravity: bool = False) -> np.ndarray:
        """Project velocity field to make it divergence-free.
        
        Solves: ∇·(β∇φ) = ∇·v*/Δt - ∇·(βg) where β = 1/ρ
        Then updates: v = v* - Δt β∇φ
        
        When implicit_gravity=True, gravity is included in the projection step
        for improved stability with large density variations.
        
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
        
        # Compute divergence of face velocities
        div = self._compute_divergence()
        
        # RHS for Poisson equation
        rhs = div / dt
        
        # Add gravity source term for implicit treatment
        if implicit_gravity and gx is not None and gy is not None:
            # Compute ∇·(βg) where β = 1/ρ on faces
            div_beta_g = self._compute_gravity_divergence(gx, gy)
            rhs -= div_beta_g
        
        # === SELECTIVE INCOMPRESSIBILITY ENFORCEMENT ===
        # Smoothly transition from incompressible (high density) to compressible (low density)
        # This prevents numerical instability while maintaining physics
        
        # Define transition range
        rho_incompressible = 100.0  # Above this: fully incompressible (kg/m³)
        rho_compressible = 1.0      # Below this: fully compressible (kg/m³)
        
        # Compute smooth weight function (1 = incompressible, 0 = compressible)
        # Using smooth tanh transition
        rho_normalized = (st.density - rho_compressible) / (rho_incompressible - rho_compressible)
        rho_normalized = np.clip(rho_normalized, 0, 1)
        incompressibility_weight = 0.5 * (1 + np.tanh(3 * (rho_normalized - 0.5)))
        
        # Apply weight to RHS - this smoothly relaxes incompressibility in space
        rhs *= incompressibility_weight
        
        # Also apply to gravity divergence term for consistency
        if implicit_gravity and 'div_beta_g' in locals():
            # The gravity term should also be weighted
            # This prevents artificial acceleration in transition regions
            rhs += (1 - incompressibility_weight) * div_beta_g
        
        # Solve variable-coefficient Poisson equation
        # Use very relaxed settings for real-time performance
        grid_size = st.nx * st.ny
        if grid_size > 8192:  # 128x64 or larger
            max_iter = 10   # Very strict iteration limit
            tol = 1e-1      # Accept approximate solutions
        else:
            max_iter = 15   # Reduced from 50
            tol = 5e-2      # Relaxed from 1e-3
            
        # Solve using MAC multigrid with configurable smoother
        bc = BoundaryCondition.NEUMANN if bc_type == "neumann" else BoundaryCondition.DIRICHLET
        
        # Custom solve with monitoring if enabled
        if self.enable_monitoring:
            phi, iterations, final_residual = self._solve_with_monitoring(
                rhs, st.beta_x, st.beta_y, st.dx, bc, tol, max_iter
            )
            self.last_iterations = iterations
            self.last_residual = final_residual
            self.convergence_history.append({
                'iterations': iterations,
                'residual': final_residual,
                'tolerance': tol,
                'converged': final_residual < tol
            })
        else:
            phi = solve_mac_poisson_configurable(
                rhs,
                st.beta_x,
                st.beta_y,
                st.dx,
                bc_type=bc,
                tol=tol,
                max_cycles=max_iter,
                initial_guess=self.phi_prev,
                smoother_type=self.smoother_type,
                verbose=False
            )
        
        # Store for next time
        if self.phi_prev is None:
            self.phi_prev = np.copy(phi)
        else:
            self.phi_prev[:] = phi
        
        # Update face velocities using pressure gradient
        # Pass incompressibility weight for consistent updates
        if implicit_gravity:
            self._update_face_velocities(phi, dt, bc, gx=gx, gy=gy, 
                                       implicit_gravity=True, 
                                       incompress_weight=incompressibility_weight)
        else:
            self._update_face_velocities(phi, dt, bc, 
                                       incompress_weight=incompressibility_weight)
        
        # Update cell-centered velocities from face velocities
        st.update_cell_velocities_from_face()
        
        # REMOVED: Don't zero velocities in space - let materials fall through!
        
        # Accumulate pressure correction
        st.pressure += phi
        
        # Force zero pressure in space regions
        space_mask = st.density < 0.1
        st.pressure[space_mask] = 0.0
        
        # Remove mean pressure to prevent drift (for Neumann BC)
        # Only consider non-space regions for the mean
        if bc_type == "neumann":
            non_space_pressure = st.pressure[~space_mask]
            if len(non_space_pressure) > 0:
                st.pressure[~space_mask] -= np.mean(non_space_pressure)
        
        return phi
        
    def _compute_divergence(self) -> np.ndarray:
        """Compute divergence of face-centered velocities.
        
        Returns:
            div: Cell-centered divergence field
        """
        st = self.state
        dx = st.dx
        
        # Divergence at cell centers
        div = np.zeros_like(st.density)
        
        # ∇·v = (vx_e - vx_w)/dx + (vy_n - vy_s)/dx
        div[:, :] = (
            (st.velocity_x_face[:, 1:] - st.velocity_x_face[:, :-1]) / dx +
            (st.velocity_y_face[1:, :] - st.velocity_y_face[:-1, :]) / dx
        )
        
        # Let space participate in incompressibility - materials need to displace it!
        # Only zero divergence in truly empty regions (no material at all)
        # space_mask = st.density < 0.1
        # div[space_mask] = 0.0
        
        return div
        
    def _update_face_velocities(self, phi: np.ndarray, dt: float, bc_type: BoundaryCondition = BoundaryCondition.NEUMANN,
                               gx: np.ndarray = None, gy: np.ndarray = None, implicit_gravity: bool = False,
                               incompress_weight: np.ndarray = None):
        """Update face velocities using pressure gradient.
        
        Explicit gravity: v_new = v* - dt * β * ∇φ
        Implicit gravity: v_new = v* + dt * g - dt * β * ∇φ
        
        Properly handles both Neumann and Dirichlet boundary conditions.
        
        Args:
            phi: Pressure correction
            dt: Time step
            bc_type: Boundary condition type
            gx, gy: Gravity fields (required if implicit_gravity=True)
            implicit_gravity: Whether to add gravity term
        """
        st = self.state
        dx = st.dx
        
        # If weight provided, interpolate to faces for selective application
        if incompress_weight is not None:
            # Interpolate weight to x-faces
            weight_x = np.ones((st.ny, st.nx + 1))
            weight_x[:, 1:-1] = 0.5 * (incompress_weight[:, :-1] + incompress_weight[:, 1:])
            weight_x[:, 0] = incompress_weight[:, 0]
            weight_x[:, -1] = incompress_weight[:, -1]
            
            # Interpolate weight to y-faces
            weight_y = np.ones((st.ny + 1, st.nx))
            weight_y[1:-1, :] = 0.5 * (incompress_weight[:-1, :] + incompress_weight[1:, :])
            weight_y[0, :] = incompress_weight[0, :]
            weight_y[-1, :] = incompress_weight[-1, :]
        else:
            # No weight - full incompressibility everywhere
            weight_x = np.ones((st.ny, st.nx + 1))
            weight_y = np.ones((st.ny + 1, st.nx))
        
        if bc_type == BoundaryCondition.NEUMANN:
            # For Neumann BC (∂φ/∂n = 0), we use ghost cells to enforce zero gradient
            
            # X-faces
            # Interior faces: standard centered difference
            grad_phi_x_int = (phi[:, 1:] - phi[:, :-1]) / dx
            st.velocity_x_face[:, 1:-1] -= dt * st.beta_x[:, 1:-1] * grad_phi_x_int * weight_x[:, 1:-1]
            
            # Boundary faces for Neumann BC
            # At left boundary (x=0): ghost cell has φ[-1,j] = φ[0,j]
            # So gradient at face is: (φ[0,j] - φ[-1,j])/dx = 0
            grad_phi_x_left = 0.0
            st.velocity_x_face[:, 0] -= dt * st.beta_x[:, 0] * grad_phi_x_left
            
            # At right boundary (x=nx): ghost cell has φ[nx,j] = φ[nx-1,j]
            # So gradient at face is: (φ[nx,j] - φ[nx-1,j])/dx = 0
            grad_phi_x_right = 0.0
            st.velocity_x_face[:, -1] -= dt * st.beta_x[:, -1] * grad_phi_x_right
            
            # Y-faces
            # Interior faces: standard centered difference
            grad_phi_y_int = (phi[1:, :] - phi[:-1, :]) / dx
            st.velocity_y_face[1:-1, :] -= dt * st.beta_y[1:-1, :] * grad_phi_y_int * weight_y[1:-1, :]
            
            # Boundary faces for Neumann BC
            # At bottom boundary (y=0): ghost cell has φ[i,-1] = φ[i,0]
            # So gradient at face is: (φ[i,0] - φ[i,-1])/dy = 0
            grad_phi_y_bottom = 0.0
            st.velocity_y_face[0, :] -= dt * st.beta_y[0, :] * grad_phi_y_bottom
            
            # At top boundary (y=ny): ghost cell has φ[i,ny] = φ[i,ny-1]
            # So gradient at face is: (φ[i,ny] - φ[i,ny-1])/dy = 0
            grad_phi_y_top = 0.0
            st.velocity_y_face[-1, :] -= dt * st.beta_y[-1, :] * grad_phi_y_top
            
        else:  # Dirichlet BC
            # For Dirichlet BC (φ = 0 at boundaries), we use one-sided differences
            
            # X-faces
            # Interior faces
            grad_phi_x_int = (phi[:, 1:] - phi[:, :-1]) / dx
            st.velocity_x_face[:, 1:-1] -= dt * st.beta_x[:, 1:-1] * grad_phi_x_int * weight_x[:, 1:-1]
            
            # Left boundary: φ[-1,j] = 0, so gradient = (φ[0,j] - 0) / dx
            grad_phi_x_left = phi[:, 0] / dx
            st.velocity_x_face[:, 0] -= dt * st.beta_x[:, 0] * grad_phi_x_left
            
            # Right boundary: φ[nx,j] = 0, so gradient = (0 - φ[nx-1,j]) / dx
            grad_phi_x_right = -phi[:, -1] / dx
            st.velocity_x_face[:, -1] -= dt * st.beta_x[:, -1] * grad_phi_x_right
            
            # Y-faces
            # Interior faces
            grad_phi_y_int = (phi[1:, :] - phi[:-1, :]) / dx
            st.velocity_y_face[1:-1, :] -= dt * st.beta_y[1:-1, :] * grad_phi_y_int * weight_y[1:-1, :]
            
            # Bottom boundary: φ[i,-1] = 0, so gradient = (φ[i,0] - 0) / dy
            grad_phi_y_bottom = phi[0, :] / dx
            st.velocity_y_face[0, :] -= dt * st.beta_y[0, :] * grad_phi_y_bottom
            
            # Top boundary: φ[i,ny] = 0, so gradient = (0 - φ[i,ny-1]) / dy
            grad_phi_y_top = -phi[-1, :] / dx
            st.velocity_y_face[-1, :] -= dt * st.beta_y[-1, :] * grad_phi_y_top
        
        # Add gravity term for implicit scheme
        if implicit_gravity and gx is not None and gy is not None:
            # Interpolate gravity to faces and add to face velocities
            # X-faces
            gx_face = np.zeros((st.ny, st.nx + 1))
            gx_face[:, 1:-1] = 0.5 * (gx[:, :-1] + gx[:, 1:])
            gx_face[:, 0] = gx[:, 0]
            gx_face[:, -1] = gx[:, -1]
            st.velocity_x_face += dt * gx_face
            
            # Y-faces
            gy_face = np.zeros((st.ny + 1, st.nx))
            gy_face[1:-1, :] = 0.5 * (gy[:-1, :] + gy[1:, :])
            gy_face[0, :] = gy[0, :]
            gy_face[-1, :] = gy[-1, :]
            st.velocity_y_face += dt * gy_face
    
    def _compute_gravity_divergence(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """Compute divergence of β*g where β = 1/ρ on faces.
        
        This is: ∇·(βg) = ∂(βx*gx)/∂x + ∂(βy*gy)/∂y
        
        Args:
            gx, gy: Cell-centered gravity fields
            
        Returns:
            div_beta_g: Cell-centered divergence of β*g
        """
        st = self.state
        dx = st.dx
        
        # Interpolate gravity to faces
        # X-faces: average adjacent cells
        gx_face = np.zeros((st.ny, st.nx + 1))
        gx_face[:, 1:-1] = 0.5 * (gx[:, :-1] + gx[:, 1:])
        gx_face[:, 0] = gx[:, 0]  # Left boundary
        gx_face[:, -1] = gx[:, -1]  # Right boundary
        
        # Y-faces: average adjacent cells  
        gy_face = np.zeros((st.ny + 1, st.nx))
        gy_face[1:-1, :] = 0.5 * (gy[:-1, :] + gy[1:, :])
        gy_face[0, :] = gy[0, :]  # Bottom boundary
        gy_face[-1, :] = gy[-1, :]  # Top boundary
        
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
    
    def _solve_with_monitoring(self, rhs, beta_x, beta_y, dx, bc_type, tol, max_cycles):
        """Solve with convergence monitoring."""
        from multigrid import compute_residual_mac_vectorized, v_cycle_mac
        
        # Setup
        ny, nx = rhs.shape
        phi = self.phi_prev.copy() if self.phi_prev is not None else np.zeros_like(rhs)
        
        # Compute initial residual
        rhs_norm = np.linalg.norm(rhs)
        if rhs_norm < 1e-14:
            return phi, 0, 0.0
        
        # Power-of-2 padding for multigrid
        ny_pad = 1 << int(np.ceil(np.log2(ny)))
        nx_pad = 1 << int(np.ceil(np.log2(nx)))
        
        # Pad arrays
        phi_pad = np.pad(phi, ((0, ny_pad-ny), (0, nx_pad-nx)), mode='edge')
        rhs_pad = np.pad(rhs, ((0, ny_pad-ny), (0, nx_pad-nx)), mode='constant')
        
        # Pad beta arrays
        beta_x_pad = np.ones((ny_pad, nx_pad + 1), dtype=beta_x.dtype)
        beta_x_pad[:beta_x.shape[0], :beta_x.shape[1]] = beta_x
        if ny_pad > ny:
            beta_x_pad[ny:, :beta_x.shape[1]] = beta_x[-1:, :]
        if nx_pad > nx:
            beta_x_pad[:, nx+1:] = beta_x_pad[:, nx:nx+1]
        
        beta_y_pad = np.ones((ny_pad + 1, nx_pad), dtype=beta_y.dtype)
        beta_y_pad[:beta_y.shape[0], :beta_y.shape[1]] = beta_y
        if nx_pad > nx:
            beta_y_pad[:beta_y.shape[0], nx:] = beta_y[:, -1:]
        if ny_pad > ny:
            beta_y_pad[ny+1:, :] = beta_y_pad[ny:ny+1, :]
        
        # Compute max levels
        max_level = max(1, int(np.log2(min(ny_pad, nx_pad))) - 2)
        
        # V-cycle iterations
        for cycle in range(max_cycles):
            # V-cycle
            phi_pad = v_cycle_mac(
                phi_pad, rhs_pad, beta_x_pad, beta_y_pad,
                dx, 0, max_level, bc_type
            )
            
            # Check convergence
            residual = compute_residual_mac_vectorized(
                phi_pad, rhs_pad, beta_x_pad, beta_y_pad, dx, bc_type
            )
            res_norm = np.linalg.norm(residual[:ny, :nx])
            rel_res = res_norm / rhs_norm
            
            if rel_res < tol:
                break
        
        # Extract solution
        phi = phi_pad[:ny, :nx]
        
        # Remove mean for Neumann BC
        if bc_type == BoundaryCondition.NEUMANN:
            phi -= np.mean(phi)
        
        return phi, cycle + 1, rel_res
    
    def get_convergence_stats(self):
        """Get convergence statistics."""
        if not self.convergence_history:
            return None
        
        history = self.convergence_history
        return {
            'total_solves': len(history),
            'avg_iterations': np.mean([h['iterations'] for h in history]),
            'max_iterations': max(h['iterations'] for h in history),
            'avg_residual': np.mean([h['residual'] for h in history]),
            'convergence_rate': sum(1 for h in history if h['converged']) / len(history)
        }


