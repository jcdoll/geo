"""
Heat transfer solver using multigrid for the implicit diffusion solve.

This implementation uses backward Euler time stepping with multigrid
for solving the resulting linear system, which is more efficient than
ADI for large grids.
"""

import numpy as np
from typing import Optional
from state import FluxState
from materials import MaterialType, MaterialDatabase
from multigrid import solve_mac_poisson_vectorized, BoundaryCondition


class HeatTransferMultigrid:
    """Heat transfer solver using multigrid for implicit diffusion."""
    
    def __init__(self, state: FluxState):
        """
        Initialize heat transfer solver.
        
        Args:
            state: FluxState instance
        """
        self.state = state
        
        # Physical constants
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        self.t_space = 2.7  # K (cosmic background temperature)
        
        # Material database
        self.material_db = MaterialDatabase()
        
        # Cache for warm start
        self.prev_temperature = None
        
    def solve_heat_equation(self, dt: float):
        """
        Solve heat equation for one timestep using multigrid.
        
        Uses backward Euler: (T^{n+1} - T^n)/dt = ∇·(α∇T^{n+1}) + Q/(ρC_p)
        
        This is solved by reformulating as a modified Poisson equation
        and using multigrid iteration.
        
        Args:
            dt: Time step in seconds
        """
        st = self.state
        
        # Update material properties
        st.update_mixture_properties(self.material_db)
        
        # Skip if no material present
        if np.max(st.density) < 1e-3:
            return
            
        # Apply implicit diffusion using multigrid
        self._solve_diffusion_multigrid(dt)
        
        # Apply heat sources
        self._apply_heat_sources(dt)
        
        # Apply radiation boundary conditions
        self._apply_radiation_boundary(dt)
        
        # Ensure positive temperature
        st.temperature = np.maximum(st.temperature, 0.1)
        
    def _solve_diffusion_multigrid(self, dt: float):
        """
        Solve thermal diffusion implicitly using multigrid.
        
        The backward Euler discretization gives:
        (I - dt*L)T^{n+1} = T^n
        
        Where L is the diffusion operator: L = ∇·(α∇)
        
        We solve this iteratively using multigrid-preconditioned iteration.
        """
        st = self.state
        ny, nx = st.temperature.shape
        dx = st.dx
        
        # Compute thermal diffusivity α = k/(ρ*C_p)
        k = st.thermal_conductivity
        rho = st.density
        cp = st.specific_heat
        
        # Avoid division by zero
        valid_mask = (rho > 1e-3) & (cp > 1e-3)
        alpha = np.zeros_like(k)
        alpha[valid_mask] = k[valid_mask] / (rho[valid_mask] * cp[valid_mask])
        
        # For implicit backward Euler, we solve:
        # T^{n+1} - dt*∇·(α∇T^{n+1}) = T^n
        #
        # Using a fixed-point iteration with multigrid preconditioning:
        # T^{k+1} = T^n + dt*∇·(α∇T^k)
        #
        # The correction δT = T^{k+1} - T^k satisfies:
        # δT - dt*∇·(α∇δT) ≈ R^k
        # Where R^k = T^n + dt*∇·(α∇T^k) - T^k is the residual
        #
        # We approximate this by solving: -∇·(α∇δT) = R^k/dt
        # using multigrid, then update T^{k+1} = T^k + ω*δT
        
        # Initial guess (use previous temperature if available)
        if self.prev_temperature is not None and self.prev_temperature.shape == st.temperature.shape:
            T = self.prev_temperature.copy()
        else:
            T = st.temperature.copy()
            
        T_old = st.temperature.copy()
        
        # Fixed-point iteration with multigrid preconditioning
        max_iter = 5  # Usually converges in 2-3 iterations
        omega = 0.8   # Under-relaxation for stability
        
        for iter in range(max_iter):
            # Compute diffusion term: ∇·(α∇T)
            diffusion = self._compute_diffusion(T, alpha, dx)
            
            # Compute residual: R = T_old + dt*diffusion - T
            residual = T_old + dt * diffusion - T
            
            # Check convergence
            res_norm = np.linalg.norm(residual) / (ny * nx)
            if res_norm < 1e-6:
                break
                
            # Solve for correction using multigrid
            # We solve: -∇·(α∇δT) = R/dt
            # But multigrid expects: ∇·(β∇φ) = f
            # So we use β = α, f = -R/dt
            
            # Create face-centered diffusion coefficients
            alpha_x = np.zeros((ny, nx+1))
            alpha_x[:, 1:-1] = 2.0 / (1.0/(alpha[:, :-1] + 1e-10) + 1.0/(alpha[:, 1:] + 1e-10))
            alpha_x[:, 0] = alpha[:, 0]
            alpha_x[:, -1] = alpha[:, -1]
            
            alpha_y = np.zeros((ny+1, nx))
            alpha_y[1:-1, :] = 2.0 / (1.0/(alpha[:-1, :] + 1e-10) + 1.0/(alpha[1:, :] + 1e-10))
            alpha_y[0, :] = alpha[0, :]
            alpha_y[-1, :] = alpha[-1, :]
            
            # Clamp diffusion coefficients to reasonable range
            alpha_x = np.clip(alpha_x, 0, 1e3)
            alpha_y = np.clip(alpha_y, 0, 1e3)
            
            # RHS for multigrid
            rhs = -residual / dt
            
            # Zero RHS in space regions (no heat diffusion in vacuum)
            space_mask = st.density < 1e-3
            rhs[space_mask] = 0.0
            
            # Clamp RHS to prevent numerical overflow in multigrid
            rhs = np.clip(rhs, -1e6, 1e6)
            
            # Solve for correction
            delta_T = solve_mac_poisson_vectorized(
                rhs, alpha_x, alpha_y, dx,
                bc_type=BoundaryCondition.NEUMANN,
                tol=1e-4,
                max_cycles=20,
                verbose=False
            )
            
            # Update temperature with under-relaxation
            delta_T = np.clip(delta_T, -1000, 1000)  # Limit correction magnitude
            T += omega * delta_T
            
            # Enforce boundary conditions
            T[space_mask] = self.t_space
            T = np.clip(T, 0.1, 10000)  # Keep temperature in reasonable range
            
        # Update state temperature
        st.temperature = T
        self.prev_temperature = T.copy()
        
    def apply_thermal_diffusion(self, dt: float):
        """
        Apply thermal diffusion using multigrid method.
        This is a public interface for testing purposes.
        
        Args:
            dt: Time step
        """
        self._solve_diffusion_multigrid(dt)
        
    def _compute_diffusion(self, T: np.ndarray, alpha: np.ndarray, dx: float) -> np.ndarray:
        """
        Compute the diffusion term ∇·(α∇T) using finite differences.
        
        Args:
            T: Temperature field
            alpha: Thermal diffusivity field
            dx: Grid spacing
            
        Returns:
            Diffusion term
        """
        ny, nx = T.shape
        diffusion = np.zeros_like(T)
        
        # X-direction diffusion
        # ∂/∂x(α ∂T/∂x)
        for j in range(ny):
            for i in range(1, nx-1):
                alpha_e = 0.5 * (alpha[j, i] + alpha[j, i+1])
                alpha_w = 0.5 * (alpha[j, i] + alpha[j, i-1])
                
                flux_e = alpha_e * (T[j, i+1] - T[j, i]) / dx
                flux_w = alpha_w * (T[j, i] - T[j, i-1]) / dx
                
                diffusion[j, i] += (flux_e - flux_w) / dx
        
        # Y-direction diffusion
        # ∂/∂y(α ∂T/∂y)
        for j in range(1, ny-1):
            for i in range(nx):
                alpha_n = 0.5 * (alpha[j, i] + alpha[j+1, i])
                alpha_s = 0.5 * (alpha[j, i] + alpha[j-1, i])
                
                flux_n = alpha_n * (T[j+1, i] - T[j, i]) / dx
                flux_s = alpha_s * (T[j, i] - T[j-1, i]) / dx
                
                diffusion[j, i] += (flux_n - flux_s) / dx
                
        return diffusion
        
    def _apply_heat_sources(self, dt: float):
        """Apply heat generation from radioactive decay."""
        st = self.state
        
        # Compute volumetric heat generation
        Q = np.zeros_like(st.temperature)
        
        for mat_idx in range(st.n_materials):
            if mat_idx == 0:  # Skip space
                continue
                
            mat_type = MaterialType(mat_idx)
            props = self.material_db.get_properties(mat_type)
            
            if props.heat_generation > 0:
                Q += st.vol_frac[mat_idx] * props.heat_generation
        
        # Add external heat sources
        Q += st.heat_source
        
        # Convert to temperature change
        valid = (st.density > 1e-3) & (st.specific_heat > 1e-3)
        dT = np.zeros_like(st.temperature)
        dT[valid] = Q[valid] / (st.density[valid] * st.specific_heat[valid])
        
        st.temperature += dt * dT
        
    def _apply_radiation_boundary(self, dt: float):
        """Apply Stefan-Boltzmann radiation at space boundaries using implicit Newton-Raphson."""
        st = self.state
        
        # Find material-space interfaces
        space_mask = st.vol_frac[0] > 0.9
        
        # Collect radiating cells
        radiating_cells = []
        for j in range(1, st.ny-1):
            for i in range(1, st.nx-1):
                if not space_mask[j, i] and st.density[j, i] > 1e-3:
                    # Check if any neighbor is space
                    if (space_mask[j-1, i] or space_mask[j+1, i] or 
                        space_mask[j, i-1] or space_mask[j, i+1]):
                        radiating_cells.append((j, i))
        
        if not radiating_cells:
            return
            
        # Apply Newton-Raphson implicit radiation
        for j, i in radiating_cells:
            T_old = st.temperature[j, i]
            T = T_old
            emissivity = st.emissivity[j, i]
            rho_cp = st.density[j, i] * st.specific_heat[j, i]
            
            if rho_cp < 1e-3:
                continue
                
            # Newton-Raphson iteration for implicit radiation
            # Solve: T_new - T_old + dt * σ * ε * (T_new^4 - T_space^4) / (ρ * cp * dx) = 0
            alpha = self.stefan_boltzmann * emissivity * dt / (rho_cp * st.dx)
            
            for iteration in range(3):  # Usually converges quickly
                # Clamp temperature to prevent overflow
                T_clamped = min(T, 5000.0)
                T_clamped = max(T_clamped, self.t_space)
                
                # Calculate T^4 with overflow protection
                if T_clamped > 2000:
                    T4 = np.exp(4 * np.log(T_clamped))
                    T3 = np.exp(3 * np.log(T_clamped))
                else:
                    T4 = T_clamped**4
                    T3 = T_clamped**3
                    
                T_space4 = self.t_space**4
                
                # Newton-Raphson update
                f = T - T_old + alpha * (T4 - T_space4)
                df_dT = 1.0 + alpha * 4.0 * T3
                
                delta_T = -f / df_dT
                T = T + delta_T
                
                # Keep physical
                T = max(self.t_space, min(5000.0, T))
                
                # Check convergence
                if abs(delta_T) < 0.1:  # 0.1K tolerance
                    break
                    
            st.temperature[j, i] = T