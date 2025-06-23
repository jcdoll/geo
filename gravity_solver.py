"""
Gravity solver for flux-based geological simulation.

Solves Poisson equation for gravitational potential:
∇²φ = 4πGρ

Then computes gravitational field as g = -∇φ

Uses multigrid V-cycle method for efficiency.
"""

import numpy as np
from typing import Tuple
from state import FluxState
from multigrid_mac_vectorized import solve_mac_poisson_vectorized, BoundaryCondition


class GravitySolver:
    """Multigrid gravity solver for flux-based simulation."""
    
    def __init__(self, state: FluxState):
        """
        Initialize gravity solver.
        
        Args:
            state: FluxState instance
        """
        self.state = state
        self.G = 6.67430e-11  # Gravitational constant
        self.use_self_gravity = False  # Default to uniform Earth gravity
        
    def solve_gravity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for gravitational field.
        
        Returns:
            (gx, gy): Gravitational field components
        """
        if self.use_self_gravity:
            # Compute self-gravity using Poisson equation
            density = self.state.density
            
            # Build RHS for Poisson equation: ∇²φ = 4πGρ
            rhs = 4.0 * np.pi * self.G * density
            
            # Solve for potential
            phi = self.solve_poisson(rhs, self.state.dx)
            
            # Compute gravity from potential: g = -∇φ
            gx, gy = self.compute_gradient(phi, self.state.dx)
            gx = -gx
            gy = -gy
        else:
            # Use uniform Earth gravity
            gx = np.zeros_like(self.state.density)
            gy = np.ones_like(self.state.density) * 9.81  # Positive is down
            
            # Zero gravity in space (low density regions)
            space_mask = self.state.density < 0.1
            gx[space_mask] = 0.0
            gy[space_mask] = 0.0
        
        # Store in state
        self.state.gravity_x = gx
        self.state.gravity_y = gy
        
        return gx, gy
        
    def solve_poisson(self, rhs: np.ndarray, dx: float, 
                      max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
        """
        Solve Poisson equation using MAC multigrid.
        
        For gravity, we have a constant-coefficient problem (∇²φ = rhs),
        which is a special case of the variable-coefficient solver.
        
        Args:
            rhs: Right-hand side of equation
            dx: Grid spacing
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Solution to Poisson equation
        """
        ny, nx = rhs.shape
        
        # For constant-coefficient Poisson, beta = 1 everywhere
        # Create face-centered coefficients
        beta_x = np.ones((ny, nx + 1), dtype=np.float32)
        beta_y = np.ones((ny + 1, nx), dtype=np.float32)
        
        # Use the MAC multigrid solver
        return solve_mac_poisson_vectorized(
            rhs, beta_x, beta_y, dx,
            bc_type=BoundaryCondition.NEUMANN,
            tol=tol,
            max_cycles=max_iter
        )
        
    def smooth(self, phi: np.ndarray, rhs: np.ndarray, h: float, 
               n_smooth: int = 1) -> np.ndarray:
        """
        Gauss-Seidel smoothing iterations.
        
        Args:
            phi: Current solution
            rhs: Right-hand side
            h: Grid spacing
            n_smooth: Number of smoothing iterations
            
        Returns:
            Smoothed solution
        """
        ny, nx = phi.shape
        h2 = h * h
        
        for _ in range(n_smooth):
            # Red-black Gauss-Seidel for better parallelization
            # Red points (i+j even)
            for j in range(1, ny-1):
                for i in range(1 + (j % 2), nx-1, 2):
                    phi[j, i] = 0.25 * (
                        phi[j-1, i] + phi[j+1, i] + 
                        phi[j, i-1] + phi[j, i+1] - h2 * rhs[j, i]
                    )
            
            # Black points (i+j odd)  
            for j in range(1, ny-1):
                for i in range(2 - (j % 2), nx-1, 2):
                    phi[j, i] = 0.25 * (
                        phi[j-1, i] + phi[j+1, i] + 
                        phi[j, i-1] + phi[j, i+1] - h2 * rhs[j, i]
                    )
                    
            # Apply boundary conditions (Dirichlet: phi = 0 at boundaries)
            phi[0, :] = 0
            phi[-1, :] = 0
            phi[:, 0] = 0
            phi[:, -1] = 0
            
        return phi
        
    def restrict(self, fine: np.ndarray) -> np.ndarray:
        """
        Restrict from fine to coarse grid (factor of 2).
        
        Uses full weighting stencil:
        1/16 * [1 2 1]
               [2 4 2]
               [1 2 1]
        
        Args:
            fine: Fine grid values
            
        Returns:
            Coarse grid values
        """
        ny_fine, nx_fine = fine.shape
        ny_coarse = ny_fine // 2
        nx_coarse = nx_fine // 2
        
        coarse = np.zeros((ny_coarse, nx_coarse))
        
        # Full weighting restriction
        for j in range(ny_coarse):
            for i in range(nx_coarse):
                # Map to fine grid indices
                jf = 2 * j
                if_ = 2 * i
                
                # Apply stencil with boundary checks
                sum_val = 4.0 * fine[jf, if_]
                
                if jf > 0:
                    sum_val += 2.0 * fine[jf-1, if_]
                if jf < ny_fine - 1:
                    sum_val += 2.0 * fine[jf+1, if_]
                if if_ > 0:
                    sum_val += 2.0 * fine[jf, if_-1]
                if if_ < nx_fine - 1:
                    sum_val += 2.0 * fine[jf, if_+1]
                    
                if jf > 0 and if_ > 0:
                    sum_val += fine[jf-1, if_-1]
                if jf > 0 and if_ < nx_fine - 1:
                    sum_val += fine[jf-1, if_+1]
                if jf < ny_fine - 1 and if_ > 0:
                    sum_val += fine[jf+1, if_-1]
                if jf < ny_fine - 1 and if_ < nx_fine - 1:
                    sum_val += fine[jf+1, if_+1]
                    
                coarse[j, i] = sum_val / 16.0
                
        return coarse
        
    def prolongate(self, coarse: np.ndarray, fine_shape: Tuple[int, int]) -> np.ndarray:
        """
        Prolongate from coarse to fine grid.
        
        Uses bilinear interpolation.
        
        Args:
            coarse: Coarse grid values
            fine_shape: Shape of fine grid
            
        Returns:
            Fine grid values
        """
        ny_fine, nx_fine = fine_shape
        ny_coarse, nx_coarse = coarse.shape
        
        fine = np.zeros(fine_shape)
        
        # Direct injection at coarse points
        for j in range(ny_coarse):
            for i in range(nx_coarse):
                jf = 2 * j
                if_ = 2 * i
                if jf < ny_fine and if_ < nx_fine:
                    fine[jf, if_] = coarse[j, i]
        
        # Interpolate intermediate points
        # Horizontal interpolation
        for j in range(0, ny_fine, 2):
            for i in range(1, nx_fine, 2):
                i_left = i - 1
                i_right = min(i + 1, nx_fine - 1)
                fine[j, i] = 0.5 * (fine[j, i_left] + fine[j, i_right])
                
        # Vertical interpolation (now includes intermediate points)
        for j in range(1, ny_fine, 2):
            for i in range(nx_fine):
                j_up = j - 1
                j_down = min(j + 1, ny_fine - 1)
                fine[j, i] = 0.5 * (fine[j_up, i] + fine[j_down, i])
                
        return fine
        
    def solve_direct(self, rhs: np.ndarray, h: float) -> np.ndarray:
        """
        Direct solver for small grids.
        
        Uses multiple Gauss-Seidel iterations.
        
        Args:
            rhs: Right-hand side
            h: Grid spacing
            
        Returns:
            Solution
        """
        phi = np.zeros_like(rhs)
        # Many iterations for coarse grid
        return self.smooth(phi, rhs, h, n_smooth=100)
        
    def compute_residual(self, phi: np.ndarray, rhs: np.ndarray, h: float) -> np.ndarray:
        """
        Compute residual r = rhs - L(phi).
        
        Args:
            phi: Current solution
            rhs: Right-hand side
            h: Grid spacing
            
        Returns:
            Residual
        """
        ny, nx = phi.shape
        h2 = h * h
        residual = np.zeros_like(phi)
        
        # Interior points
        residual[1:-1, 1:-1] = rhs[1:-1, 1:-1] - (
            phi[0:-2, 1:-1] + phi[2:, 1:-1] + 
            phi[1:-1, 0:-2] + phi[1:-1, 2:] - 4*phi[1:-1, 1:-1]
        ) / h2
        
        return residual
        
    def compute_gradient(self, field: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient using central differences.
        
        Args:
            field: Scalar field
            dx: Grid spacing
            
        Returns:
            (df/dx, df/dy)
        """
        ny, nx = field.shape
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        
        # Central differences in interior
        grad_x[:, 1:-1] = (field[:, 2:] - field[:, 0:-2]) / (2.0 * dx)
        grad_y[1:-1, :] = (field[2:, :] - field[0:-2, :]) / (2.0 * dx)
        
        # One-sided at boundaries
        grad_x[:, 0] = (field[:, 1] - field[:, 0]) / dx
        grad_x[:, -1] = (field[:, -1] - field[:, -2]) / dx
        grad_y[0, :] = (field[1, :] - field[0, :]) / dx
        grad_y[-1, :] = (field[-1, :] - field[-2, :]) / dx
        
        return grad_x, grad_y