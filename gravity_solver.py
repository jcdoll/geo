"""
Gravity solver for flux-based geological simulation with multiple solver options.

Supports:
1. DFT (Discrete Fourier Transform) - Fast, accurate, Dirichlet BC
2. Multigrid - Flexible BC support, currently has issues with Neumann BC

Solves Poisson equation for gravitational potential:
∇²φ = 4πGρ

Then computes gravitational field as g = -∇φ
"""

import numpy as np
from typing import Tuple, Literal
from enum import Enum
from scipy.fft import dstn, idstn
from state import FluxState
from multigrid import solve_mac_poisson_vectorized, BoundaryCondition


class SolverMethod(Enum):
    """Available solver methods for gravity."""
    DFT = "dft"
    MULTIGRID = "multigrid"


class GravitySolver:
    """Gravity solver for flux-based simulation with selectable methods."""
    
    def __init__(self, state: FluxState, method: SolverMethod = SolverMethod.DFT):
        """
        Initialize gravity solver.
        
        Args:
            state: FluxState instance
            method: Solver method to use (DFT or MULTIGRID)
        """
        self.state = state
        self.method = method
        
        # For geological simulation, scale up G to get reasonable gravity values
        # Real G = 6.67430e-11 produces tiny values for km-scale features
        self.G_real = 6.67430e-11  # Real gravitational constant
        self.G_scale = 10000  # Scaling factor for Earth-like gravity (~10 m/s²)
        self.G = self.G_real * self.G_scale  # Scaled gravitational constant
        self.use_self_gravity = True  # Default to self-gravity for realistic simulation
        
        # DFT-specific constants
        self._DST_TYPE = 2
        self._IDST_TYPE = 2
        
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
            
            # Solve for potential using selected method
            if self.method == SolverMethod.DFT:
                phi = self.solve_poisson_dft(rhs, self.state.dx)
            else:
                phi = self.solve_poisson_multigrid(rhs, self.state.dx)
            
            # Compute gravity from potential: g = -∇φ
            gx, gy = self.compute_gradient(phi, self.state.dx)
            gx = -gx
            gy = -gy
        else:
            # Use uniform Earth gravity everywhere (including space)
            gx = np.zeros_like(self.state.density)
            gy = np.ones_like(self.state.density) * 9.81  # Positive is down
            
            # Don't zero gravity in space - let materials fall!
        
        # Store in state
        self.state.gravity_x = gx
        self.state.gravity_y = gy
        
        return gx, gy
    
    def solve_poisson_dft(self, rhs: np.ndarray, dx: float) -> np.ndarray:
        """
        Solve Poisson equation using Discrete Sine Transform (DST).
        
        This method assumes Dirichlet boundary conditions (φ = 0 at boundaries),
        which is appropriate for an isolated body in vacuum.
        
        Args:
            rhs: Right-hand side of equation (4πGρ)
            dx: Grid spacing
            
        Returns:
            Gravitational potential φ
        """
        ny, nx = rhs.shape
        
        # Scale RHS by grid spacing squared
        rhs_scaled = rhs * dx * dx
        
        # Forward DST-II in both directions
        rhs_hat = dstn(rhs_scaled, type=self._DST_TYPE, norm="ortho")
        
        # Eigenvalues of the Laplacian with Dirichlet BCs on a square grid
        j = np.arange(1, ny + 1)  # 1-based index in DST basis
        i = np.arange(1, nx + 1)
        sin_j = np.sin(np.pi * j / (2 * (ny + 1)))
        sin_i = np.sin(np.pi * i / (2 * (nx + 1)))
        lambda_y = -4.0 * sin_j[:, None] ** 2  # shape (ny, 1)
        lambda_x = -4.0 * sin_i[None, :] ** 2  # shape (1, nx)
        eigvals = lambda_x + lambda_y  # Broadcasted sum (ny, nx) – negative values
        
        # Solve in transform domain: φ_hat = rhs_hat / λ
        phi_hat = rhs_hat / eigvals
        
        # Inverse DST to obtain φ
        phi = idstn(phi_hat, type=self._IDST_TYPE, norm="ortho")
        
        return phi
    
    def solve_poisson_multigrid(self, rhs: np.ndarray, dx: float, 
                                max_iter: int = 15, tol: float = 5e-2) -> np.ndarray:
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
        
        # Use the MAC multigrid solver with Neumann BC
        return solve_mac_poisson_vectorized(
            rhs, beta_x, beta_y, dx,
            bc_type=BoundaryCondition.NEUMANN,
            tol=tol,
            max_cycles=max_iter
        )
        
    def compute_gradient(self, phi: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of potential using numpy's gradient function.
        
        Uses second-order accurate edge formulas at boundaries.
        
        Args:
            phi: Potential field
            dx: Grid spacing
            
        Returns:
            (grad_x, grad_y): Gradient components
        """
        # numpy.gradient returns (gy, gx) for 2D arrays
        gy, gx = np.gradient(phi, dx, edge_order=2)
        return gx, gy
    
    def set_method(self, method: SolverMethod):
        """Change the solver method."""
        self.method = method
        
    def get_method_name(self) -> str:
        """Get the current solver method name."""
        return self.method.value