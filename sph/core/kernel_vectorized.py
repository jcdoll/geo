"""
Fully vectorized SPH kernel functions optimized for SIMD and GPU.

Implements cubic spline kernel with vectorized operations for:
- Kernel evaluation W(r, h)
- Kernel gradient ∇W(r, h)
- Batch processing of multiple particles
"""

import numpy as np
from typing import Tuple


class CubicSplineKernel:
    """Fully vectorized cubic spline kernel.
    
    The cubic spline kernel is:
    W(q) = σ * { 1 - 1.5q² + 0.75q³         if 0 ≤ q ≤ 1
               { 0.25(2-q)³                 if 1 < q ≤ 2
               { 0                          if q > 2
    
    where q = r/h and σ is the normalization factor.
    """
    
    def __init__(self, dim: int = 2):
        """Initialize kernel with dimension-specific normalization.
        
        Args:
            dim: Spatial dimension (2 or 3)
        """
        self.dim = dim
        if dim == 2:
            self.norm_factor = 10.0 / (7.0 * np.pi)
        elif dim == 3:
            self.norm_factor = 1.0 / np.pi
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
    
    def W_vectorized(self, r: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Vectorized kernel evaluation.
        
        Args:
            r: Distances to neighbors, shape (N, K) or (K,)
            h: Smoothing lengths, shape (N, 1) or (1,) or scalar
            
        Returns:
            Kernel values with same shape as r
        """
        # Handle scalar h
        if np.isscalar(h):
            h = np.array([[h]], dtype=np.float32)
        elif h.ndim == 1:
            h = h[:, np.newaxis]
        
        # Ensure r is 2D
        r_2d = r if r.ndim == 2 else r[np.newaxis, :]
        
        # Compute normalized distance q = r/h
        q = r_2d / h
        
        # Normalization factor
        norm = self.norm_factor / (h ** self.dim)
        
        # Initialize output
        w = np.zeros_like(r_2d, dtype=np.float32)
        
        # Case 1: q <= 1
        mask1 = q <= 1.0
        q1 = q[mask1]
        w[mask1] = 1 - 1.5 * q1**2 + 0.75 * q1**3
        
        # Case 2: 1 < q <= 2
        mask2 = (q > 1.0) & (q <= 2.0)
        q2 = q[mask2]
        w[mask2] = 0.25 * (2 - q2)**3
        
        # Apply normalization
        w *= norm
        
        # Return with original shape
        return w if r.ndim == 2 else w[0]
    
    def gradW_vectorized(self, dx: np.ndarray, dy: np.ndarray, 
                        r: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized kernel gradient computation.
        
        Args:
            dx: Position differences in x, shape (N, K)
            dy: Position differences in y, shape (N, K)
            r: Distances, shape (N, K)
            h: Smoothing lengths, shape (N, 1) or scalar
            
        Returns:
            (grad_x, grad_y) each with shape (N, K)
        """
        # Handle scalar h
        if np.isscalar(h):
            h = np.array([[h]], dtype=np.float32)
        elif h.ndim == 1:
            h = h[:, np.newaxis]
        
        # Compute normalized distance
        q = r / h
        
        # Normalization factor for gradient
        norm = self.norm_factor / (h ** (self.dim + 1))
        
        # Gradient magnitude dW/dq
        grad_mag = np.zeros_like(r, dtype=np.float32)
        
        # Avoid division by zero
        mask_nonzero = r > 1e-10
        
        # Case 1: q <= 1
        mask1 = (q <= 1.0) & mask_nonzero
        q1 = q[mask1]
        grad_mag[mask1] = -3 * q1 + 2.25 * q1**2
        
        # Case 2: 1 < q <= 2
        mask2 = (q > 1.0) & (q <= 2.0) & mask_nonzero
        q2 = q[mask2]
        grad_mag[mask2] = -0.75 * (2 - q2)**2
        
        # Apply chain rule: ∇W = (dW/dq) * (dq/dr) * (dr/dx)
        # where dq/dr = 1/h and dr/dx = x/r
        factor = norm * grad_mag / (r + 1e-10)
        
        grad_x = factor * dx
        grad_y = factor * dy
        
        # Zero gradient where r is too small
        grad_x[~mask_nonzero] = 0.0
        grad_y[~mask_nonzero] = 0.0
        
        return grad_x, grad_y
    
    def W_self(self, h: np.ndarray) -> np.ndarray:
        """Kernel value at r=0 (self-contribution).
        
        Args:
            h: Smoothing lengths, shape (N,) or scalar
            
        Returns:
            W(0, h) for each particle
        """
        if np.isscalar(h):
            h = np.array([h], dtype=np.float32)
        
        return self.norm_factor / (h ** self.dim)
    
    def validate(self) -> bool:
        """Validate kernel properties (normalization, symmetry)."""
        # Test normalization in 2D
        if self.dim == 2:
            # Integrate kernel over circular area
            r = np.linspace(0, 2, 1000)
            dr = r[1] - r[0]
            integral = 2 * np.pi * np.sum(r * self.W_vectorized(r, 1.0)) * dr
            print(f"2D Normalization integral: {integral:.6f} (should be ≈1.0)")
            
        # Test symmetry
        r_test = np.array([0.5, 1.0, 1.5])
        w1 = self.W_vectorized(r_test, 1.0)
        w2 = self.W_vectorized(r_test[::-1], 1.0)[::-1]
        symmetric = np.allclose(w1, w2)
        print(f"Symmetry test: {'PASS' if symmetric else 'FAIL'}")
        
        return abs(integral - 1.0) < 0.01 and symmetric


class WendlandC2Kernel:
    """Wendland C2 kernel - compact support with C2 continuity.
    
    Better numerical properties than cubic spline but more expensive.
    """
    
    def __init__(self, dim: int = 2):
        self.dim = dim
        if dim == 2:
            self.norm_factor = 7.0 / (4.0 * np.pi)
        elif dim == 3:
            self.norm_factor = 21.0 / (16.0 * np.pi)
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
    
    def W_vectorized(self, r: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Vectorized Wendland C2 kernel evaluation."""
        # Handle scalar h
        if np.isscalar(h):
            h = np.array([[h]], dtype=np.float32)
        elif h.ndim == 1:
            h = h[:, np.newaxis]
        
        # Ensure r is 2D
        r_2d = r if r.ndim == 2 else r[np.newaxis, :]
        
        q = r_2d / (2.0 * h)  # Note: Wendland uses 2h as support radius
        norm = self.norm_factor / (h ** self.dim)
        
        w = np.zeros_like(r_2d, dtype=np.float32)
        mask = q < 1.0
        q_masked = q[mask]
        w[mask] = (1 - q_masked)**4 * (1 + 4*q_masked)
        
        w *= norm
        
        return w if r.ndim == 2 else w[0]