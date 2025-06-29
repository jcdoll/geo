"""
Vectorized particle data structure using Structure-of-Arrays (SoA) pattern.

This design is optimized for:
- SIMD operations on CPU
- Coalesced memory access on GPU
- Cache-efficient batch processing
- Easy migration to GPU backends (CuPy, PyOpenCL)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParticleArrays:
    """Structure of Arrays for efficient vectorization.
    
    All arrays are pre-allocated with float32 for GPU compatibility.
    Uses contiguous memory layout for optimal cache performance.
    """
    # Primary state (N particles)
    position_x: np.ndarray      # shape: (N,) float32
    position_y: np.ndarray      # shape: (N,) float32
    velocity_x: np.ndarray      # shape: (N,) float32
    velocity_y: np.ndarray      # shape: (N,) float32
    
    # Particle properties
    mass: np.ndarray            # shape: (N,) float32
    density: np.ndarray         # shape: (N,) float32
    pressure: np.ndarray        # shape: (N,) float32
    temperature: np.ndarray     # shape: (N,) float32
    
    # Material info
    material_id: np.ndarray     # shape: (N,) int32
    
    # SPH parameters
    smoothing_h: np.ndarray     # shape: (N,) float32
    
    # Force accumulators
    force_x: np.ndarray         # shape: (N,) float32
    force_y: np.ndarray         # shape: (N,) float32
    
    # Neighbor data (fixed max neighbors K)
    neighbor_ids: np.ndarray    # shape: (N, K) int32
    neighbor_distances: np.ndarray  # shape: (N, K) float32
    neighbor_count: np.ndarray  # shape: (N,) int32
    
    # Optional: Solid mechanics
    stress_xx: Optional[np.ndarray] = None  # shape: (N,) float32
    stress_yy: Optional[np.ndarray] = None  # shape: (N,) float32
    stress_xy: Optional[np.ndarray] = None  # shape: (N,) float32
    
    @staticmethod
    def allocate(max_particles: int, max_neighbors: int = 64, 
                 include_solids: bool = False) -> 'ParticleArrays':
        """Pre-allocate arrays for GPU-friendly memory.
        
        Args:
            max_particles: Maximum number of particles to support
            max_neighbors: Maximum neighbors per particle (default 64)
            include_solids: Whether to allocate solid mechanics arrays
            
        Returns:
            Pre-allocated ParticleArrays instance
        """
        # Ensure 32-byte alignment for SIMD
        def aligned_zeros(shape, dtype=np.float32):
            size = np.prod(shape) * np.dtype(dtype).itemsize
            # Round up to 32-byte boundary
            aligned_size = ((size + 31) // 32) * 32
            buffer = np.zeros(aligned_size, dtype=np.uint8)
            return np.frombuffer(buffer, dtype=dtype)[:np.prod(shape)].reshape(shape)
        
        arrays = ParticleArrays(
            # Position and velocity
            position_x=aligned_zeros(max_particles),
            position_y=aligned_zeros(max_particles),
            velocity_x=aligned_zeros(max_particles),
            velocity_y=aligned_zeros(max_particles),
            
            # Properties
            mass=aligned_zeros(max_particles),
            density=aligned_zeros(max_particles),
            pressure=aligned_zeros(max_particles),
            temperature=aligned_zeros(max_particles),
            
            # Material
            material_id=np.zeros(max_particles, dtype=np.int32),
            
            # SPH
            smoothing_h=aligned_zeros(max_particles),
            
            # Forces
            force_x=aligned_zeros(max_particles),
            force_y=aligned_zeros(max_particles),
            
            # Neighbors
            neighbor_ids=np.full((max_particles, max_neighbors), -1, dtype=np.int32),
            neighbor_distances=aligned_zeros((max_particles, max_neighbors)),
            neighbor_count=np.zeros(max_particles, dtype=np.int32)
        )
        
        # Optionally add solid mechanics
        if include_solids:
            arrays.stress_xx = aligned_zeros(max_particles)
            arrays.stress_yy = aligned_zeros(max_particles)
            arrays.stress_xy = aligned_zeros(max_particles)
        
        return arrays
    
    def get_positions(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get particle positions as (N, 2) array for convenience."""
        if indices is None:
            return np.column_stack((self.position_x, self.position_y))
        else:
            return np.column_stack((self.position_x[indices], self.position_y[indices]))
    
    def get_velocities(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get particle velocities as (N, 2) array for convenience."""
        if indices is None:
            return np.column_stack((self.velocity_x, self.velocity_y))
        else:
            return np.column_stack((self.velocity_x[indices], self.velocity_y[indices]))
    
    def reset_forces(self, n_active: int):
        """Reset force accumulators to zero."""
        self.force_x[:n_active] = 0.0
        self.force_y[:n_active] = 0.0
    
    def reset_neighbors(self, n_active: int):
        """Reset neighbor data."""
        self.neighbor_ids[:n_active] = -1
        self.neighbor_count[:n_active] = 0
        self.neighbor_distances[:n_active] = 0.0