"""
Automatic switching between standard and Numba-optimized implementations.

This module provides a unified interface that automatically uses Numba
when available, falling back to standard implementations if not.
"""

import warnings
from typing import Optional

# Try to import Numba implementations
NUMBA_AVAILABLE = False
try:
    import numba
    from .spatial_hash_numba import NumbaOptimizedSpatialHash
    from ..physics.density_numba import compute_density_numba_wrapper
    from ..physics.forces_numba import compute_forces_numba_wrapper
    from ..physics.gravity_numba import compute_gravity_numba_wrapper
    NUMBA_AVAILABLE = True
    print("Numba JIT compiler available - using optimized implementations")
except ImportError:
    warnings.warn("Numba not available - using standard implementations. "
                 "Install with: pip install numba")

# Import standard implementations
from .spatial_hash_vectorized import VectorizedSpatialHash
from ..physics.density_vectorized import compute_density_vectorized
from ..physics.forces_vectorized import compute_forces_vectorized
from ..physics.gravity_vectorized import compute_gravity_direct_batched


class SpatialHashFactory:
    """Factory for creating spatial hash with automatic Numba detection."""
    
    @staticmethod
    def create(domain_size: tuple, cell_size: float, 
               max_per_cell: int = 100, force_standard: bool = False):
        """Create spatial hash, preferring Numba if available.
        
        Args:
            domain_size: Domain size
            cell_size: Cell size
            max_per_cell: Max particles per cell
            force_standard: Force use of standard implementation
            
        Returns:
            Spatial hash instance
        """
        if NUMBA_AVAILABLE and not force_standard:
            return NumbaOptimizedSpatialHash(domain_size, cell_size, max_per_cell)
        else:
            return VectorizedSpatialHash(domain_size, cell_size, max_per_cell)


# Unified functions that switch implementations
def compute_density_auto(particles, kernel, n_active):
    """Compute density using best available implementation."""
    if NUMBA_AVAILABLE:
        # Numba version doesn't need kernel object
        compute_density_numba_wrapper(particles, n_active)
    else:
        compute_density_vectorized(particles, kernel, n_active)


def compute_forces_auto(particles, kernel, n_active, 
                       gravity=None, alpha_visc=0.1):
    """Compute forces using best available implementation."""
    if NUMBA_AVAILABLE:
        compute_forces_numba_wrapper(particles, n_active, gravity, alpha_visc)
    else:
        compute_forces_vectorized(particles, kernel, n_active, gravity, alpha_visc)


def compute_gravity_auto(particles, n_active, G=6.67430e-11, softening=0.1):
    """Compute gravity using best available implementation."""
    if NUMBA_AVAILABLE:
        compute_gravity_numba_wrapper(particles, n_active, G, softening)
    else:
        compute_gravity_direct_batched(particles, n_active, G, softening)


# Performance comparison utilities
def benchmark_implementations(n_particles=5000, n_iterations=10):
    """Benchmark standard vs Numba implementations."""
    import time
    import numpy as np
    from ..core.particles import ParticleArrays
    from ..core.kernel_vectorized import CubicSplineKernel
    
    print(f"\nBenchmarking with {n_particles} particles, {n_iterations} iterations")
    
    # Create test data
    particles = ParticleArrays.allocate(n_particles)
    particles.position_x[:n_particles] = np.random.uniform(0, 100, n_particles)
    particles.position_y[:n_particles] = np.random.uniform(0, 100, n_particles)
    particles.velocity_x[:n_particles] = np.random.normal(0, 1, n_particles)
    particles.velocity_y[:n_particles] = np.random.normal(0, 1, n_particles)
    particles.mass[:n_particles] = 1.0
    particles.density[:n_particles] = 1000.0
    particles.pressure[:n_particles] = 1e5
    particles.smoothing_h[:n_particles] = 1.0
    
    kernel = CubicSplineKernel()
    
    # Benchmark spatial hash
    print("\n1. Spatial Hash Build:")
    
    # Standard
    t0 = time.perf_counter()
    hash_std = VectorizedSpatialHash((100, 100), 2.0)
    for _ in range(n_iterations):
        hash_std.build_vectorized(particles, n_particles)
    t_std = (time.perf_counter() - t0) / n_iterations
    print(f"   Standard: {t_std*1000:.2f} ms")
    
    # Numba
    if NUMBA_AVAILABLE:
        t0 = time.perf_counter()
        hash_numba = NumbaOptimizedSpatialHash((100, 100), 2.0)
        for _ in range(n_iterations):
            hash_numba.build_vectorized(particles, n_particles)
        t_numba = (time.perf_counter() - t0) / n_iterations
        print(f"   Numba:    {t_numba*1000:.2f} ms (speedup: {t_std/t_numba:.1f}x)")
    
    # Benchmark neighbor search
    print("\n2. Neighbor Search:")
    
    # Standard
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        hash_std.query_neighbors_vectorized(particles, n_particles, 2.0)
    t_std = (time.perf_counter() - t0) / n_iterations
    print(f"   Standard: {t_std*1000:.2f} ms")
    
    # Numba
    if NUMBA_AVAILABLE:
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            hash_numba.query_neighbors_vectorized(particles, n_particles, 2.0)
        t_numba = (time.perf_counter() - t0) / n_iterations
        print(f"   Numba:    {t_numba*1000:.2f} ms (speedup: {t_std/t_numba:.1f}x)")
    
    # Use Numba neighbors for remaining tests
    if NUMBA_AVAILABLE:
        current_hash = hash_numba
    else:
        current_hash = hash_std
    
    # Benchmark density
    print("\n3. Density Computation:")
    
    # Standard
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        compute_density_vectorized(particles, kernel, n_particles)
    t_std = (time.perf_counter() - t0) / n_iterations
    print(f"   Standard: {t_std*1000:.2f} ms")
    
    # Numba
    if NUMBA_AVAILABLE:
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            compute_density_numba_wrapper(particles, n_particles)
        t_numba = (time.perf_counter() - t0) / n_iterations
        print(f"   Numba:    {t_numba*1000:.2f} ms (speedup: {t_std/t_numba:.1f}x)")
    
    # Benchmark forces
    print("\n4. Force Computation:")
    
    # Standard
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        compute_forces_vectorized(particles, kernel, n_particles)
    t_std = (time.perf_counter() - t0) / n_iterations
    print(f"   Standard: {t_std*1000:.2f} ms")
    
    # Numba
    if NUMBA_AVAILABLE:
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            compute_forces_numba_wrapper(particles, n_particles)
        t_numba = (time.perf_counter() - t0) / n_iterations
        print(f"   Numba:    {t_numba*1000:.2f} ms (speedup: {t_std/t_numba:.1f}x)")
    
    # Benchmark gravity
    print("\n5. Gravity Computation:")
    
    # Standard
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        compute_gravity_direct_batched(particles, n_particles)
    t_std = (time.perf_counter() - t0) / n_iterations
    print(f"   Standard: {t_std*1000:.2f} ms")
    
    # Numba
    if NUMBA_AVAILABLE:
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            compute_gravity_numba_wrapper(particles, n_particles)
        t_numba = (time.perf_counter() - t0) / n_iterations
        print(f"   Numba:    {t_numba*1000:.2f} ms (speedup: {t_std/t_numba:.1f}x)")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    benchmark_implementations()