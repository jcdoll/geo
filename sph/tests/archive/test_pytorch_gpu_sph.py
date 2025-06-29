#!/usr/bin/env python3
"""
Test PyTorch GPU backend for SPH on RTX 5080.
"""

import numpy as np
import time
import torch
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel


def test_pytorch_gpu():
    """Test PyTorch GPU backend performance."""
    
    print("=== PyTorch GPU Backend Test for SPH ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create test particles
    n_particles = 10000
    print(f"\nCreating {n_particles} test particles...")
    
    particles = ParticleArrays.allocate(n_particles)
    kernel = CubicSplineKernel(dim=2)
    
    # Initialize random positions
    particles.position_x[:n_particles] = np.random.uniform(-50, 50, n_particles).astype(np.float32)
    particles.position_y[:n_particles] = np.random.uniform(-50, 50, n_particles).astype(np.float32)
    particles.velocity_x[:n_particles] = np.random.uniform(-1, 1, n_particles).astype(np.float32)
    particles.velocity_y[:n_particles] = np.random.uniform(-1, 1, n_particles).astype(np.float32)
    particles.mass[:n_particles] = 1000.0
    particles.smoothing_h[:n_particles] = 2.0
    particles.density[:n_particles] = 1000.0
    particles.pressure[:n_particles] = 0.0
    
    # Build neighbor lists (simplified - in real code use spatial hash)
    print("Building neighbor lists...")
    for i in range(n_particles):
        # Just use first 20 particles as neighbors for testing
        particles.neighbor_count[i] = min(20, n_particles - 1)
        for j in range(particles.neighbor_count[i]):
            if j < i:
                particles.neighbor_ids[i, j] = j
            else:
                particles.neighbor_ids[i, j] = j + 1
            # Compute distance
            dx = particles.position_x[i] - particles.position_x[particles.neighbor_ids[i, j]]
            dy = particles.position_y[i] - particles.position_y[particles.neighbor_ids[i, j]]
            particles.neighbor_distances[i, j] = np.sqrt(dx*dx + dy*dy)
    
    # Test each backend
    backends = ['cpu', 'numba', 'gpu']
    
    for backend_name in backends:
        try:
            print(f"\n--- Testing {backend_name.upper()} backend ---")
            sph.set_backend(backend_name)
            
            # Warmup
            sph.compute_density(particles, kernel, n_particles)
            sph.compute_forces(particles, kernel, n_particles, gravity=np.array([0, -9.81]))
            
            # Time density computation
            start = time.perf_counter()
            for _ in range(10):
                sph.compute_density(particles, kernel, n_particles)
            density_time = (time.perf_counter() - start) / 10
            
            # Time force computation
            start = time.perf_counter()
            for _ in range(10):
                sph.compute_forces(particles, kernel, n_particles, gravity=np.array([0, -9.81]))
            force_time = (time.perf_counter() - start) / 10
            
            print(f"Density computation: {density_time*1000:.2f} ms")
            print(f"Force computation: {force_time*1000:.2f} ms")
            print(f"Total physics step: {(density_time + force_time)*1000:.2f} ms")
            print(f"Theoretical FPS: {1.0 / (density_time + force_time):.0f}")
            
        except Exception as e:
            print(f"Failed: {e}")
    
    # Show current backend info
    print("\n" + "="*50)
    sph.print_backend_info()


if __name__ == "__main__":
    test_pytorch_gpu()