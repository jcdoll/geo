"""
Test optimized GPU SPH implementation.
"""

import numpy as np
import time
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.spatial_hash_vectorized import VectorizedSpatialHash, find_neighbors_vectorized

def benchmark_sph_step(n_particles, backend, use_neighbors=True):
    """Benchmark a single SPH timestep."""
    print(f"\nBenchmarking {backend} backend with {n_particles} particles...")
    
    # Set backend
    sph.set_backend(backend)
    
    # Create particles
    particles = ParticleArrays.allocate(n_particles, max_neighbors=50)
    kernel = CubicSplineKernel()
    
    # Initialize random positions in a box
    box_size = int(np.sqrt(n_particles) * 0.05)
    particles.position_x[:] = np.random.uniform(0, box_size, n_particles)
    particles.position_y[:] = np.random.uniform(0, box_size, n_particles)
    particles.velocity_x[:] = np.random.uniform(-1, 1, n_particles)
    particles.velocity_y[:] = np.random.uniform(-1, 1, n_particles)
    particles.mass[:] = 1.0
    particles.smoothing_h[:] = 0.1
    
    # Find neighbors if requested
    if use_neighbors:
        spatial_hash = VectorizedSpatialHash(domain_size=(box_size, box_size), 
                                           cell_size=0.2, domain_min=(0, 0))
        find_neighbors_vectorized(particles, spatial_hash, n_particles, search_radius=0.2)
    
    # Warm-up (important for GPU)
    sph.compute_density(particles, kernel, n_particles)
    sph.compute_pressure(particles, n_particles)
    sph.compute_forces(particles, kernel, n_particles)
    
    # Time individual operations
    times = {}
    
    # Density computation
    start = time.time()
    for _ in range(10):
        sph.compute_density(particles, kernel, n_particles)
    times['density'] = (time.time() - start) / 10
    
    # Pressure computation
    start = time.time()
    for _ in range(10):
        sph.compute_pressure(particles, n_particles)
    times['pressure'] = (time.time() - start) / 10
    
    # Force computation
    start = time.time()
    for _ in range(10):
        sph.compute_forces(particles, kernel, n_particles)
    times['forces'] = (time.time() - start) / 10
    
    # Integration (if available)
    if backend == 'gpu':
        start = time.time()
        for _ in range(10):
            sph.integrate(particles, n_particles, dt=0.001)
        times['integrate'] = (time.time() - start) / 10
    
    # Total step time
    start = time.time()
    for _ in range(10):
        sph.compute_density(particles, kernel, n_particles)
        sph.compute_pressure(particles, n_particles)
        sph.compute_forces(particles, kernel, n_particles)
        if backend == 'gpu':
            sph.integrate(particles, n_particles, dt=0.001)
    times['total'] = (time.time() - start) / 10
    
    return times


def main():
    """Run benchmarks."""
    # Test different particle counts
    particle_counts = [1000, 5000, 10000, 20000, 50000]
    
    # Import the unified GPU implementation
    try:
        from sph.physics.gpu_unified import compute_density_unified, compute_forces_unified
        from sph.physics.gpu_unified import integrate_unified, compute_pressure_unified
        print("Unified GPU implementation loaded successfully!")
    except ImportError as e:
        print(f"Failed to load GPU implementation: {e}")
        return
    
    # Run benchmarks
    results = {}
    
    for n in particle_counts:
        # CPU/Numba benchmark
        try:
            cpu_times = benchmark_sph_step(n, 'numba', use_neighbors=True)
            results[f'numba_{n}'] = cpu_times
        except Exception as e:
            print(f"Numba benchmark failed: {e}")
            try:
                cpu_times = benchmark_sph_step(n, 'cpu', use_neighbors=True)
                results[f'cpu_{n}'] = cpu_times
            except Exception as e:
                print(f"CPU benchmark failed: {e}")
        
        # GPU benchmark
        try:
            gpu_times = benchmark_sph_step(n, 'gpu', use_neighbors=True)
            results[f'gpu_{n}'] = gpu_times
        except Exception as e:
            print(f"GPU benchmark failed for {n} particles: {e}")
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    for n in particle_counts:
        print(f"\n{n} particles:")
        
        # Get CPU/Numba times
        cpu_key = f'numba_{n}' if f'numba_{n}' in results else f'cpu_{n}'
        if cpu_key in results:
            cpu_times = results[cpu_key]
            backend = 'Numba' if 'numba' in cpu_key else 'CPU'
            print(f"\n{backend}:")
            for op, t in cpu_times.items():
                print(f"  {op:10s}: {t*1000:8.3f} ms")
        
        # Get GPU times
        gpu_key = f'gpu_{n}'
        if gpu_key in results:
            gpu_times = results[gpu_key]
            print(f"\nGPU:")
            for op, t in gpu_times.items():
                print(f"  {op:10s}: {t*1000:8.3f} ms")
            
            # Calculate speedup
            if cpu_key in results:
                print(f"\nSpeedup (GPU vs {backend}):")
                for op in ['density', 'forces', 'total']:
                    if op in cpu_times and op in gpu_times:
                        speedup = cpu_times[op] / gpu_times[op]
                        print(f"  {op:10s}: {speedup:6.1f}x")


if __name__ == "__main__":
    main()