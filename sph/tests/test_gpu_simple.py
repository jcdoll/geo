"""
Simple GPU vs CPU/Numba benchmark for SPH.
"""

import numpy as np
import time
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel

def benchmark_backend(n_particles, backend):
    """Benchmark a backend without neighbors."""
    print(f"\nTesting {backend} with {n_particles} particles...")
    
    # Set backend
    sph.set_backend(backend)
    
    # Create particles
    particles = ParticleArrays.allocate(n_particles, max_neighbors=50)
    kernel = CubicSplineKernel()
    
    # Initialize particles
    box_size = 10.0
    particles.position_x[:] = np.random.uniform(0, box_size, n_particles)
    particles.position_y[:] = np.random.uniform(0, box_size, n_particles) 
    particles.velocity_x[:] = 0.0
    particles.velocity_y[:] = 0.0
    particles.mass[:] = 1.0
    particles.smoothing_h[:] = 0.5
    particles.density[:] = 1000.0  # Initial guess
    
    # Warm up
    sph.compute_density(particles, kernel, n_particles)
    sph.compute_pressure(particles, n_particles)
    sph.compute_forces(particles, kernel, n_particles)
    
    # Time operations
    n_iterations = 5
    
    # Density
    start = time.time()
    for _ in range(n_iterations):
        sph.compute_density(particles, kernel, n_particles)
    density_time = (time.time() - start) / n_iterations
    
    # Pressure
    start = time.time()
    for _ in range(n_iterations):
        sph.compute_pressure(particles, n_particles)
    pressure_time = (time.time() - start) / n_iterations
    
    # Forces
    start = time.time()
    for _ in range(n_iterations):
        sph.compute_forces(particles, kernel, n_particles)
    forces_time = (time.time() - start) / n_iterations
    
    # Full step
    start = time.time()
    for _ in range(n_iterations):
        sph.compute_density(particles, kernel, n_particles)
        sph.compute_pressure(particles, n_particles)
        sph.compute_forces(particles, kernel, n_particles)
        if backend == 'gpu':
            sph.integrate(particles, n_particles, dt=0.001)
    total_time = (time.time() - start) / n_iterations
    
    return {
        'density': density_time,
        'pressure': pressure_time,
        'forces': forces_time,
        'total': total_time
    }

def main():
    # Import GPU implementation
    try:
        from sph.physics.gpu_unified import compute_density_unified
        print("GPU implementation loaded!")
    except ImportError as e:
        print(f"Failed to load GPU: {e}")
        return
    
    # Test sizes
    sizes = [1000, 5000, 10000]
    
    print("\n" + "="*60)
    print("SPH GPU BENCHMARK RESULTS")
    print("="*60)
    
    for n in sizes:
        print(f"\n{n} particles:")
        
        # CPU/Numba
        try:
            cpu_times = benchmark_backend(n, 'numba')
            cpu_backend = 'Numba'
        except:
            cpu_times = benchmark_backend(n, 'cpu')
            cpu_backend = 'CPU'
        
        print(f"{cpu_backend}: density={cpu_times['density']*1000:.1f}ms, "
              f"forces={cpu_times['forces']*1000:.1f}ms, "
              f"total={cpu_times['total']*1000:.1f}ms")
        
        # GPU
        try:
            gpu_times = benchmark_backend(n, 'gpu')
            print(f"GPU: density={gpu_times['density']*1000:.1f}ms, "
                  f"forces={gpu_times['forces']*1000:.1f}ms, "
                  f"total={gpu_times['total']*1000:.1f}ms")
            
            # Speedup
            density_speedup = cpu_times['density'] / gpu_times['density']
            forces_speedup = cpu_times['forces'] / gpu_times['forces']
            total_speedup = cpu_times['total'] / gpu_times['total']
            
            print(f"Speedup: density={density_speedup:.1f}x, "
                  f"forces={forces_speedup:.1f}x, "
                  f"total={total_speedup:.1f}x")
        except Exception as e:
            print(f"GPU failed: {e}")

if __name__ == "__main__":
    main()