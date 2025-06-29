"""
Example of using different SPH backends.

Demonstrates how to select and use CPU, Numba, and GPU backends.
"""

import numpy as np
import time
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel


def run_simulation_step(particles, n_active, kernel, dt=0.001):
    """Run one SPH simulation step."""
    # Compute density from particle positions
    sph.compute_density(particles, kernel, n_active)
    
    # Compute pressure from density (equation of state)
    sph.compute_pressure(particles, n_active, rest_density=1000.0)
    
    # Compute forces from pressure gradients and viscosity
    gravity = np.array([0.0, -9.81])
    sph.compute_forces(particles, kernel, n_active, gravity=gravity, alpha_visc=0.1)
    
    # Integrate positions and velocities
    sph.integrate(particles, n_active, dt=dt)


def example_basic_usage():
    """Basic example of using SPH with different backends."""
    print("SPH Backend Usage Example")
    print("="*50)
    
    # Create particles
    n_particles = 1000
    particles = ParticleArrays.allocate(n_particles)
    kernel = CubicSplineKernel()
    
    # Initialize particle positions (random in a box)
    particles.position_x[:] = np.random.uniform(0, 10, n_particles)
    particles.position_y[:] = np.random.uniform(0, 10, n_particles)
    particles.velocity_x[:] = 0.0
    particles.velocity_y[:] = 0.0
    particles.mass[:] = 1.0
    particles.smoothing_h[:] = 0.2
    
    # Try different backends
    backends = ['cpu', 'numba', 'gpu']
    
    for backend_name in backends:
        print(f"\n{backend_name.upper()} Backend:")
        print("-"*30)
        
        try:
            # Set backend
            sph.set_backend(backend_name)
            print(f"✓ Backend set to: {sph.get_backend()}")
            
            # Run a few simulation steps
            start_time = time.time()
            for step in range(10):
                run_simulation_step(particles, n_particles, kernel)
            
            elapsed = time.time() - start_time
            print(f"✓ 10 steps completed in {elapsed:.3f}s")
            print(f"  Average: {elapsed/10*1000:.1f} ms/step")
            
            # Check results
            print(f"  Density range: [{particles.density.min():.1f}, {particles.density.max():.1f}]")
            print(f"  Avg velocity: {np.sqrt(particles.velocity_x**2 + particles.velocity_y**2).mean():.3f}")
            
        except Exception as e:
            print(f"✗ Backend not available: {e}")


def example_performance_comparison():
    """Compare performance across backends for different workloads."""
    print("\n\nPerformance Comparison")
    print("="*50)
    
    particle_counts = [100, 1000, 5000]
    results = {}
    
    for n in particle_counts:
        print(f"\n{n} particles:")
        results[n] = {}
        
        # Create particles
        particles = ParticleArrays.allocate(n)
        kernel = CubicSplineKernel()
        
        # Initialize
        particles.position_x[:n] = np.random.uniform(0, 5, n)
        particles.position_y[:n] = np.random.uniform(0, 5, n)
        particles.mass[:n] = 1.0
        particles.smoothing_h[:n] = 0.2
        
        # Test each backend
        for backend in ['cpu', 'numba', 'gpu']:
            try:
                sph.set_backend(backend)
                
                # Warmup
                run_simulation_step(particles, n, kernel)
                
                # Time 10 steps
                start = time.time()
                for _ in range(10):
                    run_simulation_step(particles, n, kernel)
                elapsed = (time.time() - start) / 10 * 1000  # ms per step
                
                results[n][backend] = elapsed
                print(f"  {backend:6s}: {elapsed:6.1f} ms/step")
                
            except:
                print(f"  {backend:6s}: not available")
    
    # Print speedup summary
    print("\nSpeedup Summary (relative to CPU):")
    print("-"*40)
    for n in particle_counts:
        if 'cpu' in results[n]:
            print(f"\n{n} particles:")
            cpu_time = results[n]['cpu']
            for backend in ['numba', 'gpu']:
                if backend in results[n]:
                    speedup = cpu_time / results[n][backend]
                    print(f"  {backend}: {speedup:.1f}x faster")


def example_gpu_optimized():
    """Example of optimized GPU usage."""
    print("\n\nOptimized GPU Usage")
    print("="*50)
    
    try:
        sph.set_backend('gpu')
    except:
        print("GPU backend not available")
        return
    
    # For optimal GPU performance with large particle counts
    n_particles = 10000
    particles = ParticleArrays.allocate(n_particles)
    kernel = CubicSplineKernel()
    
    # Initialize
    particles.position_x[:] = np.random.uniform(0, 20, n_particles)
    particles.position_y[:] = np.random.uniform(0, 20, n_particles)
    particles.mass[:] = 1.0
    particles.smoothing_h[:] = 0.3
    
    print(f"Running {n_particles} particles on GPU...")
    
    # For best GPU performance, run many steps without CPU synchronization
    from sph.physics.gpu_unified import get_gpu_sph
    
    gpu = get_gpu_sph()
    gpu.upload_particles(particles, n_particles)
    
    start = time.time()
    n_steps = 100
    
    for _ in range(n_steps):
        gpu.compute_density_gpu(n_particles, use_neighbors=False)
        gpu.compute_pressure_gpu(n_particles)
        gpu.compute_forces_gpu(n_particles)
        gpu.integrate_gpu(n_particles, dt=0.001)
    
    # Only sync at the end
    gpu.download_particles(particles, n_particles)
    
    elapsed = time.time() - start
    print(f"✓ {n_steps} steps completed in {elapsed:.3f}s")
    print(f"  Average: {elapsed/n_steps*1000:.1f} ms/step")
    print(f"  Total throughput: {n_particles * n_steps / elapsed:.0f} particle-updates/sec")


def main():
    """Run all examples."""
    # Check available backends first
    print("Checking available backends...")
    for backend in ['cpu', 'numba', 'gpu']:
        try:
            sph.set_backend(backend)
            print(f"  {backend}: ✓")
        except:
            print(f"  {backend}: ✗")
    
    # Run examples
    example_basic_usage()
    example_performance_comparison()
    example_gpu_optimized()
    
    print("\n" + "="*50)
    print("Examples completed!")


if __name__ == "__main__":
    main()