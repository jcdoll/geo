#!/usr/bin/env python3
"""
Test basic vectorized SPH implementation.

This validates:
- Particle data structure
- Kernel functions
- Spatial hashing
- Density computation
- Force calculation
- Integration
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.spatial_hash_vectorized import VectorizedSpatialHash
from sph.physics.density_vectorized import compute_density_vectorized
from sph.physics.forces_vectorized import compute_forces_vectorized, tait_equation_of_state
from sph.core.integrator_vectorized import integrate_leapfrog_vectorized, compute_adaptive_timestep


def create_dam_break_particles(nx: int = 20, ny: int = 40, spacing: float = 0.05) -> Tuple[ParticleArrays, int]:
    """Create initial dam break configuration."""
    n_particles = nx * ny
    particles = ParticleArrays.allocate(n_particles * 2)  # Extra space
    
    # Create particle grid
    idx = 0
    for i in range(nx):
        for j in range(ny):
            particles.position_x[idx] = (i + 0.5) * spacing
            particles.position_y[idx] = (j + 0.5) * spacing
            particles.velocity_x[idx] = 0.0
            particles.velocity_y[idx] = 0.0
            particles.mass[idx] = 1000.0 * spacing * spacing  # 2D mass
            particles.smoothing_h[idx] = 1.3 * spacing
            particles.material_id[idx] = 0  # Water
            particles.temperature[idx] = 293.15  # 20°C
            idx += 1
    
    return particles, n_particles


def test_kernel():
    """Test kernel properties."""
    print("\n=== Testing Kernel ===")
    kernel = CubicSplineKernel(dim=2)
    
    # Test single evaluation
    r = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    h = 1.0
    W = kernel.W_vectorized(r, h)
    print(f"W(r={r}, h={h}) = {W}")
    
    # Test gradient
    dx = np.array([[0.5, 1.0]])
    dy = np.array([[0.5, 0.0]])
    r = np.array([[np.sqrt(0.5), 1.0]])
    grad_x, grad_y = kernel.gradW_vectorized(dx, dy, r, h)
    print(f"gradW at (0.5,0.5): ({grad_x[0,0]:.4f}, {grad_y[0,0]:.4f})")
    
    # Validate
    kernel.validate()


def test_spatial_hash():
    """Test spatial hashing."""
    print("\n=== Testing Spatial Hash ===")
    
    # Create particles
    particles, n_active = create_dam_break_particles(10, 10)
    
    # Create spatial hash
    domain_size = (2.0, 2.0)
    cell_size = 0.1  # 2 * smoothing_h
    spatial_hash = VectorizedSpatialHash(domain_size, cell_size)
    
    # Build hash
    t0 = time.perf_counter()
    spatial_hash.build_vectorized(particles, n_active)
    t1 = time.perf_counter()
    print(f"Build time: {(t1-t0)*1000:.2f} ms for {n_active} particles")
    
    # Find neighbors
    t0 = time.perf_counter()
    spatial_hash.query_neighbors_vectorized(particles, n_active, 2 * 0.065)
    t1 = time.perf_counter()
    print(f"Neighbor search time: {(t1-t0)*1000:.2f} ms")
    
    # Statistics
    stats = spatial_hash.get_statistics()
    print(f"Hash statistics: {stats}")
    
    # Check neighbors
    avg_neighbors = np.mean(particles.neighbor_count[:n_active])
    max_neighbors = np.max(particles.neighbor_count[:n_active])
    print(f"Average neighbors: {avg_neighbors:.1f}, Max: {max_neighbors}")


def test_density():
    """Test density computation."""
    print("\n=== Testing Density ===")
    
    # Create particles
    particles, n_active = create_dam_break_particles(5, 5)
    
    # Set up neighbors
    spatial_hash = VectorizedSpatialHash((1.0, 1.0), 0.1)
    spatial_hash.build_vectorized(particles, n_active)
    spatial_hash.query_neighbors_vectorized(particles, n_active, 0.13)
    
    # Compute density
    kernel = CubicSplineKernel(dim=2)
    t0 = time.perf_counter()
    compute_density_vectorized(particles, kernel, n_active)
    t1 = time.perf_counter()
    print(f"Density computation time: {(t1-t0)*1000:.2f} ms")
    
    print(f"Density range: {np.min(particles.density[:n_active]):.1f} - "
          f"{np.max(particles.density[:n_active]):.1f} kg/m³")


def test_forces():
    """Test force computation."""
    print("\n=== Testing Forces ===")
    
    # Create particles
    particles, n_active = create_dam_break_particles(5, 5)
    
    # Set up
    spatial_hash = VectorizedSpatialHash((1.0, 1.0), 0.1)
    spatial_hash.build_vectorized(particles, n_active)
    spatial_hash.query_neighbors_vectorized(particles, n_active, 0.13)
    
    kernel = CubicSplineKernel(dim=2)
    compute_density_vectorized(particles, kernel, n_active)
    
    # Compute pressure (Tait EOS)
    rho0 = np.full(n_active, 1000.0)  # Water reference density
    B = np.full(n_active, 2.2e5)      # Bulk modulus
    particles.pressure[:n_active] = tait_equation_of_state(
        particles.density[:n_active], rho0, B
    )
    
    # Compute forces
    t0 = time.perf_counter()
    compute_forces_vectorized(particles, kernel, n_active)
    t1 = time.perf_counter()
    print(f"Force computation time: {(t1-t0)*1000:.2f} ms")
    
    # Check forces
    f_mag = np.sqrt(particles.force_x[:n_active]**2 + particles.force_y[:n_active]**2)
    print(f"Force magnitude range: {np.min(f_mag):.1f} - {np.max(f_mag):.1f} N")


def run_dam_break_simulation():
    """Run a simple dam break simulation."""
    print("\n=== Running Dam Break Simulation ===")
    
    # Parameters
    nx, ny = 20, 40
    domain_size = (2.0, 1.0)
    
    # Create particles
    particles, n_active = create_dam_break_particles(nx, ny)
    print(f"Created {n_active} particles")
    
    # Initialize modules
    kernel = CubicSplineKernel(dim=2)
    spatial_hash = VectorizedSpatialHash(domain_size, 0.1)
    
    # Material properties
    rho0 = np.full(particles.mass.shape[0], 1000.0)
    B = np.full(particles.mass.shape[0], 2.2e5)
    
    # Visualization
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time loop
    t = 0.0
    step = 0
    dt = 1e-4
    
    while t < 0.5 and step < 1000:
        # Update spatial hash
        spatial_hash.build_vectorized(particles, n_active)
        spatial_hash.query_neighbors_vectorized(particles, n_active, 0.13)
        
        # Compute density
        compute_density_vectorized(particles, kernel, n_active)
        
        # Compute pressure
        particles.pressure[:n_active] = tait_equation_of_state(
            particles.density[:n_active], rho0[:n_active], B[:n_active]
        )
        
        # Compute forces
        compute_forces_vectorized(particles, kernel, n_active)
        
        # Integrate
        integrate_leapfrog_vectorized(particles, n_active, dt, 
                                     (0, domain_size[0], 0, domain_size[1]))
        
        # Adaptive timestep
        if step % 10 == 0:
            dt = compute_adaptive_timestep(particles, n_active)
        
        # Visualize every 20 steps
        if step % 20 == 0:
            ax1.clear()
            ax2.clear()
            
            # Particle positions colored by density
            scatter1 = ax1.scatter(particles.position_x[:n_active], 
                                  particles.position_y[:n_active],
                                  c=particles.density[:n_active], 
                                  cmap='Blues',
                                  s=20, alpha=0.8,
                                  vmin=900, vmax=1100)
            ax1.set_xlim(0, domain_size[0])
            ax1.set_ylim(0, domain_size[1])
            ax1.set_aspect('equal')
            ax1.set_title(f'Particles (t={t:.3f}s)')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            
            # Velocity field
            v_mag = np.sqrt(particles.velocity_x[:n_active]**2 + 
                           particles.velocity_y[:n_active]**2)
            scatter2 = ax2.scatter(particles.position_x[:n_active], 
                                  particles.position_y[:n_active],
                                  c=v_mag, cmap='viridis',
                                  s=20, alpha=0.8,
                                  vmin=0, vmax=2.0)
            ax2.set_xlim(0, domain_size[0])
            ax2.set_ylim(0, domain_size[1])
            ax2.set_aspect('equal')
            ax2.set_title('Velocity Magnitude')
            ax2.set_xlabel('X (m)')
            
            if step == 0:
                plt.colorbar(scatter1, ax=ax1, label='Density (kg/m³)')
                plt.colorbar(scatter2, ax=ax2, label='Velocity (m/s)')
            
            plt.pause(0.01)
        
        t += dt
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}, t={t:.4f}s, dt={dt*1000:.2f}ms")
    
    plt.ioff()
    plt.show()


def benchmark_performance():
    """Benchmark vectorized operations."""
    print("\n=== Performance Benchmark ===")
    
    for n_particles in [1000, 5000, 10000]:
        print(f"\nBenchmarking with {n_particles} particles:")
        
        # Create particles
        particles = ParticleArrays.allocate(n_particles)
        particles.position_x[:n_particles] = np.random.uniform(0, 10, n_particles)
        particles.position_y[:n_particles] = np.random.uniform(0, 10, n_particles)
        particles.mass[:n_particles] = 1.0
        particles.smoothing_h[:n_particles] = 0.1
        
        # Modules
        kernel = CubicSplineKernel(dim=2)
        spatial_hash = VectorizedSpatialHash((10.0, 10.0), 0.2)
        
        # Benchmark spatial hash
        t0 = time.perf_counter()
        spatial_hash.build_vectorized(particles, n_particles)
        t1 = time.perf_counter()
        print(f"  Spatial hash build: {(t1-t0)*1000:.2f} ms")
        
        # Benchmark neighbor search
        t0 = time.perf_counter()
        spatial_hash.query_neighbors_vectorized(particles, n_particles, 0.2)
        t1 = time.perf_counter()
        print(f"  Neighbor search: {(t1-t0)*1000:.2f} ms")
        
        # Benchmark density
        t0 = time.perf_counter()
        compute_density_vectorized(particles, kernel, n_particles)
        t1 = time.perf_counter()
        print(f"  Density computation: {(t1-t0)*1000:.2f} ms")
        
        # Benchmark forces
        particles.pressure[:n_particles] = 1e5
        t0 = time.perf_counter()
        compute_forces_vectorized(particles, kernel, n_particles)
        t1 = time.perf_counter()
        print(f"  Force computation: {(t1-t0)*1000:.2f} ms")
        
        # Total time per step
        total_ms = ((t1-t0) + (t1-t0) + (t1-t0) + (t1-t0)) * 1000
        fps = 1000.0 / total_ms
        print(f"  Estimated FPS: {fps:.1f}")


if __name__ == "__main__":
    print("SPH Vectorized Implementation Test")
    print("==================================")
    
    # Run tests
    test_kernel()
    test_spatial_hash()
    test_density()
    test_forces()
    
    # Benchmark
    benchmark_performance()
    
    # Run simulation
    print("\nPress Enter to run dam break simulation...")
    input()
    run_dam_break_simulation()