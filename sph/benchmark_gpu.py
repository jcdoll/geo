#!/usr/bin/env python3
"""
Benchmark GPU performance improvements.

Tests different particle counts and backend configurations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

import sys
sys.path.insert(0, '..')
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.physics import MaterialType, MaterialDatabase


def create_test_particles(n_particles: int, domain_size: float = 100.0) -> tuple:
    """Create test particles in a square domain."""
    # Create particles in a grid
    n_side = int(np.sqrt(n_particles))
    spacing = domain_size / n_side
    
    particles = ParticleArrays.allocate(n_particles + 1000)
    material_db = MaterialDatabase()
    
    idx = 0
    for i in range(n_side):
        for j in range(n_side):
            if idx >= n_particles:
                break
            
            x = (i - n_side/2) * spacing + spacing/2
            y = (j - n_side/2) * spacing + spacing/2
            
            particles.position_x[idx] = x
            particles.position_y[idx] = y
            particles.velocity_x[idx] = np.random.uniform(-1, 1)
            particles.velocity_y[idx] = np.random.uniform(-1, 1)
            particles.mass[idx] = 1.0
            particles.smoothing_h[idx] = spacing * 1.3
            particles.material_id[idx] = MaterialType.WATER.value
            particles.density[idx] = 1000.0  # Water density
            particles.temperature[idx] = 300.0
            particles.pressure[idx] = 101325.0
            
            idx += 1
    
    n_active = idx
    return particles, n_active


def benchmark_backend(backend: str, particle_counts: List[int], 
                     n_steps: int = 100) -> Dict[str, List[float]]:
    """Benchmark a specific backend."""
    print(f"\nBenchmarking {backend} backend...")
    
    try:
        sph.set_backend(backend)
    except Exception as e:
        print(f"  Failed to set backend: {e}")
        return {"times": [np.nan] * len(particle_counts)}
    
    times = []
    kernel = CubicSplineKernel(dim=2)
    
    for n_particles in particle_counts:
        print(f"  Testing {n_particles} particles...")
        
        # Create particles
        particles, n_active = create_test_particles(n_particles)
        
        # Create spatial hash
        spatial_hash = sph.create_spatial_hash(
            domain_size=(100.0, 100.0),
            cell_size=2.0,
            domain_min=(-50.0, -50.0)
        )
        
        # Warmup for JIT compilation
        if backend in ['numba', 'gpu']:
            for _ in range(3):
                spatial_hash.update_particles(particles, n_active)
                spatial_hash.find_neighbors(particles, n_active, search_radius=2.0)
                sph.compute_density(particles, kernel, n_active)
                sph.compute_forces(particles, kernel, n_active, gravity=np.array([0, -9.81]))
        
        # Time the simulation
        start_time = time.time()
        
        for step in range(n_steps):
            # Update spatial hash
            spatial_hash.update_particles(particles, n_active)
            spatial_hash.find_neighbors(particles, n_active, search_radius=2.0)
            
            # Compute density
            sph.compute_density(particles, kernel, n_active)
            
            # Compute forces
            sph.compute_forces(particles, kernel, n_active, gravity=np.array([0, -9.81]))
            
            # Simple integration (just for benchmarking)
            dt = 0.001
            particles.velocity_x[:n_active] += (particles.force_x[:n_active] / particles.mass[:n_active]) * dt
            particles.velocity_y[:n_active] += (particles.force_y[:n_active] / particles.mass[:n_active]) * dt
            particles.position_x[:n_active] += particles.velocity_x[:n_active] * dt
            particles.position_y[:n_active] += particles.velocity_y[:n_active] * dt
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        steps_per_sec = n_steps / elapsed
        particles_per_sec = n_particles * steps_per_sec
        
        print(f"    Time: {elapsed:.3f}s ({steps_per_sec:.1f} steps/s, {particles_per_sec:.0f} particles/s)")
    
    return {"times": times}


def plot_results(results: Dict[str, Dict], particle_counts: List[int]):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 8))
    
    # Plot time vs particle count
    plt.subplot(2, 1, 1)
    for backend, data in results.items():
        if not all(np.isnan(data["times"])):
            plt.plot(particle_counts, data["times"], 'o-', label=backend, linewidth=2, markersize=8)
    
    plt.xlabel("Number of Particles")
    plt.ylabel("Time for 100 steps (s)")
    plt.title("SPH Backend Performance Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # Plot speedup relative to CPU
    plt.subplot(2, 1, 2)
    cpu_times = results.get("cpu", {}).get("times", [np.nan] * len(particle_counts))
    
    for backend, data in results.items():
        if backend != "cpu" and not all(np.isnan(data["times"])):
            speedup = [cpu_t / t for cpu_t, t in zip(cpu_times, data["times"])]
            plt.plot(particle_counts, speedup, 'o-', label=f"{backend} speedup", linewidth=2, markersize=8)
    
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.xlabel("Number of Particles")
    plt.ylabel("Speedup relative to CPU")
    plt.title("Backend Speedup Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig("sph_gpu_benchmark.png", dpi=150)
    print(f"\nBenchmark plot saved to sph_gpu_benchmark.png")


def main():
    """Run GPU benchmark."""
    print("SPH GPU Performance Benchmark")
    print("=" * 40)
    
    # Check available backends
    sph.print_backend_info()
    
    # Test configurations
    particle_counts = [100, 500, 1000, 2000, 5000, 10000, 20000]
    backends = ["cpu", "numba", "gpu"]
    
    # Run benchmarks
    results = {}
    for backend in backends:
        results[backend] = benchmark_backend(backend, particle_counts)
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    for i, n_particles in enumerate(particle_counts):
        print(f"\n{n_particles} particles:")
        for backend in backends:
            if backend in results:
                time_taken = results[backend]["times"][i]
                if not np.isnan(time_taken):
                    particles_per_sec = n_particles * 100 / time_taken
                    print(f"  {backend:8s}: {time_taken:8.3f}s ({particles_per_sec:12.0f} particles/s)")
    
    # Plot results
    plot_results(results, particle_counts)


if __name__ == "__main__":
    main()