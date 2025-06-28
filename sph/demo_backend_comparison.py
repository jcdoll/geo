#!/usr/bin/env python3
"""
Demo comparing CPU, Numba, and GPU backends for SPH.

Shows performance differences and automatic backend selection.
"""

import numpy as np
import time
import sys
from typing import Dict, List

# Use the unified API
from sph.api import (
    ParticleArrays, CubicSplineKernel,
    compute_density, compute_forces, compute_gravity,
    create_spatial_hash,
    set_backend, get_backend, auto_select_backend, print_backend_info
)
from sph.core.integrator_vectorized import integrate_leapfrog_vectorized
from sph.physics import MaterialDatabase
from sph.scenarios import create_planet_simple


def benchmark_backend(backend: str, n_particles: int, n_steps: int = 50) -> Dict[str, float]:
    """Benchmark a specific backend.
    
    Returns:
        Dictionary with timing results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {backend.upper()} backend with {n_particles} particles")
    print(f"{'='*60}")
    
    # Set backend
    success = set_backend(backend)
    if not success:
        print(f"Backend {backend} not available, skipping...")
        return {}
    
    # Create simulation
    radius = np.sqrt(n_particles / 500) * 1000
    spacing = radius / np.sqrt(n_particles) * 3
    
    particles, n_active = create_planet_simple(
        radius=radius,
        particle_spacing=spacing,
        center=(5000, 5000)
    )
    
    # Initialize
    material_db = MaterialDatabase()
    kernel = CubicSplineKernel(dim=2)
    spatial_hash = create_spatial_hash(
        domain_size=(10000, 10000),
        cell_size=spacing * 2
    )
    
    # Warmup for JIT
    if backend in ['numba', 'gpu']:
        print("Warming up JIT/GPU...")
        for _ in range(3):
            spatial_hash.build_vectorized(particles, n_active)
            spatial_hash.query_neighbors_vectorized(particles, n_active, spacing * 2)
            compute_density(particles, kernel, n_active)
            compute_forces(particles, kernel, n_active)
            if n_active < 5000:
                compute_gravity(particles, n_active)
    
    # Timing storage
    times = {
        'spatial_hash': [],
        'neighbor_search': [],
        'density': [],
        'forces': [],
        'gravity': [],
        'total': []
    }
    
    # Run simulation
    print(f"Running {n_steps} steps...")
    dt = 0.01
    
    for step in range(n_steps):
        t_start = time.perf_counter()
        
        # Spatial hash
        t0 = time.perf_counter()
        spatial_hash.build_vectorized(particles, n_active)
        times['spatial_hash'].append(time.perf_counter() - t0)
        
        # Neighbors
        t0 = time.perf_counter()
        spatial_hash.query_neighbors_vectorized(particles, n_active, spacing * 2)
        times['neighbor_search'].append(time.perf_counter() - t0)
        
        # Density
        t0 = time.perf_counter()
        compute_density(particles, kernel, n_active)
        times['density'].append(time.perf_counter() - t0)
        
        # Pressure
        bulk_modulus = material_db.get_bulk_modulus_array(particles.material_id[:n_active])
        density_ref = material_db.get_density_ref_array(particles.material_id[:n_active])
        particles.pressure[:n_active] = bulk_modulus * (
            (particles.density[:n_active] / density_ref)**7 - 1
        )
        
        # Forces
        t0 = time.perf_counter()
        compute_forces(particles, kernel, n_active)
        times['forces'].append(time.perf_counter() - t0)
        
        # Gravity (only for smaller systems)
        t0 = time.perf_counter()
        if n_active < 5000:
            compute_gravity(particles, n_active, G=6.67430e-11 * 1e6)
        times['gravity'].append(time.perf_counter() - t0)
        
        # Integration
        integrate_leapfrog_vectorized(particles, n_active, dt, (0, 10000, 0, 10000))
        
        times['total'].append(time.perf_counter() - t_start)
        
        # Progress
        if (step + 1) % 10 == 0:
            avg_time = np.mean(times['total'][-10:])
            fps = 1.0 / avg_time
            print(f"  Step {step+1}: {avg_time*1000:.1f} ms, {fps:.1f} FPS")
    
    # Compute averages
    results = {}
    for key, values in times.items():
        if values:
            results[key] = np.mean(values)
    
    return results


def compare_all_backends(particle_counts: List[int] = [1000, 5000, 10000]):
    """Compare all available backends."""
    
    print_backend_info()
    
    backends = ['cpu', 'numba', 'gpu']
    results = {}
    
    for n_particles in particle_counts:
        results[n_particles] = {}
        
        for backend in backends:
            timing = benchmark_backend(backend, n_particles, n_steps=30)
            if timing:
                results[n_particles][backend] = timing
    
    # Print comparison table
    print("\n" + "="*80)
    print("Performance Comparison Summary")
    print("="*80)
    
    # FPS comparison
    print("\nFrames Per Second (FPS):")
    print(f"{'Particles':<10}", end='')
    for backend in backends:
        print(f"{backend.upper():>15}", end='')
    print()
    print("-"*55)
    
    for n_particles in particle_counts:
        print(f"{n_particles:<10}", end='')
        for backend in backends:
            if backend in results[n_particles]:
                fps = 1.0 / results[n_particles][backend]['total']
                print(f"{fps:>15.1f}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        print()
    
    # Speedup comparison
    print("\nSpeedup vs CPU:")
    print(f"{'Particles':<10}", end='')
    for backend in ['numba', 'gpu']:
        print(f"{backend.upper():>15}", end='')
    print()
    print("-"*40)
    
    for n_particles in particle_counts:
        print(f"{n_particles:<10}", end='')
        
        if 'cpu' in results[n_particles]:
            cpu_time = results[n_particles]['cpu']['total']
            
            for backend in ['numba', 'gpu']:
                if backend in results[n_particles]:
                    backend_time = results[n_particles][backend]['total']
                    speedup = cpu_time / backend_time
                    print(f"{speedup:>14.1f}x", end='')
                else:
                    print(f"{'N/A':>15}", end='')
        else:
            print(f"{'N/A':>15}" * 2, end='')
        print()
    
    # Component breakdown for largest system
    if particle_counts:
        n = particle_counts[-1]
        if n in results and results[n]:
            print(f"\nComponent Breakdown ({n} particles):")
            print(f"{'Component':<20}", end='')
            for backend in backends:
                if backend in results[n]:
                    print(f"{backend.upper():>15}", end='')
            print()
            print("-"*65)
            
            components = ['spatial_hash', 'neighbor_search', 'density', 'forces', 'gravity']
            for comp in components:
                print(f"{comp:<20}", end='')
                for backend in backends:
                    if backend in results[n] and comp in results[n][backend]:
                        time_ms = results[n][backend][comp] * 1000
                        print(f"{time_ms:>14.1f}ms", end='')
                    else:
                        print(f"{'--':>15}", end='')
                print()


def demo_auto_backend():
    """Demonstrate automatic backend selection."""
    print("\n" + "="*60)
    print("Automatic Backend Selection Demo")
    print("="*60)
    
    test_sizes = [500, 5000, 50000, 100000]
    
    for n_particles in test_sizes:
        # Auto-select backend
        selected = auto_select_backend(n_particles)
        print(f"\n{n_particles:,} particles -> Selected backend: {selected.upper()}")
        
        # Quick performance test
        if n_particles <= 10000:  # Don't run huge tests
            try:
                radius = np.sqrt(n_particles / 500) * 1000
                spacing = radius / np.sqrt(n_particles) * 3
                
                particles, n_active = create_planet_simple(
                    radius=radius,
                    particle_spacing=spacing
                )
                
                kernel = CubicSplineKernel()
                
                # Time one iteration
                t0 = time.perf_counter()
                compute_density(particles, kernel, n_active)
                compute_forces(particles, kernel, n_active)
                t1 = time.perf_counter()
                
                fps_estimate = 1.0 / (t1 - t0)
                print(f"  Estimated FPS: {fps_estimate:.1f}")
                
            except Exception as e:
                print(f"  Test failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto':
            demo_auto_backend()
        elif sys.argv[1] == '--full':
            compare_all_backends([1000, 2000, 5000, 10000, 20000])
        elif sys.argv[1] == '--gpu':
            # Test GPU only
            set_backend('gpu')
            print(f"Current backend: {get_backend()}")
            benchmark_backend('gpu', 50000, n_steps=100)
    else:
        # Default: compare small to medium systems
        compare_all_backends([1000, 5000, 10000])