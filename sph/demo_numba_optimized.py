#!/usr/bin/env python3
"""
Demo showing Numba-optimized SPH performance improvements.

Compares standard vs Numba implementations and demonstrates
the significant speedup achieved.
"""

import numpy as np
import time
import sys
from sph.core import ParticleArrays, CubicSplineKernel, integrate_leapfrog_vectorized
from sph.core.numba_switch import (
    NUMBA_AVAILABLE, SpatialHashFactory,
    compute_density_auto, compute_forces_auto, compute_gravity_auto,
    benchmark_implementations
)
from sph.physics import MaterialDatabase
from sph.scenarios import create_planet_simple


def run_optimized_simulation(n_particles=5000, n_steps=100, use_numba=True):
    """Run simulation with optional Numba optimization."""
    
    print(f"\n{'='*60}")
    print(f"Running {'Numba-optimized' if use_numba and NUMBA_AVAILABLE else 'Standard'} "
          f"simulation with {n_particles} particles")
    print(f"{'='*60}")
    
    # Create planet
    radius = np.sqrt(n_particles / 500) * 1000  # Scale radius with particle count
    spacing = radius / np.sqrt(n_particles) * 3
    
    particles, n_active = create_planet_simple(
        radius=radius,
        particle_spacing=spacing,
        center=(5000, 5000)
    )
    
    print(f"Created planet: radius={radius:.0f}m, spacing={spacing:.0f}m, "
          f"particles={n_active}")
    
    # Initialize modules
    material_db = MaterialDatabase()
    kernel = CubicSplineKernel(dim=2)
    
    # Create spatial hash (auto-selects Numba if available and requested)
    spatial_hash = SpatialHashFactory.create(
        domain_size=(10000, 10000),
        cell_size=spacing * 2,
        force_standard=not use_numba
    )
    
    # Timing arrays
    times = {
        'spatial_build': [],
        'neighbor_search': [],
        'density': [],
        'pressure': [],
        'forces': [],
        'gravity': [],
        'integration': [],
        'total': []
    }
    
    # Warmup (for JIT compilation)
    if use_numba and NUMBA_AVAILABLE:
        print("Warming up Numba JIT...")
        for _ in range(3):
            spatial_hash.build_vectorized(particles, n_active)
            spatial_hash.query_neighbors_vectorized(particles, n_active, spacing * 2)
            compute_density_auto(particles, kernel, n_active)
            compute_forces_auto(particles, kernel, n_active)
            if n_active < 2000:  # Only warmup gravity for small systems
                compute_gravity_auto(particles, n_active)
    
    # Main simulation loop
    print(f"\nRunning {n_steps} steps...")
    dt = 0.01
    
    for step in range(n_steps):
        t_start = time.perf_counter()
        
        # Spatial hash build
        t0 = time.perf_counter()
        spatial_hash.build_vectorized(particles, n_active)
        times['spatial_build'].append(time.perf_counter() - t0)
        
        # Neighbor search
        t0 = time.perf_counter()
        spatial_hash.query_neighbors_vectorized(particles, n_active, spacing * 2)
        times['neighbor_search'].append(time.perf_counter() - t0)
        
        # Density
        t0 = time.perf_counter()
        if use_numba and NUMBA_AVAILABLE:
            compute_density_auto(particles, kernel, n_active)
        else:
            from sph.physics import compute_density_vectorized
            compute_density_vectorized(particles, kernel, n_active)
        times['density'].append(time.perf_counter() - t0)
        
        # Pressure (material-aware)
        t0 = time.perf_counter()
        bulk_modulus = material_db.get_bulk_modulus_array(particles.material_id[:n_active])
        density_ref = material_db.get_density_ref_array(particles.material_id[:n_active])
        particles.pressure[:n_active] = bulk_modulus * (
            (particles.density[:n_active] / density_ref)**7 - 1
        )
        times['pressure'].append(time.perf_counter() - t0)
        
        # Forces
        t0 = time.perf_counter()
        if use_numba and NUMBA_AVAILABLE:
            compute_forces_auto(particles, kernel, n_active)
        else:
            from sph.physics import compute_forces_vectorized
            compute_forces_vectorized(particles, kernel, n_active)
        times['forces'].append(time.perf_counter() - t0)
        
        # Gravity (only for smaller systems)
        t0 = time.perf_counter()
        if n_active < 5000:
            if use_numba and NUMBA_AVAILABLE:
                compute_gravity_auto(particles, n_active, G=6.67430e-11 * 1e6)
            else:
                from sph.physics import compute_gravity_direct_batched
                compute_gravity_direct_batched(particles, n_active, G=6.67430e-11 * 1e6)
        times['gravity'].append(time.perf_counter() - t0)
        
        # Integration
        t0 = time.perf_counter()
        integrate_leapfrog_vectorized(particles, n_active, dt, (0, 10000, 0, 10000))
        times['integration'].append(time.perf_counter() - t0)
        
        times['total'].append(time.perf_counter() - t_start)
        
        # Progress
        if (step + 1) % 20 == 0:
            avg_time = np.mean(times['total'][-20:])
            fps = 1.0 / avg_time
            print(f"  Step {step+1}/{n_steps}: {avg_time*1000:.1f} ms/step, {fps:.1f} FPS")
    
    # Report timing breakdown
    print(f"\n{'='*50}")
    print("Timing Breakdown (average over all steps):")
    print(f"{'='*50}")
    
    total_avg = np.mean(times['total']) * 1000
    
    for name, time_list in times.items():
        if name == 'total':
            continue
        avg_ms = np.mean(time_list) * 1000
        percentage = avg_ms / total_avg * 100
        print(f"{name:15s}: {avg_ms:7.2f} ms ({percentage:5.1f}%)")
    
    print(f"{'-'*50}")
    print(f"{'Total':15s}: {total_avg:7.2f} ms")
    print(f"{'FPS':15s}: {1000.0/total_avg:7.1f}")
    
    return times


def compare_implementations(particle_counts=[1000, 2000, 5000]):
    """Compare standard vs Numba performance across different particle counts."""
    
    if not NUMBA_AVAILABLE:
        print("\nNumba not available - cannot compare implementations")
        print("Install with: pip install numba")
        return
    
    print("\n" + "="*60)
    print("Performance Comparison: Standard vs Numba")
    print("="*60)
    
    results = {}
    
    for n_particles in particle_counts:
        print(f"\n{n_particles} particles:")
        
        # Run standard
        times_std = run_optimized_simulation(n_particles, n_steps=50, use_numba=False)
        fps_std = 1000.0 / (np.mean(times_std['total']) * 1000)
        
        # Run Numba
        times_numba = run_optimized_simulation(n_particles, n_steps=50, use_numba=True)
        fps_numba = 1000.0 / (np.mean(times_numba['total']) * 1000)
        
        # Store results
        results[n_particles] = {
            'fps_std': fps_std,
            'fps_numba': fps_numba,
            'speedup': fps_numba / fps_std
        }
    
    # Summary table
    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(f"{'Particles':>10} | {'Standard FPS':>12} | {'Numba FPS':>12} | {'Speedup':>8}")
    print("-"*60)
    
    for n_particles in particle_counts:
        r = results[n_particles]
        print(f"{n_particles:10d} | {r['fps_std']:12.1f} | {r['fps_numba']:12.1f} | "
              f"{r['speedup']:7.1f}x")
    
    # Component speedups
    print("\n" + "="*60)
    print("Component Speedups (5000 particles)")
    print("="*60)
    benchmark_implementations(5000, 5)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--benchmark':
            # Detailed benchmark
            benchmark_implementations(5000, 10)
        elif sys.argv[1] == '--compare':
            # Compare implementations
            compare_implementations([1000, 2000, 5000, 10000])
        elif sys.argv[1] == '--large':
            # Large simulation
            run_optimized_simulation(20000, n_steps=100, use_numba=True)
    else:
        # Default: medium simulation with Numba
        run_optimized_simulation(5000, n_steps=100, use_numba=True)