"""
Comprehensive performance benchmark for flux simulation.

Tests all major components and scenarios to detect performance regressions.
"""

import pytest
import numpy as np
import time
from typing import Dict, List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from gravity_solver import GravitySolver, SolverMethod
from state import FluxState
from multigrid import solve_mac_poisson_vectorized, BoundaryCondition


class PerformanceBenchmark:
    """Run performance benchmarks on simulation components."""
    
    @staticmethod
    def benchmark_gravity_solvers(size: int = 64, iterations: int = 10) -> Dict[str, float]:
        """Benchmark gravity solver methods."""
        # Create test state with circular mass
        state = FluxState(size, size, 50.0)
        center = size // 2
        radius = size // 3
        
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((j - center)**2 + (i - center)**2)
                if dist < radius:
                    state.density[i, j] = 5000.0
                else:
                    state.density[i, j] = 0.1
        
        results = {}
        
        # Test DFT solver
        solver_dft = GravitySolver(state, method=SolverMethod.DFT)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            solver_dft.solve_gravity()
            times.append(time.perf_counter() - t0)
        results['gravity_dft'] = np.mean(times[1:])  # Skip first for warmup
        
        # Test Multigrid solver
        solver_mg = GravitySolver(state, method=SolverMethod.MULTIGRID)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            solver_mg.solve_gravity()
            times.append(time.perf_counter() - t0)
        results['gravity_multigrid'] = np.mean(times[1:])
        
        return results
    
    @staticmethod
    def benchmark_multigrid_neumann(size: int = 64, iterations: int = 10) -> Dict[str, float]:
        """Benchmark multigrid solver with Neumann BC."""
        # Create test problem
        rhs = np.ones((size, size))
        beta_x = np.ones((size, size + 1), dtype=np.float32)
        beta_y = np.ones((size + 1, size), dtype=np.float32)
        
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            phi = solve_mac_poisson_vectorized(
                rhs, beta_x, beta_y, 50.0,
                bc_type=BoundaryCondition.NEUMANN,
                tol=1e-6, max_cycles=20, verbose=False
            )
            times.append(time.perf_counter() - t0)
        
        return {'multigrid_neumann': np.mean(times[1:])}
    
    @staticmethod
    def benchmark_full_timestep(scenario: str, size: int = 64, steps: int = 20) -> Dict[str, float]:
        """Benchmark full simulation timestep."""
        sim = FluxSimulation(nx=size, ny=size, dx=50.0, scenario=scenario)
        sim.paused = False  # Unpause for testing
        
        # Warmup
        for _ in range(5):
            sim.step_forward()
        
        # Measure
        times = []
        for _ in range(steps):
            t0 = time.perf_counter()
            sim.step_forward()
            times.append(time.perf_counter() - t0)
        
        # Get component timings if available
        results = {
            f'{scenario}_total': np.mean(times),
            f'{scenario}_min': np.min(times),
            f'{scenario}_max': np.max(times)
        }
        
        # Add component timings
        if hasattr(sim, 'step_timings') and sim.step_timings:
            for key, value in sim.step_timings.items():
                results[f'{scenario}_{key}'] = value
        
        return results


# Pytest benchmarks
@pytest.mark.parametrize("size", [32, 64, 128])
def test_gravity_solver_performance(size):
    """Test gravity solver performance at different grid sizes."""
    results = PerformanceBenchmark.benchmark_gravity_solvers(size=size, iterations=5)
    
    print(f"\nGravity solver performance ({size}x{size}):")
    for method, time_sec in results.items():
        print(f"  {method}: {time_sec*1000:.1f}ms")
    
    # Check performance bounds
    if size == 32:
        assert results['gravity_dft'] < 0.005  # 5ms
        assert results['gravity_multigrid'] < 0.010  # 10ms
    elif size == 64:
        assert results['gravity_dft'] < 0.020  # 20ms
        assert results['gravity_multigrid'] < 0.040  # 40ms
    elif size == 128:
        assert results['gravity_dft'] < 0.080  # 80ms
        assert results['gravity_multigrid'] < 0.200  # 200ms


def test_multigrid_neumann_performance():
    """Test improved Neumann BC performance."""
    results = PerformanceBenchmark.benchmark_multigrid_neumann(size=64, iterations=10)
    
    print(f"\nMultigrid Neumann BC performance:")
    print(f"  Time: {results['multigrid_neumann']*1000:.1f}ms")
    
    # Should be fast with vectorized implementation
    assert results['multigrid_neumann'] < 0.050  # 50ms for 64x64


@pytest.mark.parametrize("scenario", ["empty", "layered", "volcanic"])
def test_full_simulation_performance(scenario):
    """Test full simulation performance."""
    results = PerformanceBenchmark.benchmark_full_timestep(scenario, size=64, steps=20)
    
    print(f"\n{scenario.upper()} simulation performance (64x64):")
    print(f"  Average: {results[f'{scenario}_total']*1000:.1f}ms")
    print(f"  Min: {results[f'{scenario}_min']*1000:.1f}ms")
    print(f"  Max: {results[f'{scenario}_max']*1000:.1f}ms")
    
    # Component breakdown if available
    for key, value in results.items():
        if key.endswith('_total') or key.endswith('_min') or key.endswith('_max'):
            continue
        if scenario in key:
            component = key.replace(f'{scenario}_', '')
            print(f"  {component}: {value*1000:.1f}ms")
    
    # Performance bounds
    if scenario == "empty":
        assert results[f'{scenario}_total'] < 0.050  # 50ms
    else:
        assert results[f'{scenario}_total'] < 0.200  # 200ms


def test_scaling_analysis():
    """Analyze performance scaling with grid size."""
    sizes = [32, 64, 128]
    scenario = "layered"
    
    print(f"\nScaling analysis for {scenario} scenario:")
    print("Size\tTime(ms)\tScaling")
    
    base_time = None
    base_size = None
    
    for size in sizes:
        results = PerformanceBenchmark.benchmark_full_timestep(scenario, size=size, steps=10)
        avg_time = results[f'{scenario}_total']
        
        if base_time is None:
            base_time = avg_time
            base_size = size
            scaling = 1.0
        else:
            # Expected O(N²) scaling
            expected_scaling = (size / base_size) ** 2
            actual_scaling = avg_time / base_time
            scaling = actual_scaling / expected_scaling
        
        print(f"{size}\t{avg_time*1000:.1f}\t{scaling:.2f}x")


def test_no_performance_regression():
    """Test that there are no major performance regressions."""
    # Run volcanic scenario as stress test
    results = PerformanceBenchmark.benchmark_full_timestep("volcanic", size=100, steps=10)
    
    avg_time_ms = results['volcanic_total'] * 1000
    print(f"\nStress test (100x100 volcanic): {avg_time_ms:.1f}ms per step")
    
    # Should achieve at least 5 FPS on 100x100
    assert avg_time_ms < 200  # 200ms = 5 FPS


if __name__ == "__main__":
    print("Running performance benchmarks...\n")
    
    # Gravity solvers
    print("=== Gravity Solver Benchmarks ===")
    for size in [32, 64, 128]:
        results = PerformanceBenchmark.benchmark_gravity_solvers(size=size)
        print(f"\n{size}x{size} grid:")
        for method, time_sec in results.items():
            print(f"  {method}: {time_sec*1000:.1f}ms")
    
    # Multigrid
    print("\n=== Multigrid Neumann BC ===")
    results = PerformanceBenchmark.benchmark_multigrid_neumann(size=64)
    print(f"  64x64: {results['multigrid_neumann']*1000:.1f}ms")
    
    # Full simulation
    print("\n=== Full Simulation Benchmarks ===")
    for scenario in ["empty", "layered", "volcanic"]:
        results = PerformanceBenchmark.benchmark_full_timestep(scenario, size=64)
        print(f"\n{scenario}:")
        print(f"  Average: {results[f'{scenario}_total']*1000:.1f}ms/step")
    
    # Scaling
    print("\n=== Scaling Analysis ===")
    test_scaling_analysis()