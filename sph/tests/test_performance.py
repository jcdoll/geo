"""
Performance benchmarking for SPH implementations.

Compares performance across CPU, Numba, and GPU backends.
"""

import numpy as np
import time
import pytest
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.spatial_hash_vectorized import VectorizedSpatialHash, find_neighbors_vectorized


class BenchmarkResults:
    """Store and display benchmark results."""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, backend, n_particles, operation, time_ms):
        """Add a benchmark result."""
        if backend not in self.results:
            self.results[backend] = {}
        if n_particles not in self.results[backend]:
            self.results[backend][n_particles] = {}
        self.results[backend][n_particles][operation] = time_ms
    
    def print_summary(self):
        """Print benchmark summary."""
        if not self.results:
            return
        
        # Get all particle counts and operations
        all_n = set()
        all_ops = set()
        for backend_data in self.results.values():
            all_n.update(backend_data.keys())
            for ops in backend_data.values():
                all_ops.update(ops.keys())
        
        all_n = sorted(all_n)
        all_ops = sorted(all_ops)
        backends = sorted(self.results.keys())
        
        print("\n" + "="*80)
        print("SPH PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        for n in all_n:
            print(f"\n{n} particles:")
            print("-" * 50)
            
            # Print times
            for backend in backends:
                if n in self.results[backend]:
                    print(f"\n{backend.upper()}:")
                    for op in all_ops:
                        if op in self.results[backend][n]:
                            time_ms = self.results[backend][n][op]
                            print(f"  {op:20s}: {time_ms:8.2f} ms")
            
            # Print speedups relative to CPU
            if 'cpu' in backends and n in self.results['cpu']:
                print("\nSpeedup vs CPU:")
                cpu_times = self.results['cpu'][n]
                
                for backend in backends:
                    if backend != 'cpu' and n in self.results[backend]:
                        print(f"\n{backend.upper()}:")
                        for op in all_ops:
                            if op in cpu_times and op in self.results[backend][n]:
                                speedup = cpu_times[op] / self.results[backend][n][op]
                                print(f"  {op:20s}: {speedup:6.1f}x")


def benchmark_backend(backend, n_particles, use_neighbors=True, n_iterations=10):
    """Benchmark a single backend."""
    try:
        sph.set_backend(backend)
    except Exception as e:
        print(f"Backend {backend} not available: {e}")
        return None
    
    # Create particles
    particles = ParticleArrays.allocate(n_particles, max_neighbors=64)
    kernel = CubicSplineKernel()
    
    # Initialize particles in a box
    box_size = np.sqrt(n_particles) * 0.05
    particles.position_x[:] = np.random.uniform(0, box_size, n_particles)
    particles.position_y[:] = np.random.uniform(0, box_size, n_particles)
    particles.velocity_x[:] = np.random.uniform(-1, 1, n_particles)
    particles.velocity_y[:] = np.random.uniform(-1, 1, n_particles)
    particles.mass[:] = 1.0
    particles.smoothing_h[:] = 0.1
    
    # Find neighbors if requested
    if use_neighbors:
        spatial_hash = VectorizedSpatialHash(
            domain_size=(box_size, box_size),
            cell_size=0.2,
            domain_min=(0, 0)
        )
        find_neighbors_vectorized(particles, spatial_hash, n_particles, search_radius=0.2)
    
    # Warm-up (important for GPU)
    sph.compute_density(particles, kernel, n_particles)
    sph.compute_pressure(particles, n_particles)
    sph.compute_forces(particles, kernel, n_particles)
    if backend == 'gpu':
        # Extra warmup for GPU
        for _ in range(5):
            sph.compute_density(particles, kernel, n_particles)
    
    results = {}
    
    # Benchmark density computation
    start = time.time()
    for _ in range(n_iterations):
        sph.compute_density(particles, kernel, n_particles)
    results['density'] = (time.time() - start) / n_iterations * 1000
    
    # Benchmark pressure computation
    start = time.time()
    for _ in range(n_iterations):
        sph.compute_pressure(particles, n_particles)
    results['pressure'] = (time.time() - start) / n_iterations * 1000
    
    # Benchmark force computation
    start = time.time()
    for _ in range(n_iterations):
        sph.compute_forces(particles, kernel, n_particles)
    results['forces'] = (time.time() - start) / n_iterations * 1000
    
    # Benchmark integration
    start = time.time()
    for _ in range(n_iterations):
        sph.integrate(particles, n_particles, dt=0.001)
    results['integration'] = (time.time() - start) / n_iterations * 1000
    
    # Benchmark full timestep
    start = time.time()
    for _ in range(n_iterations):
        sph.compute_density(particles, kernel, n_particles)
        sph.compute_pressure(particles, n_particles)
        sph.compute_forces(particles, kernel, n_particles)
        sph.integrate(particles, n_particles, dt=0.001)
    results['full_timestep'] = (time.time() - start) / n_iterations * 1000
    
    return results


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for pytest.
    
    Requires pytest-benchmark to be installed. Tests will be skipped if not available.
    Install with: pip install pytest-benchmark
    """
    
    @pytest.fixture
    def benchmark(self, request):
        """Provide benchmark fixture or skip if not available."""
        # Check if pytest-benchmark is installed
        try:
            from pytest_benchmark.fixture import BenchmarkFixture
            # Get the actual benchmark fixture from pytest-benchmark
            return request.getfixturevalue('benchmark')
        except (ImportError, pytest.FixtureLookupError):
            pytest.skip("pytest-benchmark not installed. Install with: pip install pytest-benchmark")
    
    @pytest.mark.parametrize("n_particles", [1000, 5000, 10000, 50000])
    @pytest.mark.parametrize("backend", ['cpu', 'numba', 'gpu'])
    def test_performance(self, n_particles, backend, benchmark):
        """Benchmark SPH operations."""
        # Skip if benchmark fixture not available (already handled above)
        if benchmark is None:
            pytest.skip("Benchmark fixture not available")
        
        # Skip if backend not available
        try:
            sph.set_backend(backend)
        except:
            pytest.skip(f"Backend {backend} not available")
        
        # Create particles
        particles = ParticleArrays.allocate(n_particles)
        kernel = CubicSplineKernel()
        
        # Initialize
        particles.position_x[:] = np.random.uniform(0, 10, n_particles)
        particles.position_y[:] = np.random.uniform(0, 10, n_particles)
        particles.mass[:] = 1.0
        particles.smoothing_h[:] = 0.1
        
        # Benchmark full timestep
        def timestep():
            sph.compute_density(particles, kernel, n_particles)
            sph.compute_pressure(particles, n_particles)
            sph.compute_forces(particles, kernel, n_particles)
            sph.integrate(particles, n_particles, dt=0.001)
        
        # Run benchmark
        benchmark(timestep)


def main():
    """Run performance benchmarks."""
    # Particle counts to test
    particle_counts = [1000, 5000, 10000, 25000, 50000]
    backends = ['cpu', 'numba', 'gpu']
    
    results = BenchmarkResults()
    
    # Check available backends
    print("Checking available backends...")
    available_backends = []
    for backend in backends:
        try:
            sph.set_backend(backend)
            available_backends.append(backend)
            print(f"  {backend}: ✓")
        except Exception as e:
            print(f"  {backend}: ✗ ({e})")
    
    print(f"\nRunning benchmarks with backends: {available_backends}")
    
    # Run benchmarks
    for n in particle_counts:
        print(f"\nBenchmarking with {n} particles...")
        
        for backend in available_backends:
            print(f"  {backend}...", end='', flush=True)
            
            # Skip very large particle counts on CPU
            if backend == 'cpu' and n > 10000:
                print(" skipped (too slow)")
                continue
            
            result = benchmark_backend(backend, n, use_neighbors=(n < 10000))
            
            if result:
                for op, time_ms in result.items():
                    results.add_result(backend, n, op, time_ms)
                print(f" done ({result['full_timestep']:.1f} ms/step)")
            else:
                print(" failed")
    
    # Print results
    results.print_summary()
    
    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n1. For small simulations (<5k particles): Use Numba backend")
    print("2. For medium simulations (5k-50k particles): Use Numba or GPU")
    print("3. For large simulations (>50k particles): Use GPU backend")
    print("4. For development/debugging: Use CPU backend")
    print("\nNote: GPU performance improves significantly with larger particle counts")
    print("      due to better utilization of parallel compute units.")


if __name__ == "__main__":
    main()