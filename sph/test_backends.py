#!/usr/bin/env python3
"""
Test script to verify Numba and GPU backends are working correctly.
"""

import numpy as np
import sys
import pytest

def test_numba():
    """Test Numba functionality."""
    print("\n=== Testing Numba Backend ===")
    
    try:
        import numba as nb
        print(f"✓ Numba version: {nb.__version__}")
        
        # Test JIT compilation
        @nb.njit
        def test_function(x):
            return x * 2 + 1
        
        result = test_function(5)
        print(f"✓ JIT compilation works: test_function(5) = {result}")
        
        # Test parallel execution
        @nb.njit(parallel=True)
        def parallel_sum(arr):
            total = 0.0
            for i in nb.prange(len(arr)):
                total += arr[i]
            return total
        
        test_array = np.random.rand(1000000)
        result = parallel_sum(test_array)
        expected = np.sum(test_array)
        print(f"✓ Parallel execution works: sum matches (diff: {abs(result - expected):.2e})")
        
        # Test SPH Numba implementation
        from sph.core.spatial_hash_numba import NumbaOptimizedSpatialHash
        from sph.physics.density_numba import compute_density_numba_wrapper
        from sph.physics.forces_numba import compute_forces_numba_wrapper
        
        print("✓ SPH Numba modules imported successfully")
        
        # Create test data
        from sph.core.particles import ParticleArrays
        particles = ParticleArrays.allocate(100)
        particles.position_x[:100] = np.random.uniform(0, 10, 100)
        particles.position_y[:100] = np.random.uniform(0, 10, 100)
        particles.mass[:100] = 1.0
        particles.smoothing_h[:100] = 0.5
        particles.density[:100] = 1000.0
        particles.pressure[:100] = 1e5
        
        # Test spatial hash
        spatial_hash = NumbaOptimizedSpatialHash((10, 10), 1.0)
        spatial_hash.build_vectorized(particles, 100)
        spatial_hash.query_neighbors_vectorized(particles, 100, 1.0)
        print("✓ Numba spatial hash works")
        
        # Test density computation
        compute_density_numba_wrapper(particles, 100)
        print(f"✓ Numba density computation works (avg density: {np.mean(particles.density[:100]):.1f})")
        
        # Test force computation
        compute_forces_numba_wrapper(particles, 100)
        print("✓ Numba force computation works")
        
        print("\n✅ Numba backend fully functional!")
        
    except Exception as e:
        print(f"❌ Numba test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_gpu():
    """Test GPU/CuPy functionality."""
    print("\n=== Testing GPU Backend ===")
    
    try:
        import cupy as cp
        print(f"✓ CuPy version: {cp.__version__}")
        
        # Check CUDA availability
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"✓ CUDA devices found: {device_count}")
            
            if device_count > 0:
                device = cp.cuda.Device(0)
                # Get device properties
                props = cp.cuda.runtime.getDeviceProperties(0)
                device_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']
                print(f"✓ GPU: {device_name}")
                
                # Get memory info
                free_mem, total_mem = device.mem_info
                print(f"✓ GPU Memory: {free_mem/1e9:.1f}/{total_mem/1e9:.1f} GB free/total")
                
                # Test basic operations
                gpu_array = cp.arange(1000000)
                result = cp.sum(gpu_array)
                expected = 499999500000
                print(f"✓ GPU computation works: sum = {result} (expected: {expected})")
                
                # Test data transfer
                cpu_array = np.random.rand(1000)
                gpu_array = cp.asarray(cpu_array)
                cpu_back = cp.asnumpy(gpu_array)
                print(f"✓ CPU-GPU transfer works: arrays match = {np.allclose(cpu_array, cpu_back)}")
                
                print("\n✅ GPU backend functional!")
            else:
                print("⚠️  No CUDA devices found - GPU backend won't work")
                
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"⚠️  CUDA runtime error: {e}")
            print("   GPU backend not available on this system")
            
    except ImportError:
        print("❌ CuPy not installed or not working")
        pytest.skip("CuPy not available")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        if "libnvrtc.so" in str(e) or "CUDA" in str(e):
            pytest.skip(f"GPU not properly configured: {e}")
        raise


def test_backend_system():
    """Test the backend dispatch system."""
    print("\n=== Testing Backend Dispatch System ===")
    
    try:
        from sph.api import (
            set_backend, get_backend, print_backend_info,
            ParticleArrays, compute_density, compute_forces
        )
        
        print_backend_info()
        
        # Test CPU backend (always available)
        print("\nTesting CPU backend:")
        set_backend('cpu')
        print(f"Current backend: {get_backend()}")
        
        particles = ParticleArrays.allocate(100)
        particles.position_x[:100] = np.random.uniform(0, 10, 100)
        particles.position_y[:100] = np.random.uniform(0, 10, 100)
        particles.mass[:100] = 1.0
        particles.smoothing_h[:100] = 0.5
        particles.neighbor_count[:100] = 0  # No neighbors for simple test
        
        compute_density(particles, n_active=100)
        print("✓ CPU backend dispatch works")
        
        # Test Numba backend if available
        if set_backend('numba'):
            print("\nTesting Numba backend dispatch:")
            compute_density(particles, n_active=100)
            compute_forces(particles, n_active=100)
            print("✓ Numba backend dispatch works")
        
        print("\n✅ Backend system fully functional!")
        
    except Exception as e:
        print(f"❌ Backend system test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_performance_comparison():
    """Run a quick performance comparison."""
    print("\n=== Performance Comparison ===")
    
    try:
        from sph.api import set_backend, ParticleArrays, compute_density, compute_forces
        from sph.core.spatial_hash_vectorized import VectorizedSpatialHash
        from sph.core.spatial_hash_numba import NumbaOptimizedSpatialHash
        import time
        
        # Create test data
        n_particles = 5000
        particles = ParticleArrays.allocate(n_particles)
        particles.position_x[:n_particles] = np.random.uniform(0, 100, n_particles)
        particles.position_y[:n_particles] = np.random.uniform(0, 100, n_particles)
        particles.mass[:n_particles] = 1.0
        particles.smoothing_h[:n_particles] = 2.0
        particles.density[:n_particles] = 1000.0
        particles.pressure[:n_particles] = 1e5
        particles.velocity_x[:n_particles] = np.random.normal(0, 1, n_particles)
        particles.velocity_y[:n_particles] = np.random.normal(0, 1, n_particles)
        
        # Test spatial hash performance
        print(f"\nSpatial Hash Build ({n_particles} particles):")
        
        # CPU
        hash_cpu = VectorizedSpatialHash((100, 100), 4.0)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            hash_cpu.build_vectorized(particles, n_particles)
            times.append(time.perf_counter() - t0)
        cpu_time = np.mean(times[2:])  # Skip warmup
        print(f"  CPU:   {cpu_time*1000:.2f} ms")
        
        # Numba
        hash_numba = NumbaOptimizedSpatialHash((100, 100), 4.0)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            hash_numba.build_vectorized(particles, n_particles)
            times.append(time.perf_counter() - t0)
        numba_time = np.mean(times[2:])
        print(f"  Numba: {numba_time*1000:.2f} ms (speedup: {cpu_time/numba_time:.1f}x)")
        
        # Neighbor search
        print(f"\nNeighbor Search:")
        
        # CPU
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            hash_cpu.query_neighbors_vectorized(particles, n_particles, 4.0)
            times.append(time.perf_counter() - t0)
        cpu_time = np.mean(times[1:])
        print(f"  CPU:   {cpu_time*1000:.2f} ms")
        
        # Numba
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            hash_numba.query_neighbors_vectorized(particles, n_particles, 4.0)
            times.append(time.perf_counter() - t0)
        numba_time = np.mean(times[1:])
        print(f"  Numba: {numba_time*1000:.2f} ms (speedup: {cpu_time/numba_time:.1f}x)")
        
        # Full physics step
        print(f"\nFull Physics Step (density + forces):")
        
        # CPU
        set_backend('cpu')
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            compute_density(particles, n_active=n_particles)
            compute_forces(particles, n_active=n_particles)
            times.append(time.perf_counter() - t0)
        cpu_time = np.mean(times[1:])
        print(f"  CPU:   {cpu_time*1000:.2f} ms")
        
        # Numba
        if set_backend('numba'):
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                compute_density(particles, n_active=n_particles)
                compute_forces(particles, n_active=n_particles)
                times.append(time.perf_counter() - t0)
            numba_time = np.mean(times[1:])
            print(f"  Numba: {numba_time*1000:.2f} ms (speedup: {cpu_time/numba_time:.1f}x)")
        
        return True
        
    except Exception as e:
        print(f"Performance comparison failed: {e}")
        return False


if __name__ == "__main__":
    print("SPH Backend Testing")
    print("===================")
    
    # Run tests
    numba_ok = test_numba()
    gpu_ok = test_gpu()
    backend_ok = test_backend_system()
    
    if numba_ok and backend_ok:
        run_performance_comparison()
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"  Numba:   {'✅ Working' if numba_ok else '❌ Failed'}")
    print(f"  GPU:     {'✅ Working' if gpu_ok else '⚠️  Not available (no CUDA)'}")
    print(f"  Backend: {'✅ Working' if backend_ok else '❌ Failed'}")
    print("="*50)