# SPH Test Suite

This directory contains comprehensive tests for the SPH (Smoothed Particle Hydrodynamics) implementation.

## Test Categories

### 1. Backend Tests (`test_backends.py`)
- Tests all three backends (CPU, Numba, GPU) for correctness
- Ensures consistent results across backends
- Validates individual operations (density, pressure, forces, integration)

### 2. Physics Tests (`test_physics.py`)
- Validates physical correctness:
  - Hydrostatic equilibrium
  - Momentum conservation
  - Energy behavior
  - Numerical stability

### 3. Performance Tests (`test_performance.py`)
- Benchmarks performance across backends
- Tests with various particle counts (1k to 50k)
- Provides recommendations for backend selection

## Running Tests

### Run all tests:
```bash
python run_all_tests.py
```

### Run individual test suites:
```bash
# Backend tests
python test_backends.py

# Physics validation
python test_physics.py

# Performance benchmarks
python test_performance.py
```

### Run with pytest:
```bash
# Run all tests
pytest

# Run specific test
pytest test_backends.py::TestBackends::test_density_computation

# Run benchmarks
pytest test_performance.py -v --benchmark-only
```

## Backend Selection Guidelines

Based on benchmark results:

1. **Small simulations (<5k particles)**: Use Numba backend
   - Best single-threaded performance
   - Low overhead
   
2. **Medium simulations (5k-50k particles)**: Use Numba or GPU
   - GPU starts to show benefits
   - Consider your specific hardware
   
3. **Large simulations (>50k particles)**: Use GPU backend
   - Significant speedup from parallelization
   - Better memory bandwidth utilization
   
4. **Development/debugging**: Use CPU backend
   - Easiest to debug
   - No compilation overhead

## GPU Performance Notes

The GPU backend shows best performance when:
- Particle count is high (>10k)
- Data stays on GPU for multiple timesteps
- Neighbor lists are used to reduce computation

For optimal GPU performance:
```python
# Keep data on GPU for multiple steps
from sph.physics.gpu_unified import get_gpu_sph

gpu = get_gpu_sph()
gpu.upload_particles(particles, n_active)

for _ in range(100):  # Many timesteps
    gpu.compute_density_gpu(n_active)
    gpu.compute_pressure_gpu(n_active)
    gpu.compute_forces_gpu(n_active)
    gpu.integrate_gpu(n_active, dt)

gpu.download_particles(particles, n_active)
```

## Adding New Tests

When adding new tests:
1. Follow the existing test structure
2. Test all three backends when applicable
3. Include both correctness and performance aspects
4. Document expected behavior and assumptions