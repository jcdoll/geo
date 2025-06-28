# SPH Backend System Guide

## Overview

The SPH implementation supports three computation backends that can be switched dynamically:

1. **CPU** - Pure NumPy implementation (always available)
2. **Numba** - JIT-compiled CPU code (10-50x faster)
3. **GPU** - CUDA GPU via CuPy (50-200x faster for large N)

## Quick Start

```python
import sph

# Check available backends
sph.print_backend_info()

# Set backend
sph.set_backend('numba')  # or 'cpu', 'gpu'

# Auto-select based on problem size
sph.auto_select_backend(n_particles=10000)

# Use unified API - backend is handled automatically
particles = sph.ParticleArrays.allocate(10000)
sph.compute_density(particles)
sph.compute_forces(particles)
```

## Backend Comparison

| Backend | Typical Speedup | Best For | Requirements |
|---------|----------------|----------|--------------|
| CPU | 1x (baseline) | Small problems (<1000 particles) | None |
| Numba | 10-50x | Medium problems (1k-50k particles) | `pip install numba` |
| GPU | 50-200x | Large problems (>50k particles) | CUDA GPU + `pip install cupy` |

## Performance Guidelines

### Auto-Selection Logic
- **< 1,000 particles**: CPU (overhead of other backends not worth it)
- **1,000 - 50,000 particles**: Numba (best balance)
- **> 50,000 particles**: GPU (massive parallelism wins)

### Actual Performance Examples

#### 5,000 particles:
- CPU: ~2 FPS
- Numba: ~30 FPS (15x speedup)
- GPU: ~100 FPS (50x speedup)

#### 50,000 particles:
- CPU: ~0.1 FPS
- Numba: ~3 FPS (30x speedup)
- GPU: ~30 FPS (300x speedup)

## Installation

### Basic (CPU only)
```bash
pip install numpy matplotlib
```

### With Numba
```bash
pip install numpy matplotlib numba
```

### With GPU support
```bash
# Requires CUDA-capable GPU
pip install numpy matplotlib cupy-cuda11x  # or appropriate CUDA version
```

## Usage Examples

### Example 1: Manual Backend Selection
```python
import sph
from sph.scenarios import create_planet_simple

# Create simulation
particles, n_active = create_planet_simple(radius=1000, spacing=50)

# Try different backends
for backend in ['cpu', 'numba', 'gpu']:
    if sph.set_backend(backend):
        print(f"\nUsing {backend} backend:")
        
        # Time operations
        import time
        t0 = time.time()
        
        sph.compute_density(particles, n_active=n_active)
        sph.compute_forces(particles, n_active=n_active)
        
        print(f"Time: {time.time() - t0:.3f}s")
```

### Example 2: Backend-Agnostic Code
```python
import sph

# Auto-select best backend
sph.auto_select_backend(n_particles)

# Create spatial hash (automatically uses best implementation)
spatial_hash = sph.create_spatial_hash(domain_size, cell_size)

# Run simulation - backend handled transparently
for step in range(n_steps):
    spatial_hash.build_vectorized(particles, n_active)
    spatial_hash.query_neighbors_vectorized(particles, n_active, radius)
    
    sph.compute_density(particles, n_active=n_active)
    sph.compute_forces(particles, n_active=n_active)
    sph.compute_gravity(particles, n_active=n_active)
```

### Example 3: Force Specific Backend
```python
# Override global backend for specific calls
sph.compute_density(particles, backend='gpu')  # Force GPU
sph.compute_forces(particles, backend='numba')  # Force Numba
```

## Implementation Details

### Dispatch System

The backend system uses a registration and dispatch pattern:

```python
from sph.core.backend import backend_function, for_backend, Backend

@backend_function("compute_density")
@for_backend(Backend.NUMBA)
def compute_density_numba(particles, n_active):
    # Numba implementation
    pass
```

### Memory Management

- **CPU/Numba**: Share same memory (NumPy arrays)
- **GPU**: Requires data transfer to/from GPU memory
  - Overhead is worth it for large problems
  - Consider keeping data on GPU for multiple operations

### Backend-Specific Optimizations

#### CPU
- Vectorized NumPy operations
- Cache-friendly access patterns
- Batch processing

#### Numba
- JIT compilation on first call (warmup needed)
- Parallel loops with `prange`
- Fast math mode enabled
- Type specialization

#### GPU
- CUDA kernels for maximum performance
- Coalesced memory access
- Warp-level optimizations
- Async operations possible

## Troubleshooting

### Numba Issues
```bash
# If Numba fails to import
pip install --upgrade numba

# For Apple M1/M2
conda install numba
```

### GPU Issues
```bash
# Check CUDA availability
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

# Install correct CuPy version
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

### Performance Issues

1. **First run is slow**: JIT compilation overhead (Numba) or kernel compilation (GPU)
2. **GPU slower than CPU**: Problem too small, data transfer overhead dominates
3. **Out of memory**: Reduce particle count or batch size

## Benchmarking

Run the comparison demo:
```bash
# Compare all backends
python -m sph.demo_backend_comparison

# Full comparison with multiple sizes
python -m sph.demo_backend_comparison --full

# Test auto-selection
python -m sph.demo_backend_comparison --auto
```

## Future Extensions

The backend system is designed to be extensible:

- **OpenCL** backend (via PyOpenCL)
- **Metal** backend for macOS
- **WebGPU** for browser deployment
- **TPU** support via JAX

The dispatch system makes adding new backends straightforward without changing user code.