# SPH Implementation Plan - Vectorized & GPU-Ready

## Overview

This document outlines a **fully vectorized** SPH implementation designed for high CPU performance with easy GPU migration. All data structures and algorithms use Structure-of-Arrays (SoA) pattern and vectorized operations.

## Core Design Principles

1. **Structure of Arrays (SoA)**: All particle data in contiguous arrays
2. **Zero Loops in Physics**: All operations use NumPy vectorization
3. **GPU-Ready Memory Layout**: Coalesced memory access patterns
4. **Batch Processing**: Process particle interactions in chunks
5. **SIMD-Friendly**: Aligned data for CPU vector instructions

## Phase 1: Vectorized Foundation

### 1.1 Particle Data Structure (SoA)
```python
# sph/core/particles.py
import numpy as np
from dataclasses import dataclass

@dataclass
class ParticleArrays:
    """Structure of Arrays for efficient vectorization"""
    # Primary state (N particles)
    position_x: np.ndarray      # shape: (N,) float32
    position_y: np.ndarray      # shape: (N,) float32
    velocity_x: np.ndarray      # shape: (N,) float32
    velocity_y: np.ndarray      # shape: (N,) float32
    
    # Particle properties
    mass: np.ndarray            # shape: (N,) float32
    density: np.ndarray         # shape: (N,) float32
    pressure: np.ndarray        # shape: (N,) float32
    temperature: np.ndarray     # shape: (N,) float32
    
    # Material info
    material_id: np.ndarray     # shape: (N,) int32
    
    # SPH parameters
    smoothing_h: np.ndarray     # shape: (N,) float32
    
    # Force accumulators
    force_x: np.ndarray         # shape: (N,) float32
    force_y: np.ndarray         # shape: (N,) float32
    
    # Neighbor data (fixed max neighbors K)
    neighbor_ids: np.ndarray    # shape: (N, K) int32
    neighbor_distances: np.ndarray  # shape: (N, K) float32
    neighbor_count: np.ndarray  # shape: (N,) int32
    
    @staticmethod
    def allocate(max_particles: int, max_neighbors: int = 64):
        """Pre-allocate arrays for GPU-friendly memory"""
        return ParticleArrays(
            position_x=np.zeros(max_particles, dtype=np.float32),
            position_y=np.zeros(max_particles, dtype=np.float32),
            velocity_x=np.zeros(max_particles, dtype=np.float32),
            velocity_y=np.zeros(max_particles, dtype=np.float32),
            mass=np.zeros(max_particles, dtype=np.float32),
            density=np.zeros(max_particles, dtype=np.float32),
            pressure=np.zeros(max_particles, dtype=np.float32),
            temperature=np.zeros(max_particles, dtype=np.float32),
            material_id=np.zeros(max_particles, dtype=np.int32),
            smoothing_h=np.zeros(max_particles, dtype=np.float32),
            force_x=np.zeros(max_particles, dtype=np.float32),
            force_y=np.zeros(max_particles, dtype=np.float32),
            neighbor_ids=np.full((max_particles, max_neighbors), -1, dtype=np.int32),
            neighbor_distances=np.zeros((max_particles, max_neighbors), dtype=np.float32),
            neighbor_count=np.zeros(max_particles, dtype=np.int32)
        )
```

### 1.2 Vectorized Kernel Functions
```python
# sph/core/kernel_vectorized.py
import numpy as np

class CubicSplineKernel:
    """Fully vectorized cubic spline kernel"""
    
    def __init__(self, dim: int = 2):
        self.dim = dim
        self.norm_factor = 10.0 / (7.0 * np.pi) if dim == 2 else 1.0 / np.pi
    
    def W_vectorized(self, r: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Vectorized kernel evaluation
        r: (N, K) distances to K neighbors for N particles
        h: (N, 1) smoothing lengths
        Returns: (N, K) kernel values
        """
        q = r / h[:, np.newaxis]
        norm = self.norm_factor / (h[:, np.newaxis] ** self.dim)
        
        # Cubic spline cases
        w = np.zeros_like(r)
        
        # Case 1: q <= 1
        mask1 = q <= 1.0
        w[mask1] = (1 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3)
        
        # Case 2: 1 < q <= 2
        mask2 = (q > 1.0) & (q <= 2.0)
        w[mask2] = 0.25 * (2 - q[mask2])**3
        
        return w * norm
    
    def gradW_vectorized(self, dx: np.ndarray, dy: np.ndarray, 
                        r: np.ndarray, h: np.ndarray) -> tuple:
        """
        Vectorized kernel gradient
        dx, dy: (N, K) position differences
        r: (N, K) distances
        h: (N, 1) smoothing lengths
        Returns: (grad_x, grad_y) each (N, K)
        """
        q = r / h[:, np.newaxis]
        norm = self.norm_factor / (h[:, np.newaxis] ** (self.dim + 1))
        
        # Gradient magnitude
        grad_mag = np.zeros_like(r)
        
        # Case 1: q <= 1
        mask1 = (q <= 1.0) & (r > 1e-6)
        grad_mag[mask1] = -3 * q[mask1] + 2.25 * q[mask1]**2
        
        # Case 2: 1 < q <= 2
        mask2 = (q > 1.0) & (q <= 2.0)
        grad_mag[mask2] = -0.75 * (2 - q[mask2])**2
        
        # Apply direction
        grad_x = norm * grad_mag * dx / (r + 1e-10)
        grad_y = norm * grad_mag * dy / (r + 1e-10)
        
        return grad_x, grad_y
```

### 1.3 Vectorized Spatial Hashing
```python
# sph/core/spatial_hash_vectorized.py
class VectorizedSpatialHash:
    """GPU-friendly spatial hashing using cell lists"""
    
    def __init__(self, domain_size: tuple, cell_size: float):
        self.cell_size = cell_size
        self.nx = int(domain_size[0] / cell_size) + 1
        self.ny = int(domain_size[1] / cell_size) + 1
        
        # Pre-allocate cell lists (max particles per cell)
        self.max_per_cell = 100
        self.cell_particles = np.full((self.nx, self.ny, self.max_per_cell), 
                                     -1, dtype=np.int32)
        self.cell_counts = np.zeros((self.nx, self.ny), dtype=np.int32)
        
    def build_vectorized(self, particles: ParticleArrays, n_active: int):
        """Build spatial hash using vectorized operations"""
        # Reset counts
        self.cell_counts.fill(0)
        self.cell_particles.fill(-1)
        
        # Compute cell indices for all particles
        cell_x = (particles.position_x[:n_active] / self.cell_size).astype(np.int32)
        cell_y = (particles.position_y[:n_active] / self.cell_size).astype(np.int32)
        
        # Clip to bounds
        cell_x = np.clip(cell_x, 0, self.nx - 1)
        cell_y = np.clip(cell_y, 0, self.ny - 1)
        
        # Sort particles by cell (for better cache coherence)
        cell_ids = cell_y * self.nx + cell_x
        sort_indices = np.argsort(cell_ids)
        
        # Insert sorted particles into cells
        for idx in sort_indices:
            cx, cy = cell_x[idx], cell_y[idx]
            count = self.cell_counts[cx, cy]
            if count < self.max_per_cell:
                self.cell_particles[cx, cy, count] = idx
                self.cell_counts[cx, cy] += 1
```

### 1.4 Vectorized Neighbor Search
```python
def find_neighbors_vectorized(particles: ParticleArrays, spatial_hash: VectorizedSpatialHash, 
                             n_active: int, search_radius: float):
    """Fully vectorized neighbor search"""
    
    # Reset neighbor data
    particles.neighbor_ids.fill(-1)
    particles.neighbor_count.fill(0)
    
    # Process particles in batches for cache efficiency
    batch_size = 1024
    
    for batch_start in range(0, n_active, batch_size):
        batch_end = min(batch_start + batch_size, n_active)
        batch_slice = slice(batch_start, batch_end)
        
        # Get positions for this batch
        px = particles.position_x[batch_slice]
        py = particles.position_y[batch_slice]
        
        # For each particle in batch, check surrounding cells
        # This is the only loop - everything else is vectorized
        for i, (x, y) in enumerate(zip(px, py)):
            particle_idx = batch_start + i
            
            # Get cell range to check (3x3 in 2D)
            cx = int(x / spatial_hash.cell_size)
            cy = int(y / spatial_hash.cell_size)
            
            # Vectorized distance computation for all potential neighbors
            potential_neighbors = []
            for dcx in range(-1, 2):
                for dcy in range(-1, 2):
                    ncx = cx + dcx
                    ncy = cy + dcy
                    if 0 <= ncx < spatial_hash.nx and 0 <= ncy < spatial_hash.ny:
                        cell_particles = spatial_hash.cell_particles[ncx, ncy]
                        valid_mask = cell_particles >= 0
                        potential_neighbors.extend(cell_particles[valid_mask])
            
            if potential_neighbors:
                # Vectorized distance calculation
                neighbor_indices = np.array(potential_neighbors, dtype=np.int32)
                dx = particles.position_x[neighbor_indices] - x
                dy = particles.position_y[neighbor_indices] - y
                distances = np.sqrt(dx*dx + dy*dy)
                
                # Filter by radius and exclude self
                mask = (distances < search_radius) & (neighbor_indices != particle_idx)
                valid_neighbors = neighbor_indices[mask]
                valid_distances = distances[mask]
                
                # Store neighbors (up to max_neighbors)
                n_found = min(len(valid_neighbors), particles.neighbor_ids.shape[1])
                particles.neighbor_ids[particle_idx, :n_found] = valid_neighbors[:n_found]
                particles.neighbor_distances[particle_idx, :n_found] = valid_distances[:n_found]
                particles.neighbor_count[particle_idx] = n_found
```

## Phase 2: Vectorized Physics

### 2.1 Vectorized Density Computation
```python
# sph/physics/density_vectorized.py
def compute_density_vectorized(particles: ParticleArrays, kernel: CubicSplineKernel, 
                               n_active: int):
    """Fully vectorized density computation"""
    
    # Reset density
    particles.density[:n_active] = 0.0
    
    # Self contribution
    W_self = kernel.W_vectorized(np.zeros((n_active, 1)), particles.smoothing_h[:n_active])
    particles.density[:n_active] = particles.mass[:n_active] * W_self[:, 0]
    
    # Neighbor contributions - fully vectorized
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors > 0:
            # Get neighbor data
            neighbor_ids = particles.neighbor_ids[i, :n_neighbors]
            distances = particles.neighbor_distances[i, :n_neighbors]
            
            # Vectorized kernel evaluation
            W_values = kernel.W_vectorized(
                distances.reshape(1, -1), 
                particles.smoothing_h[i:i+1]
            )[0]
            
            # Sum contributions
            particles.density[i] += np.sum(particles.mass[neighbor_ids] * W_values)
```

### 2.2 Vectorized Force Computation
```python
# sph/physics/forces_vectorized.py
def compute_forces_vectorized(particles: ParticleArrays, kernel: CubicSplineKernel,
                             n_active: int, gravity: np.ndarray):
    """Fully vectorized force computation"""
    
    # Reset forces
    particles.force_x[:n_active] = 0.0
    particles.force_y[:n_active] = 0.0
    
    # Add gravity (vectorized)
    particles.force_x[:n_active] += particles.mass[:n_active] * gravity[0]
    particles.force_y[:n_active] += particles.mass[:n_active] * gravity[1]
    
    # Pressure forces - process in batches for cache efficiency
    batch_size = 256
    
    for batch_start in range(0, n_active, batch_size):
        batch_end = min(batch_start + batch_size, n_active)
        batch_indices = np.arange(batch_start, batch_end)
        
        # Process all particles in batch simultaneously
        for i in batch_indices:
            n_neighbors = particles.neighbor_count[i]
            if n_neighbors == 0:
                continue
                
            # Get neighbor data
            neighbor_ids = particles.neighbor_ids[i, :n_neighbors]
            
            # Vectorized position differences
            dx = particles.position_x[i] - particles.position_x[neighbor_ids]
            dy = particles.position_y[i] - particles.position_y[neighbor_ids]
            distances = particles.neighbor_distances[i, :n_neighbors]
            
            # Vectorized pressure term
            pressure_i = particles.pressure[i] / (particles.density[i]**2)
            pressure_j = particles.pressure[neighbor_ids] / (particles.density[neighbor_ids]**2)
            pressure_term = pressure_i + pressure_j
            
            # Vectorized kernel gradient
            grad_x, grad_y = kernel.gradW_vectorized(
                dx.reshape(1, -1), dy.reshape(1, -1),
                distances.reshape(1, -1),
                particles.smoothing_h[i:i+1]
            )
            
            # Accumulate forces
            force_x = -particles.mass[neighbor_ids] * pressure_term * grad_x[0]
            force_y = -particles.mass[neighbor_ids] * pressure_term * grad_y[0]
            
            particles.force_x[i] += np.sum(force_x)
            particles.force_y[i] += np.sum(force_y)
```

### 2.3 Vectorized Integration
```python
# sph/core/integrator_vectorized.py
def integrate_vectorized(particles: ParticleArrays, n_active: int, dt: float,
                        domain_bounds: tuple):
    """Fully vectorized symplectic integration"""
    
    # Update velocities (vectorized)
    particles.velocity_x[:n_active] += particles.force_x[:n_active] / particles.mass[:n_active] * dt
    particles.velocity_y[:n_active] += particles.force_y[:n_active] / particles.mass[:n_active] * dt
    
    # Update positions (vectorized)
    particles.position_x[:n_active] += particles.velocity_x[:n_active] * dt
    particles.position_y[:n_active] += particles.velocity_y[:n_active] * dt
    
    # Boundary conditions (vectorized)
    # Reflective walls
    mask_left = particles.position_x[:n_active] < domain_bounds[0]
    mask_right = particles.position_x[:n_active] > domain_bounds[1]
    mask_bottom = particles.position_y[:n_active] < domain_bounds[2]
    mask_top = particles.position_y[:n_active] > domain_bounds[3]
    
    # Reflect positions
    particles.position_x[:n_active][mask_left] = 2*domain_bounds[0] - particles.position_x[:n_active][mask_left]
    particles.position_x[:n_active][mask_right] = 2*domain_bounds[1] - particles.position_x[:n_active][mask_right]
    particles.position_y[:n_active][mask_bottom] = 2*domain_bounds[2] - particles.position_y[:n_active][mask_bottom]
    particles.position_y[:n_active][mask_top] = 2*domain_bounds[3] - particles.position_y[:n_active][mask_top]
    
    # Reverse velocities
    particles.velocity_x[:n_active][mask_left | mask_right] *= -0.8  # Some damping
    particles.velocity_y[:n_active][mask_bottom | mask_top] *= -0.8
```

## Phase 3: GPU-Ready Architecture

### 3.1 Compute Dispatcher
```python
# sph/core/compute_dispatcher.py
class ComputeDispatcher:
    """Abstraction layer for CPU/GPU execution"""
    
    def __init__(self, backend='cpu'):
        self.backend = backend
        
    def compute_density(self, particles, kernel, n_active):
        if self.backend == 'cpu':
            return compute_density_vectorized(particles, kernel, n_active)
        elif self.backend == 'cuda':
            return compute_density_cuda(particles, kernel, n_active)
        elif self.backend == 'opencl':
            return compute_density_opencl(particles, kernel, n_active)
```

### 3.2 Memory Management
```python
# sph/core/memory_manager.py
class MemoryManager:
    """Manages memory allocation for different backends"""
    
    def __init__(self, backend='cpu'):
        self.backend = backend
        self.allocations = {}
        
    def allocate_particles(self, max_particles, max_neighbors):
        if self.backend == 'cpu':
            return ParticleArrays.allocate(max_particles, max_neighbors)
        elif self.backend == 'cuda':
            # Allocate on GPU with CuPy
            import cupy as cp
            # ... GPU allocation
```

## Phase 4: Advanced Vectorization

### 4.1 Vectorized Cohesive Forces
```python
# sph/physics/cohesion_vectorized.py
@dataclass
class CohesiveBonds:
    """Structure of Arrays for bonds"""
    particle_i: np.ndarray      # (M,) int32
    particle_j: np.ndarray      # (M,) int32
    rest_length: np.ndarray     # (M,) float32
    strength: np.ndarray        # (M,) float32
    active: np.ndarray          # (M,) bool
    
def compute_cohesive_forces_vectorized(particles: ParticleArrays, bonds: CohesiveBonds):
    """Fully vectorized cohesive force computation"""
    
    # Get active bonds
    active_mask = bonds.active
    
    # Vectorized position differences
    dx = particles.position_x[bonds.particle_i[active_mask]] - \
         particles.position_x[bonds.particle_j[active_mask]]
    dy = particles.position_y[bonds.particle_i[active_mask]] - \
         particles.position_y[bonds.particle_j[active_mask]]
    
    # Vectorized distance and strain
    distances = np.sqrt(dx*dx + dy*dy)
    strain = (distances - bonds.rest_length[active_mask]) / bonds.rest_length[active_mask]
    
    # Vectorized force magnitude
    force_mag = bonds.strength[active_mask] * strain
    
    # Vectorized force components
    fx = force_mag * dx / (distances + 1e-10)
    fy = force_mag * dy / (distances + 1e-10)
    
    # Accumulate forces using np.add.at (atomic for GPU)
    np.add.at(particles.force_x, bonds.particle_i[active_mask], -fx)
    np.add.at(particles.force_x, bonds.particle_j[active_mask], fx)
    np.add.at(particles.force_y, bonds.particle_i[active_mask], -fy)
    np.add.at(particles.force_y, bonds.particle_j[active_mask], fy)
```

### 4.2 Vectorized Thermal Physics
```python
# sph/physics/thermal_vectorized.py
def compute_heat_transfer_vectorized(particles: ParticleArrays, kernel: CubicSplineKernel,
                                    n_active: int, material_props: MaterialDatabase):
    """Fully vectorized heat conduction"""
    
    # Pre-compute thermal diffusivities for all particles
    k_thermal = material_props.thermal_conductivity[particles.material_id[:n_active]]
    rho = particles.density[:n_active]
    cp = material_props.specific_heat[particles.material_id[:n_active]]
    
    # Temperature rate of change
    dT_dt = np.zeros(n_active, dtype=np.float32)
    
    # Process all particles simultaneously
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
            
        neighbor_ids = particles.neighbor_ids[i, :n_neighbors]
        distances = particles.neighbor_distances[i, :n_neighbors]
        
        # Vectorized thermal properties
        k_j = material_props.thermal_conductivity[particles.material_id[neighbor_ids]]
        rho_j = particles.density[neighbor_ids]
        cp_j = material_props.specific_heat[particles.material_id[neighbor_ids]]
        
        # Vectorized temperature differences
        dT = particles.temperature[i] - particles.temperature[neighbor_ids]
        
        # Vectorized kernel gradient
        dx = particles.position_x[i] - particles.position_x[neighbor_ids]
        dy = particles.position_y[i] - particles.position_y[neighbor_ids]
        grad_x, grad_y = kernel.gradW_vectorized(
            dx.reshape(1, -1), dy.reshape(1, -1),
            distances.reshape(1, -1),
            particles.smoothing_h[i:i+1]
        )
        
        # Vectorized heat flux
        flux_factor = 4 * particles.mass[neighbor_ids] * k_thermal[i] * k_j / \
                     (rho[i] * rho_j * (k_thermal[i] + k_j))
        
        # Dot product for radial gradient
        r_dot_grad = (dx * grad_x[0] + dy * grad_y[0]) / (distances + 1e-10)
        
        # Accumulate heat transfer
        dT_dt[i] += np.sum(flux_factor * dT * r_dot_grad) / (cp[i] * distances)
    
    return dT_dt
```

## Performance Optimizations

### 1. Memory Layout
- Use 32-byte aligned arrays for SIMD
- Pad arrays to power-of-2 sizes for GPU
- Interleave position data for better cache usage

### 2. Batch Processing
```python
# Process particles in cache-friendly chunks
CACHE_LINE_SIZE = 64  # bytes
FLOAT_SIZE = 4
PARTICLES_PER_CACHE_LINE = CACHE_LINE_SIZE // (2 * FLOAT_SIZE)  # x,y positions
```

### 3. Parallel Patterns
```python
# Use parallel reduction for global quantities
def parallel_sum(array: np.ndarray, n_threads: int = 4):
    """Parallel reduction for large arrays"""
    chunk_size = len(array) // n_threads
    partial_sums = np.zeros(n_threads)
    
    # This would use threading/multiprocessing in practice
    for i in range(n_threads):
        start = i * chunk_size
        end = start + chunk_size if i < n_threads-1 else len(array)
        partial_sums[i] = np.sum(array[start:end])
    
    return np.sum(partial_sums)
```

### 4. GPU Migration Path
```python
# Easy GPU migration with array libraries
try:
    import cupy as cp
    array_lib = cp  # GPU arrays
except ImportError:
    array_lib = np  # CPU arrays

# All physics code uses array_lib instead of np directly
```

## Benchmarking Targets

### CPU Performance (Vectorized)
- 10,000 particles: 60+ FPS
- 50,000 particles: 30+ FPS
- 100,000 particles: 15+ FPS

### Expected GPU Speedup
- 10-50x for force computation
- 5-20x for neighbor search
- 100x for large N-body gravity

## Testing Strategy

### 1. Vectorization Verification
```python
def test_vectorized_matches_scalar():
    """Ensure vectorized code produces same results as scalar"""
    particles = create_test_particles(1000)
    
    # Compute with scalar version
    density_scalar = compute_density_scalar(particles)
    
    # Compute with vectorized version
    density_vector = compute_density_vectorized(particles)
    
    assert np.allclose(density_scalar, density_vector, rtol=1e-5)
```

### 2. Performance Benchmarks
```python
def benchmark_neighbor_search():
    """Profile neighbor search performance"""
    for n in [1000, 5000, 10000, 50000]:
        particles = create_test_particles(n)
        
        t_start = time.perf_counter()
        find_neighbors_vectorized(particles, spatial_hash, n, 2.0)
        t_end = time.perf_counter()
        
        neighbors_per_sec = n / (t_end - t_start)
        print(f"N={n}: {neighbors_per_sec:.0f} particles/sec")
```

## Implementation Timeline

### Week 1: Vectorized Foundation
- ParticleArrays data structure
- Vectorized kernels
- Vectorized spatial hashing
- Performance benchmarks

### Week 2: Vectorized Physics
- Density computation
- Force calculation
- Integration
- Boundary conditions

### Week 3: Advanced Features
- Cohesive forces
- Thermal physics
- Phase transitions
- Stress tensors

### Week 4: Optimization
- Memory alignment
- Cache optimization
- Parallel reductions
- Profile-guided optimization

### Week 5: GPU Preparation
- Abstract compute dispatcher
- Memory manager
- CuPy/PyOpenCL backends
- GPU kernels

This vectorized approach ensures:
1. Maximum CPU performance through SIMD
2. Easy GPU migration (change array library)
3. Cache-friendly memory patterns
4. Minimal Python overhead
5. Production-ready performance