"""
GPU-accelerated force computation using CuPy/CUDA.

Provides massive speedup for the most expensive computation.
"""

try:
    # Set architecture for RTX 5080 compatibility
    import os
    os.environ.setdefault('CUDA_ARCH_LIST', '8.9')
    os.environ.setdefault('CUPY_CUDA_COMPILE_WITH_DEBUG', '0')
    
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

import numpy as np
from ..core.backend import backend_function, for_backend, Backend
from ..core.particles import ParticleArrays


if GPU_AVAILABLE:
    # CUDA kernel for force computation
    forces_kernel_code = '''
    extern "C" __global__
    void compute_forces_cuda(
        const float* __restrict__ pos_x,
        const float* __restrict__ pos_y,
        const float* __restrict__ vel_x,
        const float* __restrict__ vel_y,
        const float* __restrict__ mass,
        const float* __restrict__ density,
        const float* __restrict__ pressure,
        const float* __restrict__ h,
        const int* __restrict__ neighbor_ids,
        const float* __restrict__ neighbor_distances,
        const int* __restrict__ neighbor_count,
        float* __restrict__ force_x,
        float* __restrict__ force_y,
        const int n_active,
        const int max_neighbors,
        const float gravity_x,
        const float gravity_y,
        const float alpha_visc
    ) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_active) return;
        
        // Initialize with gravity
        force_x[i] = mass[i] * gravity_x;
        force_y[i] = mass[i] * gravity_y;
        
        const int n_neighbors = neighbor_count[i];
        if (n_neighbors == 0) return;
        
        // Particle i properties
        const float px_i = pos_x[i];
        const float py_i = pos_y[i];
        const float vx_i = vel_x[i];
        const float vy_i = vel_y[i];
        const float h_i = h[i];
        const float pressure_i = pressure[i] / (density[i] * density[i]);
        
        const float h_i3 = h_i * h_i * h_i;
        const float norm_2d = 10.0f / (7.0f * 3.14159265f * h_i3);
        
        // Process neighbors
        for (int j_idx = 0; j_idx < n_neighbors; j_idx++) {
            const int j = neighbor_ids[i * max_neighbors + j_idx];
            if (j < 0) continue;
            
            const float r = neighbor_distances[i * max_neighbors + j_idx];
            if (r < 1e-6f) continue;
            
            // Position and velocity differences
            const float dx = px_i - pos_x[j];
            const float dy = py_i - pos_y[j];
            const float dvx = vx_i - vel_x[j];
            const float dvy = vy_i - vel_y[j];
            
            // Kernel gradient
            const float q = r / h_i;
            float grad_mag = 0.0f;
            
            if (q <= 1.0f) {
                grad_mag = norm_2d * (-3.0f * q + 2.25f * q * q);
            } else if (q <= 2.0f) {
                grad_mag = -norm_2d * 0.75f * (2.0f - q) * (2.0f - q);
            } else {
                continue;
            }
            
            const float grad_x = grad_mag * dx / r;
            const float grad_y = grad_mag * dy / r;
            
            // Pressure term
            const float pressure_j = pressure[j] / (density[j] * density[j]);
            float pressure_term = pressure_i + pressure_j;
            
            // Artificial viscosity
            float visc_term = 0.0f;
            if (alpha_visc > 0.0f) {
                const float v_dot_r = dvx * dx + dvy * dy;
                if (v_dot_r < 0.0f) {
                    const float c_i = 10.0f * sqrtf(fabsf(pressure[i]) / density[i] + 1e-6f);
                    const float c_j = 10.0f * sqrtf(fabsf(pressure[j]) / density[j] + 1e-6f);
                    const float c_ij = 0.5f * (c_i + c_j);
                    const float h_ij = 0.5f * (h_i + h[j]);
                    const float mu_ij = h_ij * v_dot_r / (r * r + 0.01f * h_ij * h_ij);
                    const float rho_ij = 0.5f * (density[i] + density[j]);
                    visc_term = -alpha_visc * c_ij * mu_ij / rho_ij;
                }
            }
            
            // Total force
            const float force_factor = -mass[j] * (pressure_term + visc_term);
            force_x[i] += force_factor * grad_x;
            force_y[i] += force_factor * grad_y;
        }
    }
    '''
    
    # Compile kernel
    forces_kernel = cp.RawKernel(forces_kernel_code, 'compute_forces_cuda')


@backend_function("compute_forces")
@for_backend(Backend.GPU)
def compute_forces_gpu(particles: ParticleArrays, n_active: int,
                      gravity: np.ndarray = None, alpha_visc: float = 0.1):
    """GPU-accelerated force computation."""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU backend not available. Install CuPy.")
    
    if gravity is None:
        gravity = np.array([0.0, -9.81], dtype=np.float32)
    
    # Transfer to GPU
    pos_x_gpu = cp.asarray(particles.position_x[:n_active])
    pos_y_gpu = cp.asarray(particles.position_y[:n_active])
    vel_x_gpu = cp.asarray(particles.velocity_x[:n_active])
    vel_y_gpu = cp.asarray(particles.velocity_y[:n_active])
    mass_gpu = cp.asarray(particles.mass[:n_active])
    density_gpu = cp.asarray(particles.density[:n_active])
    pressure_gpu = cp.asarray(particles.pressure[:n_active])
    h_gpu = cp.asarray(particles.smoothing_h[:n_active])
    neighbor_ids_gpu = cp.asarray(particles.neighbor_ids[:n_active])
    neighbor_distances_gpu = cp.asarray(particles.neighbor_distances[:n_active])
    neighbor_count_gpu = cp.asarray(particles.neighbor_count[:n_active])
    
    # Allocate output
    force_x_gpu = cp.zeros(n_active, dtype=cp.float32)
    force_y_gpu = cp.zeros(n_active, dtype=cp.float32)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (n_active + threads_per_block - 1) // threads_per_block
    max_neighbors = particles.neighbor_ids.shape[1]
    
    forces_kernel(
        (blocks,), (threads_per_block,),
        (pos_x_gpu, pos_y_gpu, vel_x_gpu, vel_y_gpu,
         mass_gpu, density_gpu, pressure_gpu, h_gpu,
         neighbor_ids_gpu, neighbor_distances_gpu, neighbor_count_gpu,
         force_x_gpu, force_y_gpu,
         n_active, max_neighbors,
         float(gravity[0]), float(gravity[1]), float(alpha_visc))
    )
    
    # Transfer back
    particles.force_x[:n_active] = cp.asnumpy(force_x_gpu)
    particles.force_y[:n_active] = cp.asnumpy(force_y_gpu)