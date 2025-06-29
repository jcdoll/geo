"""
Optimized PyTorch GPU density computation for SPH.

Uses advanced PyTorch features for maximum performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from ..core.backend import backend_function, for_backend, Backend
from ..core.particles import ParticleArrays


# Pre-compile kernel function with torch.jit.script
@torch.jit.script
def cubic_spline_kernel_jit(r: torch.Tensor, h: float) -> torch.Tensor:
    """JIT-compiled cubic spline kernel."""
    q = r / h
    norm_2d = 10.0 / (7.0 * 3.14159265359 * h * h)
    
    W = torch.zeros_like(q)
    
    # Mask-based computation (no branches in GPU)
    mask1 = q <= 1.0
    mask2 = (q > 1.0) & (q <= 2.0)
    
    # Apply kernel
    W = torch.where(mask1, 
                    norm_2d * (1.0 - 1.5 * q**2 + 0.75 * q**3),
                    W)
    W = torch.where(mask2,
                    norm_2d * 0.25 * (2.0 - q)**3,
                    W)
    
    return W


class DensityComputerGPU:
    """Persistent GPU density computer to avoid repeated memory transfers."""
    
    def __init__(self, max_particles: int = 100000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_particles = max_particles
        
        # Pre-allocate GPU memory
        self.pos_x_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.pos_y_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.mass_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.h_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.density_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        
    def compute_density_all_pairs(self, particles: ParticleArrays, n_active: int):
        """All-pairs density computation optimized for GPU."""
        # Transfer only active particles
        self.pos_x_gpu[:n_active] = torch.from_numpy(particles.position_x[:n_active])
        self.pos_y_gpu[:n_active] = torch.from_numpy(particles.position_y[:n_active])
        self.mass_gpu[:n_active] = torch.from_numpy(particles.mass[:n_active])
        self.h_gpu[:n_active] = torch.from_numpy(particles.smoothing_h[:n_active])
        
        # Work with views of active particles only
        pos_x = self.pos_x_gpu[:n_active]
        pos_y = self.pos_y_gpu[:n_active]
        mass = self.mass_gpu[:n_active]
        h = self.h_gpu[:n_active]
        
        # Batch compute all pairwise distances at once
        # Using broadcasting: (n, 1) - (1, n) = (n, n)
        dx = pos_x.unsqueeze(1) - pos_x.unsqueeze(0)
        dy = pos_y.unsqueeze(1) - pos_y.unsqueeze(0)
        r = torch.sqrt(dx**2 + dy**2 + 1e-10)
        
        # Use average h for simplicity (could be improved)
        h_avg = h.mean().item()
        
        # Compute kernel for all pairs
        W = cubic_spline_kernel_jit(r, h_avg)
        
        # Density = sum of mass * kernel
        # Broadcasting: (1, n) * (n, n) then sum over dim 1
        density = torch.sum(mass.unsqueeze(0) * W, dim=1)
        
        # Copy back to CPU
        particles.density[:n_active] = density.cpu().numpy()
    
    def compute_density_neighbors(self, particles: ParticleArrays, n_active: int):
        """Neighbor-based density computation (more efficient for large systems)."""
        # Transfer data
        self.pos_x_gpu[:n_active] = torch.from_numpy(particles.position_x[:n_active])
        self.pos_y_gpu[:n_active] = torch.from_numpy(particles.position_y[:n_active])
        self.mass_gpu[:n_active] = torch.from_numpy(particles.mass[:n_active])
        self.h_gpu[:n_active] = torch.from_numpy(particles.smoothing_h[:n_active])
        
        neighbor_ids = torch.from_numpy(particles.neighbor_ids[:n_active]).to(self.device)
        neighbor_distances = torch.from_numpy(particles.neighbor_distances[:n_active]).to(self.device)
        neighbor_count = torch.from_numpy(particles.neighbor_count[:n_active]).to(self.device)
        
        # Initialize density with self-contribution
        h = self.h_gpu[:n_active]
        mass = self.mass_gpu[:n_active]
        norm_2d = 10.0 / (7.0 * 3.14159265359)
        self.density_gpu[:n_active] = mass * norm_2d / (h * h)
        
        # Create mask for valid neighbors
        max_neighbors = neighbor_ids.shape[1]
        neighbor_range = torch.arange(max_neighbors, device=self.device).unsqueeze(0)
        valid_mask = neighbor_range < neighbor_count.unsqueeze(1)
        
        # Set invalid neighbor IDs to 0 (will be masked out)
        neighbor_ids_safe = torch.where(valid_mask, neighbor_ids, 0)
        
        # Gather neighbor masses (with safe indexing)
        neighbor_masses = mass[neighbor_ids_safe]
        
        # Compute kernel values for all neighbors at once
        # Use average h for the kernel computation
        h_avg = h.mean().item()
        W = cubic_spline_kernel_jit(neighbor_distances, h_avg)
        
        # Mask out invalid neighbors
        W = torch.where(valid_mask, W, 0.0)
        
        # Sum contributions
        density_contributions = neighbor_masses * W
        self.density_gpu[:n_active] += density_contributions.sum(dim=1)
        
        # Copy back
        particles.density[:n_active] = self.density_gpu[:n_active].cpu().numpy()


# Global instance to maintain GPU state
_density_computer = None


@backend_function("compute_density")
@for_backend(Backend.GPU)
def compute_density_torch_optimized(particles: ParticleArrays, kernel, n_active: int):
    """Optimized GPU density computation."""
    global _density_computer
    
    # Create persistent computer on first use
    if _density_computer is None:
        _density_computer = DensityComputerGPU()
    
    # Use neighbor-based method for better scaling
    if particles.neighbor_count[0] > 0:  # Check if neighbors are available
        _density_computer.compute_density_neighbors(particles, n_active)
    else:
        _density_computer.compute_density_all_pairs(particles, n_active)