"""
GPU-accelerated spatial hash for neighbor finding.

Uses PyTorch for fast neighbor lookups on GPU.
"""

import torch
import numpy as np
from typing import Tuple
from ..core.backend import backend_function, for_backend, Backend
from ..core.particles import ParticleArrays


class SpatialHashGPU:
    """GPU-based spatial hash for fast neighbor finding."""
    
    def __init__(self, domain_size: Tuple[float, float], cell_size: float,
                 domain_min: Tuple[float, float] = None, max_particles: int = 100000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.domain_size = domain_size
        self.cell_size = cell_size
        self.max_particles = max_particles
        
        if domain_min is None:
            self.domain_min = (-domain_size[0] / 2, -domain_size[1] / 2)
        else:
            self.domain_min = domain_min
        
        # Compute grid dimensions
        self.n_cells_x = int(np.ceil(domain_size[0] / cell_size))
        self.n_cells_y = int(np.ceil(domain_size[1] / cell_size))
        self.n_cells = self.n_cells_x * self.n_cells_y
        
        # Pre-allocate GPU memory for positions
        self.pos_x_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.pos_y_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        
        # Pre-allocate cell lists
        self.max_per_cell = 50  # Reasonable max particles per cell
        self.cell_particles = torch.full((self.n_cells, self.max_per_cell), -1, 
                                       device=self.device, dtype=torch.int32)
        self.cell_count = torch.zeros(self.n_cells, device=self.device, dtype=torch.int32)
        
        # Pre-allocate neighbor arrays
        self.max_neighbors = 150
        self.neighbor_ids = torch.zeros((max_particles, self.max_neighbors),
                                       device=self.device, dtype=torch.int32)
        self.neighbor_distances = torch.zeros((max_particles, self.max_neighbors),
                                            device=self.device, dtype=torch.float32)
        self.neighbor_count = torch.zeros(max_particles, device=self.device, dtype=torch.int32)
        
    def _hash_position(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Convert position to cell index."""
        cell_x = ((x - self.domain_min[0]) / self.cell_size).long()
        cell_y = ((y - self.domain_min[1]) / self.cell_size).long()
        
        # Clamp to valid range
        cell_x = torch.clamp(cell_x, 0, self.n_cells_x - 1)
        cell_y = torch.clamp(cell_y, 0, self.n_cells_y - 1)
        
        return cell_y * self.n_cells_x + cell_x
    
    def build_hash(self, particles: ParticleArrays, n_active: int):
        """Build spatial hash on GPU."""
        # Transfer positions to GPU
        self.pos_x_gpu[:n_active] = torch.from_numpy(particles.position_x[:n_active])
        self.pos_y_gpu[:n_active] = torch.from_numpy(particles.position_y[:n_active])
        
        # Clear cell lists
        self.cell_particles.fill_(-1)
        self.cell_count.zero_()
        
        # Compute cell indices for all particles
        pos_x = self.pos_x_gpu[:n_active]
        pos_y = self.pos_y_gpu[:n_active]
        cell_indices = self._hash_position(pos_x, pos_y)
        
        # Sort particles by cell index for better cache performance
        sorted_indices = torch.argsort(cell_indices)
        sorted_cells = cell_indices[sorted_indices]
        
        # Build cell lists (this part is still sequential but fast on GPU)
        for i, (particle_idx, cell_idx) in enumerate(zip(sorted_indices, sorted_cells)):
            count = self.cell_count[cell_idx]
            if count < self.max_per_cell:
                self.cell_particles[cell_idx, count] = particle_idx
                self.cell_count[cell_idx] = count + 1
    
    def find_neighbors_gpu(self, particles: ParticleArrays, n_active: int, search_radius: float):
        """Find all neighbors within search radius using GPU."""
        # Build hash
        self.build_hash(particles, n_active)
        
        # Clear neighbor lists
        self.neighbor_ids[:n_active].fill_(-1)
        self.neighbor_distances[:n_active].fill_(0)
        self.neighbor_count[:n_active].zero_()
        
        pos_x = self.pos_x_gpu[:n_active]
        pos_y = self.pos_y_gpu[:n_active]
        
        # Determine search range in cells
        n_search_cells = int(np.ceil(search_radius / self.cell_size))
        
        # Process each particle
        for i in range(n_active):
            x_i = pos_x[i]
            y_i = pos_y[i]
            
            # Get cell of particle i
            cell_x = int((x_i - self.domain_min[0]) / self.cell_size)
            cell_y = int((y_i - self.domain_min[1]) / self.cell_size)
            
            neighbor_idx = 0
            
            # Search neighboring cells
            for dy in range(-n_search_cells, n_search_cells + 1):
                for dx in range(-n_search_cells, n_search_cells + 1):
                    # Get neighbor cell coordinates
                    nx = cell_x + dx
                    ny = cell_y + dy
                    
                    # Skip if out of bounds
                    if nx < 0 or nx >= self.n_cells_x or ny < 0 or ny >= self.n_cells_y:
                        continue
                    
                    # Get cell index
                    cell_idx = ny * self.n_cells_x + nx
                    n_in_cell = self.cell_count[cell_idx]
                    
                    # Check all particles in this cell
                    for k in range(n_in_cell):
                        j = self.cell_particles[cell_idx, k].item()
                        if j < 0 or j == i:
                            continue
                        
                        # Compute distance
                        dx_p = x_i - pos_x[j]
                        dy_p = y_i - pos_y[j]
                        r = torch.sqrt(dx_p**2 + dy_p**2)
                        
                        # Add to neighbors if within radius
                        if r < search_radius and neighbor_idx < self.max_neighbors:
                            self.neighbor_ids[i, neighbor_idx] = j
                            self.neighbor_distances[i, neighbor_idx] = r
                            neighbor_idx += 1
            
            self.neighbor_count[i] = neighbor_idx
        
        # Copy back to CPU
        particles.neighbor_ids[:n_active] = self.neighbor_ids[:n_active].cpu().numpy()
        particles.neighbor_distances[:n_active] = self.neighbor_distances[:n_active].cpu().numpy()
        particles.neighbor_count[:n_active] = self.neighbor_count[:n_active].cpu().numpy()


# Global spatial hash instance
_spatial_hash_gpu = None


@backend_function("find_neighbors")
@for_backend(Backend.GPU)
def find_neighbors_torch(particles: ParticleArrays, spatial_hash, n_active: int, search_radius: float):
    """GPU-accelerated neighbor finding."""
    global _spatial_hash_gpu
    
    # Create GPU spatial hash on first use
    if _spatial_hash_gpu is None:
        # Extract parameters from CPU spatial hash
        domain_size = spatial_hash.domain_size
        cell_size = spatial_hash.cell_size
        domain_min = spatial_hash.domain_min
        
        _spatial_hash_gpu = SpatialHashGPU(domain_size, cell_size, domain_min)
    
    # Find neighbors on GPU
    _spatial_hash_gpu.find_neighbors_gpu(particles, n_active, search_radius)