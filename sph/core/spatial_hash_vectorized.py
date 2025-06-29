"""
Vectorized spatial hashing for O(N) neighbor searches.

This implementation uses:
- Cell lists for spatial subdivision
- Vectorized operations for building and querying
- Cache-friendly memory access patterns
- GPU-ready data structures
"""

import numpy as np
from typing import Tuple, Optional
from .particles import ParticleArrays


class VectorizedSpatialHash:
    """GPU-friendly spatial hashing using cell lists.
    
    Uses a regular grid of cells where each cell contains a list of
    particle indices. Optimized for:
    - Vectorized cell assignment
    - Sorted particle access for cache coherence
    - Fixed-size cell lists for GPU compatibility
    """
    
    def __init__(self, domain_size: Tuple[float, float], cell_size: float,
                 max_per_cell: int = 100, domain_min: Tuple[float, float] = None):
        """Initialize spatial hash grid.
        
        Args:
            domain_size: (width, height) of simulation domain
            cell_size: Size of each cell (should be ~2*smoothing_length)
            max_per_cell: Maximum particles per cell
            domain_min: (min_x, min_y) of domain. If None, assumes (0, 0)
        """
        self.cell_size = cell_size
        self.domain_size = domain_size
        self.max_per_cell = max_per_cell
        
        # Domain bounds
        if domain_min is None:
            self.domain_min = (0.0, 0.0)
        else:
            self.domain_min = domain_min
        
        # Grid dimensions
        self.nx = int(domain_size[0] / cell_size) + 1
        self.ny = int(domain_size[1] / cell_size) + 1
        self.n_cells = self.nx * self.ny
        
        # Pre-allocate cell lists
        # cell_particles[i, j, k] = particle index in cell (i,j), slot k
        self.cell_particles = np.full((self.nx, self.ny, max_per_cell), 
                                     -1, dtype=np.int32)
        self.cell_counts = np.zeros((self.nx, self.ny), dtype=np.int32)
        
        # For sorted access
        self.sorted_indices = None
        self.cell_start = np.zeros(self.n_cells + 1, dtype=np.int32)
        
        print(f"Spatial hash: {self.nx}x{self.ny} cells, cell size {cell_size}")
    
    def build_vectorized(self, particles: ParticleArrays, n_active: int):
        """Build spatial hash using vectorized operations.
        
        Args:
            particles: Particle arrays
            n_active: Number of active particles
        """
        # Reset
        self.cell_counts.fill(0)
        self.cell_particles.fill(-1)
        
        # Compute cell indices for all particles (vectorized)
        # Offset by domain minimum to handle negative coordinates
        cell_x = np.floor((particles.position_x[:n_active] - self.domain_min[0]) / self.cell_size).astype(np.int32)
        cell_y = np.floor((particles.position_y[:n_active] - self.domain_min[1]) / self.cell_size).astype(np.int32)
        
        # Clip to bounds
        cell_x = np.clip(cell_x, 0, self.nx - 1)
        cell_y = np.clip(cell_y, 0, self.ny - 1)
        
        # Convert to linear cell index
        cell_ids = cell_y * self.nx + cell_x
        
        # Sort particles by cell for better cache coherence
        self.sorted_indices = np.argsort(cell_ids)
        sorted_cells = cell_ids[self.sorted_indices]
        
        # Build cell start indices
        self.cell_start.fill(n_active)  # Default to end
        self.cell_start[0] = 0
        
        # Find where cell changes occur
        cell_changes = np.where(sorted_cells[:-1] != sorted_cells[1:])[0] + 1
        self.cell_start[sorted_cells[cell_changes]] = cell_changes
        
        # Fill cell lists (this is the only loop - could be parallelized)
        for idx in range(n_active):
            particle_id = self.sorted_indices[idx]
            cx = cell_x[particle_id]
            cy = cell_y[particle_id]
            
            count = self.cell_counts[cx, cy]
            if count < self.max_per_cell:
                self.cell_particles[cx, cy, count] = particle_id
                self.cell_counts[cx, cy] += 1
    
    def query_neighbors_vectorized(self, particles: ParticleArrays, n_active: int, 
                                  search_radius: float):
        """Find all neighbors within radius using vectorized operations.
        
        Args:
            particles: Particle arrays
            n_active: Number of active particles
            search_radius: Search radius (typically 2*smoothing_length)
        """
        # Reset neighbor data
        particles.reset_neighbors(n_active)
        
        # Number of cells to search in each direction
        n_search = int(np.ceil(search_radius / self.cell_size))
        
        # Process particles in batches for cache efficiency
        batch_size = min(1024, n_active)
        
        for batch_start in range(0, n_active, batch_size):
            batch_end = min(batch_start + batch_size, n_active)
            batch_slice = slice(batch_start, batch_end)
            batch_size_actual = batch_end - batch_start
            
            # Get positions for this batch
            px = particles.position_x[batch_slice]
            py = particles.position_y[batch_slice]
            
            # Compute cell indices for batch
            # Account for domain minimum to handle negative coordinates
            batch_cx = np.floor((px - self.domain_min[0]) / self.cell_size).astype(np.int32)
            batch_cy = np.floor((py - self.domain_min[1]) / self.cell_size).astype(np.int32)
            
            # For each particle in batch
            for i in range(batch_size_actual):
                particle_idx = batch_start + i
                cx, cy = batch_cx[i], batch_cy[i]
                x, y = px[i], py[i]
                
                # Collect potential neighbors from surrounding cells
                potential_neighbors = []
                
                for dcx in range(-n_search, n_search + 1):
                    for dcy in range(-n_search, n_search + 1):
                        ncx = cx + dcx
                        ncy = cy + dcy
                        
                        # Check bounds
                        if 0 <= ncx < self.nx and 0 <= ncy < self.ny:
                            # Get particles in this cell
                            cell_count = self.cell_counts[ncx, ncy]
                            if cell_count > 0:
                                cell_particles = self.cell_particles[ncx, ncy, :cell_count]
                                potential_neighbors.extend(cell_particles)
                
                if potential_neighbors:
                    # Vectorized distance calculation
                    neighbor_indices = np.array(potential_neighbors, dtype=np.int32)
                    
                    # Compute distances
                    dx = particles.position_x[neighbor_indices] - x
                    dy = particles.position_y[neighbor_indices] - y
                    distances = np.sqrt(dx*dx + dy*dy)
                    
                    # Filter by radius and exclude self
                    mask = (distances < search_radius) & (neighbor_indices != particle_idx)
                    valid_neighbors = neighbor_indices[mask]
                    valid_distances = distances[mask]
                    
                    # Sort by distance for better convergence
                    if len(valid_neighbors) > 0:
                        sort_idx = np.argsort(valid_distances)
                        valid_neighbors = valid_neighbors[sort_idx]
                        valid_distances = valid_distances[sort_idx]
                        
                        # Store neighbors (up to max_neighbors)
                        n_found = min(len(valid_neighbors), particles.neighbor_ids.shape[1])
                        particles.neighbor_ids[particle_idx, :n_found] = valid_neighbors[:n_found]
                        particles.neighbor_distances[particle_idx, :n_found] = valid_distances[:n_found]
                        particles.neighbor_count[particle_idx] = n_found
    
    def get_cell_particles(self, cell_x: int, cell_y: int) -> np.ndarray:
        """Get particle indices in a specific cell."""
        if 0 <= cell_x < self.nx and 0 <= cell_y < self.ny:
            count = self.cell_counts[cell_x, cell_y]
            if count > 0:
                return self.cell_particles[cell_x, cell_y, :count]
        return np.array([], dtype=np.int32)
    
    def get_statistics(self) -> dict:
        """Get hash table statistics for debugging."""
        counts = self.cell_counts.flatten()
        occupied = counts > 0
        
        return {
            'total_cells': self.n_cells,
            'occupied_cells': np.sum(occupied),
            'occupancy_rate': np.sum(occupied) / self.n_cells,
            'max_particles_per_cell': np.max(counts),
            'mean_particles_per_occupied_cell': np.mean(counts[occupied]) if np.any(occupied) else 0,
            'particles_in_full_cells': np.sum(counts == self.max_per_cell)
        }


def find_neighbors_vectorized(particles: ParticleArrays, spatial_hash: VectorizedSpatialHash,
                             n_active: int, search_radius: float):
    """Convenience function for neighbor search.
    
    Args:
        particles: Particle arrays
        spatial_hash: Spatial hash structure
        n_active: Number of active particles
        search_radius: Search radius
    """
    spatial_hash.query_neighbors_vectorized(particles, n_active, search_radius)