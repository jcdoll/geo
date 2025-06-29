"""
Vectorized density computation for SPH.

Implements both:
- Direct summation: ρᵢ = Σⱼ mⱼ W(rᵢⱼ, h)
- Continuity equation: dρ/dt = -ρ∇·v
"""

import numpy as np
from ..core.particles import ParticleArrays
from ..core.kernel_vectorized import CubicSplineKernel


def compute_density_vectorized(particles: ParticleArrays, kernel: CubicSplineKernel,
                              n_active: int):
    """Fully vectorized density computation using direct summation.
    
    This is the standard SPH density calculation:
    ρᵢ = Σⱼ mⱼ W(|rᵢ - rⱼ|, h)
    
    Args:
        particles: Particle arrays with neighbor information
        kernel: SPH kernel function
        n_active: Number of active particles
    """
    # Reset density
    particles.density[:n_active] = 0.0
    
    # Self contribution: W(0, h) * mass
    W_self = kernel.W_self(particles.smoothing_h[:n_active])
    particles.density[:n_active] = particles.mass[:n_active] * W_self
    
    # Process all particles - vectorized per particle
    # This loop processes one particle at a time but uses vectorized
    # operations for all its neighbors
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        # Get neighbor data
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        distances = particles.neighbor_distances[i, neighbor_slice]
        
        # Vectorized kernel evaluation for all neighbors at once
        W_values = kernel.W_vectorized(distances, particles.smoothing_h[i])
        
        # Vectorized mass contribution
        mass_contributions = particles.mass[neighbor_ids] * W_values
        
        # Sum all contributions
        particles.density[i] += np.sum(mass_contributions)


def compute_density_vectorized_batch(particles: ParticleArrays, kernel: CubicSplineKernel,
                                    n_active: int, batch_size: int = 256):
    """Batch-vectorized density computation for better cache usage.
    
    Processes multiple particles simultaneously for improved performance.
    
    Args:
        particles: Particle arrays with neighbor information
        kernel: SPH kernel function
        n_active: Number of active particles
        batch_size: Number of particles to process together
    """
    # Reset density with self contribution
    W_self = kernel.W_self(particles.smoothing_h[:n_active])
    particles.density[:n_active] = particles.mass[:n_active] * W_self
    
    # Process in batches
    for batch_start in range(0, n_active, batch_size):
        batch_end = min(batch_start + batch_size, n_active)
        batch_indices = np.arange(batch_start, batch_end)
        
        # Maximum neighbors in this batch
        max_neighbors_batch = np.max(particles.neighbor_count[batch_indices])
        if max_neighbors_batch == 0:
            continue
        
        # Process each neighbor index
        for k in range(max_neighbors_batch):
            # Mask for particles that have a k-th neighbor
            has_kth_neighbor = particles.neighbor_count[batch_indices] > k
            if not np.any(has_kth_neighbor):
                continue
            
            # Indices of particles with k-th neighbor
            active_in_batch = batch_indices[has_kth_neighbor]
            
            # Get k-th neighbor for active particles
            neighbor_ids = particles.neighbor_ids[active_in_batch, k]
            distances = particles.neighbor_distances[active_in_batch, k]
            
            # Vectorized kernel evaluation
            h_values = particles.smoothing_h[active_in_batch]
            W_values = kernel.W_vectorized(distances, h_values)
            
            # Add mass contributions
            particles.density[active_in_batch] += particles.mass[neighbor_ids] * W_values


def compute_density_continuity_vectorized(particles: ParticleArrays, kernel: CubicSplineKernel,
                                         n_active: int, dt: float):
    """Compute density using continuity equation: dρ/dt = -ρ∇·v.
    
    This is an alternative to direct summation, useful for
    incompressible or weakly compressible flows.
    
    Args:
        particles: Particle arrays
        kernel: SPH kernel function
        n_active: Number of active particles
        dt: Time step
    """
    # Compute velocity divergence for each particle
    div_v = np.zeros(n_active, dtype=np.float32)
    
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        # Get neighbor data
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        
        # Position differences
        dx = particles.position_x[i] - particles.position_x[neighbor_ids]
        dy = particles.position_y[i] - particles.position_y[neighbor_ids]
        
        # Velocity differences
        dvx = particles.velocity_x[i] - particles.velocity_x[neighbor_ids]
        dvy = particles.velocity_y[i] - particles.velocity_y[neighbor_ids]
        
        # Get kernel gradients
        distances = particles.neighbor_distances[i, neighbor_slice]
        grad_x, grad_y = kernel.gradW_vectorized(
            dx.reshape(1, -1), dy.reshape(1, -1),
            distances.reshape(1, -1),
            particles.smoothing_h[i]
        )
        
        # Divergence: ∇·v = Σⱼ mⱼ/ρⱼ (vᵢ - vⱼ)·∇W
        mass_over_rho = particles.mass[neighbor_ids] / particles.density[neighbor_ids]
        div_contribution = mass_over_rho * (dvx * grad_x[0] + dvy * grad_y[0])
        
        div_v[i] = np.sum(div_contribution)
    
    # Update density: ρ(t+dt) = ρ(t) - dt * ρ * ∇·v
    particles.density[:n_active] -= dt * particles.density[:n_active] * div_v


def validate_density_computation():
    """Test density computation with known configuration."""
    from ..core.particles import ParticleArrays
    
    # Create simple test case: 4 particles in a square
    particles = ParticleArrays.allocate(4, max_neighbors=4)
    
    # Position particles in unit square
    particles.position_x[:4] = [0.0, 1.0, 0.0, 1.0]
    particles.position_y[:4] = [0.0, 0.0, 1.0, 1.0]
    particles.mass[:4] = 1.0
    particles.smoothing_h[:4] = 1.5  # Large enough to see all neighbors
    
    # Set up neighbors manually
    for i in range(4):
        neighbors = []
        distances = []
        for j in range(4):
            if i != j:
                dx = particles.position_x[i] - particles.position_x[j]
                dy = particles.position_y[i] - particles.position_y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                neighbors.append(j)
                distances.append(dist)
        
        particles.neighbor_ids[i, :3] = neighbors
        particles.neighbor_distances[i, :3] = distances
        particles.neighbor_count[i] = 3
    
    # Compute density
    kernel = CubicSplineKernel(dim=2)
    compute_density_vectorized(particles, kernel, 4)
    
    print("Test density computation:")
    print(f"Particle densities: {particles.density[:4]}")
    print(f"All equal: {np.allclose(particles.density[:4], particles.density[0])}")
    
    return particles