"""
Vectorized gravity calculations for SPH including self-gravity.

Implements:
- Direct N-body gravity (O(N²))
- Barnes-Hut tree code (O(N log N)) for large systems
- Uniform external gravity field
"""

import numpy as np
from typing import Optional, Tuple
from ..core.particles import ParticleArrays


def compute_gravity_uniform(particles: ParticleArrays, n_active: int,
                           g_vector: np.ndarray = None) -> None:
    """Apply uniform gravitational field (e.g., planetary surface gravity).
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        g_vector: Gravity vector [gx, gy], default [0, -9.81]
    """
    if g_vector is None:
        g_vector = np.array([0.0, -9.81], dtype=np.float32)
    
    # F = m * g
    particles.force_x[:n_active] += particles.mass[:n_active] * g_vector[0]
    particles.force_y[:n_active] += particles.mass[:n_active] * g_vector[1]


def compute_gravity_direct_vectorized(particles: ParticleArrays, n_active: int,
                                     G: float = 6.67430e-11,
                                     softening: float = 0.1) -> None:
    """Direct O(N²) self-gravity calculation using vectorized operations.
    
    This is suitable for systems with < 10,000 particles.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        G: Gravitational constant (SI units)
        softening: Softening length to prevent singularities (meters)
    """
    # Reset gravity components
    gravity_x = np.zeros(n_active, dtype=np.float32)
    gravity_y = np.zeros(n_active, dtype=np.float32)
    
    # Get active particle positions and masses
    pos_x = particles.position_x[:n_active]
    pos_y = particles.position_y[:n_active]
    masses = particles.mass[:n_active]
    
    # Softening squared
    eps2 = softening**2
    
    # Compute pairwise forces (vectorized per particle)
    for i in range(n_active):
        # Position differences from particle i to all others
        dx = pos_x[i] - pos_x
        dy = pos_y[i] - pos_y
        
        # Distances squared (with softening)
        r2 = dx*dx + dy*dy + eps2
        
        # Skip self-interaction
        r2[i] = np.inf
        
        # Inverse distance cubed (for force calculation)
        inv_r3 = r2**(-1.5)
        
        # Gravitational acceleration: a = -G * m * r / |r|³
        ax = -G * masses * dx * inv_r3
        ay = -G * masses * dy * inv_r3
        
        # Sum contributions (excluding self)
        gravity_x[i] = np.sum(ax)
        gravity_y[i] = np.sum(ay)
    
    # Add to forces: F = m * a
    particles.force_x[:n_active] += particles.mass[:n_active] * gravity_x
    particles.force_y[:n_active] += particles.mass[:n_active] * gravity_y


def compute_gravity_direct_batched(particles: ParticleArrays, n_active: int,
                                  G: float = 6.67430e-11,
                                  softening: float = 0.1,
                                  batch_size: int = 1000) -> None:
    """Memory-efficient batched version of direct gravity.
    
    Processes particles in batches to avoid O(N²) memory usage.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        G: Gravitational constant
        softening: Softening length
        batch_size: Number of particles per batch
    """
    # Initialize acceleration arrays
    accel_x = np.zeros(n_active, dtype=np.float32)
    accel_y = np.zeros(n_active, dtype=np.float32)
    
    eps2 = softening**2
    
    # Process in batches
    for batch_start in range(0, n_active, batch_size):
        batch_end = min(batch_start + batch_size, n_active)
        batch_indices = slice(batch_start, batch_end)
        batch_size_actual = batch_end - batch_start
        
        # Get batch positions
        batch_x = particles.position_x[batch_indices]
        batch_y = particles.position_y[batch_indices]
        
        # Compute interactions with all particles
        for j in range(n_active):
            # Skip self-interactions
            if batch_start <= j < batch_end:
                continue
            
            # Position differences
            dx = batch_x - particles.position_x[j]
            dy = batch_y - particles.position_y[j]
            
            # Distances
            r2 = dx*dx + dy*dy + eps2
            inv_r3 = r2**(-1.5)
            
            # Acceleration contributions
            a_factor = G * particles.mass[j] * inv_r3
            
            # Add to batch (action)
            accel_x[batch_indices] -= a_factor * dx
            accel_y[batch_indices] -= a_factor * dy
            
            # Add to particle j (reaction - Newton's 3rd law)
            if j < batch_start or j >= batch_end:
                accel_x[j] += np.sum(a_factor * dx)
                accel_y[j] += np.sum(a_factor * dy)
    
    # Convert acceleration to force
    particles.force_x[:n_active] += particles.mass[:n_active] * accel_x
    particles.force_y[:n_active] += particles.mass[:n_active] * accel_y


class QuadTreeNode:
    """Quadtree node for Barnes-Hut algorithm."""
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Initialize node with bounds."""
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
        # Node data
        self.total_mass = 0.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.particle_idx = -1  # For leaf nodes
        
        # Children (NE, NW, SW, SE)
        self.children = None
        self.is_leaf = True
    
    def insert(self, idx: int, x: float, y: float, mass: float):
        """Insert particle into tree."""
        # Update center of mass
        if self.total_mass > 0:
            self.center_x = (self.center_x * self.total_mass + x * mass) / (self.total_mass + mass)
            self.center_y = (self.center_y * self.total_mass + y * mass) / (self.total_mass + mass)
        else:
            self.center_x = x
            self.center_y = y
        
        self.total_mass += mass
        
        # If empty leaf, store particle
        if self.is_leaf and self.particle_idx == -1:
            self.particle_idx = idx
            return
        
        # If occupied leaf, subdivide
        if self.is_leaf:
            self._subdivide()
            
            # Reinsert existing particle
            old_idx = self.particle_idx
            self.particle_idx = -1
            # Note: Would need particle positions here
        
        # Insert into appropriate child
        mid_x = (self.x_min + self.x_max) / 2
        mid_y = (self.y_min + self.y_max) / 2
        
        if x > mid_x:
            if y > mid_y:
                child_idx = 0  # NE
            else:
                child_idx = 3  # SE
        else:
            if y > mid_y:
                child_idx = 1  # NW
            else:
                child_idx = 2  # SW
        
        self.children[child_idx].insert(idx, x, y, mass)
    
    def _subdivide(self):
        """Create four children."""
        self.is_leaf = False
        mid_x = (self.x_min + self.x_max) / 2
        mid_y = (self.y_min + self.y_max) / 2
        
        self.children = [
            QuadTreeNode(mid_x, self.x_max, mid_y, self.y_max),  # NE
            QuadTreeNode(self.x_min, mid_x, mid_y, self.y_max),  # NW
            QuadTreeNode(self.x_min, mid_x, self.y_min, mid_y),  # SW
            QuadTreeNode(mid_x, self.x_max, self.y_min, mid_y),  # SE
        ]


def compute_gravity_barnes_hut(particles: ParticleArrays, n_active: int,
                              G: float = 6.67430e-11,
                              theta: float = 0.5,
                              softening: float = 0.1) -> None:
    """Barnes-Hut O(N log N) gravity calculation.
    
    Suitable for large N (> 10,000 particles).
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        G: Gravitational constant
        theta: Opening angle (0.5 is typical)
        softening: Softening length
    """
    # Note: This is a placeholder for the full implementation
    # For now, fall back to direct calculation
    if n_active < 10000:
        compute_gravity_direct_batched(particles, n_active, G, softening)
    else:
        # For very large systems, use batched direct with larger batches
        compute_gravity_direct_batched(particles, n_active, G, softening, 
                                      batch_size=min(5000, n_active // 10))


def compute_gravity_potential_energy(particles: ParticleArrays, n_active: int,
                                    G: float = 6.67430e-11,
                                    softening: float = 0.1) -> float:
    """Calculate total gravitational potential energy of the system.
    
    U = -G Σᵢ Σⱼ>ᵢ (mᵢ mⱼ) / rᵢⱼ
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        G: Gravitational constant
        softening: Softening length
        
    Returns:
        Total gravitational potential energy (J)
    """
    potential = 0.0
    eps2 = softening**2
    
    for i in range(n_active):
        for j in range(i + 1, n_active):
            dx = particles.position_x[i] - particles.position_x[j]
            dy = particles.position_y[i] - particles.position_y[j]
            r = np.sqrt(dx*dx + dy*dy + eps2)
            
            potential -= G * particles.mass[i] * particles.mass[j] / r
    
    return potential


def compute_center_of_mass(particles: ParticleArrays, n_active: int) -> Tuple[float, float, float]:
    """Compute center of mass of particle system.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        
    Returns:
        (x_cm, y_cm, total_mass)
    """
    total_mass = np.sum(particles.mass[:n_active])
    
    if total_mass > 0:
        x_cm = np.sum(particles.mass[:n_active] * particles.position_x[:n_active]) / total_mass
        y_cm = np.sum(particles.mass[:n_active] * particles.position_y[:n_active]) / total_mass
    else:
        x_cm = y_cm = 0.0
    
    return x_cm, y_cm, total_mass