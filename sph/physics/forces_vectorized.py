"""
Fully vectorized force computation for SPH.

Includes:
- Pressure forces
- Viscous forces
- External forces (gravity)
- Artificial viscosity for stability
"""

import numpy as np
from typing import Optional, Tuple
from ..core.particles import ParticleArrays
from ..core.kernel_vectorized import CubicSplineKernel


def compute_forces_vectorized(particles: ParticleArrays, kernel: CubicSplineKernel,
                             n_active: int, gravity: np.ndarray = None,
                             alpha_visc: float = 0.1, beta_visc: float = 0.0):
    """Fully vectorized force computation.
    
    Computes:
    - Pressure gradient forces: -∇P/ρ
    - Artificial viscosity for shock handling
    - External forces (gravity)
    
    Args:
        particles: Particle arrays with neighbor information
        kernel: SPH kernel function
        n_active: Number of active particles
        gravity: Gravity vector [gx, gy], default [0, -9.81]
        alpha_visc: Artificial viscosity parameter (bulk)
        beta_visc: Artificial viscosity parameter (von Neumann-Richtmyer)
    """
    # Reset forces
    particles.reset_forces(n_active)
    
    # Add gravity if specified
    if gravity is not None:
        particles.force_x[:n_active] += particles.mass[:n_active] * gravity[0]
        particles.force_y[:n_active] += particles.mass[:n_active] * gravity[1]
    
    # Process particles in batches for cache efficiency
    batch_size = 256
    
    for batch_start in range(0, n_active, batch_size):
        batch_end = min(batch_start + batch_size, n_active)
        
        # Process each particle in batch
        for i in range(batch_start, batch_end):
            n_neighbors = particles.neighbor_count[i]
            if n_neighbors == 0:
                continue
            
            # Get neighbor data
            neighbor_slice = slice(0, n_neighbors)
            neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
            distances = particles.neighbor_distances[i, neighbor_slice]
            
            # Vectorized position differences
            dx = particles.position_x[i] - particles.position_x[neighbor_ids]
            dy = particles.position_y[i] - particles.position_y[neighbor_ids]
            
            # Vectorized velocity differences
            dvx = particles.velocity_x[i] - particles.velocity_x[neighbor_ids]
            dvy = particles.velocity_y[i] - particles.velocity_y[neighbor_ids]
            
            # Pressure gradient term (symmetric formulation)
            # F_pressure = -m_j * (P_i/ρ_i² + P_j/ρ_j²) * ∇W_ij
            pressure_i = particles.pressure[i] / (particles.density[i]**2)
            pressure_j = particles.pressure[neighbor_ids] / (particles.density[neighbor_ids]**2)
            pressure_term = pressure_i + pressure_j
            
            # Artificial viscosity
            visc_term = np.zeros_like(distances)
            if alpha_visc > 0:
                # Compute viscosity for approaching particles
                v_dot_r = dvx * dx + dvy * dy
                approaching = v_dot_r < 0
                
                if np.any(approaching):
                    # Monaghan artificial viscosity
                    h_i = particles.smoothing_h[i]
                    h_j = particles.smoothing_h[neighbor_ids[approaching]]
                    h_ij = 0.5 * (h_i + h_j)
                    
                    # Sound speed estimate (simplified)
                    c_i = 10.0 * np.sqrt(np.abs(particles.pressure[i]) / particles.density[i] + 1e-6)
                    c_j = 10.0 * np.sqrt(np.abs(particles.pressure[neighbor_ids[approaching]]) / 
                                        particles.density[neighbor_ids[approaching]] + 1e-6)
                    c_ij = 0.5 * (c_i + c_j)
                    
                    # Viscosity
                    mu_ij = h_ij * v_dot_r[approaching] / (distances[approaching]**2 + 0.01 * h_ij**2)
                    Pi_ij = (-alpha_visc * c_ij * mu_ij + beta_visc * mu_ij**2) / \
                            (0.5 * (particles.density[i] + particles.density[neighbor_ids[approaching]]))
                    
                    visc_term[approaching] = Pi_ij
            
            # Get kernel gradients
            grad_x, grad_y = kernel.gradW_vectorized(
                dx.reshape(1, -1), dy.reshape(1, -1),
                distances.reshape(1, -1),
                particles.smoothing_h[i]
            )
            
            # Total force contribution from neighbors
            force_factor = -particles.mass[neighbor_ids] * (pressure_term + visc_term)
            
            particles.force_x[i] += np.sum(force_factor * grad_x[0])
            particles.force_y[i] += np.sum(force_factor * grad_y[0])


def compute_viscous_forces_vectorized(particles: ParticleArrays, kernel: CubicSplineKernel,
                                     n_active: int, viscosity: float = 0.01):
    """Add physical viscosity forces (Cleary-Monaghan formulation).
    
    This is in addition to artificial viscosity and represents
    real fluid viscosity.
    
    Args:
        particles: Particle arrays
        kernel: SPH kernel
        n_active: Number of active particles
        viscosity: Dynamic viscosity coefficient
    """
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        distances = particles.neighbor_distances[i, neighbor_slice]
        
        # Position and velocity differences
        dx = particles.position_x[i] - particles.position_x[neighbor_ids]
        dy = particles.position_y[i] - particles.position_y[neighbor_ids]
        dvx = particles.velocity_x[i] - particles.velocity_x[neighbor_ids]
        dvy = particles.velocity_y[i] - particles.velocity_y[neighbor_ids]
        
        # Kernel gradients
        grad_x, grad_y = kernel.gradW_vectorized(
            dx.reshape(1, -1), dy.reshape(1, -1),
            distances.reshape(1, -1),
            particles.smoothing_h[i]
        )
        
        # Cleary-Monaghan viscosity
        # F_visc = 2μ * m_j/ρ_j * (v_i - v_j) · r_ij / (r_ij² + 0.01h²) * ∇W_ij
        r_dot_grad = (dx * grad_x[0] + dy * grad_y[0]) / (distances**2 + 
                     0.01 * particles.smoothing_h[i]**2)
        
        visc_factor = 2 * viscosity * particles.mass[neighbor_ids] / \
                     particles.density[neighbor_ids] * r_dot_grad
        
        particles.force_x[i] += np.sum(visc_factor * dvx)
        particles.force_y[i] += np.sum(visc_factor * dvy)


def tait_equation_of_state(density: np.ndarray, material_rho0: np.ndarray,
                          material_B: np.ndarray, gamma: float = 7.0) -> np.ndarray:
    """Tait equation of state for weakly compressible fluids.
    
    P = B[(ρ/ρ₀)^γ - 1]
    
    Args:
        density: Current density array
        material_rho0: Reference density for each particle's material
        material_B: Bulk modulus for each particle's material
        gamma: Polytropic constant (typically 7 for water)
    
    Returns:
        Pressure array
    """
    return material_B * ((density / material_rho0)**gamma - 1)


def ideal_gas_eos(density: np.ndarray, temperature: np.ndarray,
                  specific_gas_constant: np.ndarray) -> np.ndarray:
    """Ideal gas equation of state.
    
    P = ρRT
    
    Args:
        density: Density array
        temperature: Temperature array
        specific_gas_constant: R for each particle's material
    
    Returns:
        Pressure array
    """
    return density * specific_gas_constant * temperature


def compute_acceleration(particles: ParticleArrays, n_active: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute acceleration from forces.
    
    Returns:
        (ax, ay) acceleration arrays
    """
    ax = particles.force_x[:n_active] / particles.mass[:n_active]
    ay = particles.force_y[:n_active] / particles.mass[:n_active]
    return ax, ay