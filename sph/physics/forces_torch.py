"""
PyTorch GPU-accelerated force computation for SPH.

Provides massive speedup on RTX 5080 and other modern GPUs.
"""

import torch
import numpy as np
from ..core.backend import backend_function, for_backend, Backend
from ..core.particles import ParticleArrays


def cubic_spline_gradient_torch(dx: torch.Tensor, dy: torch.Tensor, r: torch.Tensor, h: float) -> tuple:
    """Compute cubic spline kernel gradient."""
    q = r / h
    norm_2d = 10.0 / (7.0 * np.pi * h * h * h)
    
    grad_mag = torch.zeros_like(r)
    
    mask1 = (q <= 1.0) & (r > 1e-10)
    mask2 = (q > 1.0) & (q <= 2.0) & (r > 1e-10)
    
    grad_mag[mask1] = norm_2d * (-3 * q[mask1] + 2.25 * q[mask1]**2)
    grad_mag[mask2] = -norm_2d * 0.75 * (2 - q[mask2])**2
    
    # Apply direction
    factor = grad_mag / (r + 1e-10)
    grad_x = factor * dx
    grad_y = factor * dy
    
    return grad_x, grad_y


@backend_function("compute_forces")
@for_backend(Backend.GPU)
def compute_forces_torch(particles: ParticleArrays, kernel, n_active: int, 
                        gravity: np.ndarray = None, alpha_visc: float = 0.1):
    """
    GPU-accelerated force computation using PyTorch.
    
    Computes pressure and viscosity forces efficiently on GPU.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transfer to GPU
    pos_x = torch.from_numpy(particles.position_x[:n_active]).to(device)
    pos_y = torch.from_numpy(particles.position_y[:n_active]).to(device)
    vel_x = torch.from_numpy(particles.velocity_x[:n_active]).to(device)
    vel_y = torch.from_numpy(particles.velocity_y[:n_active]).to(device)
    mass = torch.from_numpy(particles.mass[:n_active]).to(device)
    density = torch.from_numpy(particles.density[:n_active]).to(device)
    pressure = torch.from_numpy(particles.pressure[:n_active]).to(device)
    h = torch.from_numpy(particles.smoothing_h[:n_active]).to(device)
    
    # Initialize forces
    force_x = torch.zeros(n_active, device=device)
    force_y = torch.zeros(n_active, device=device)
    
    # Add gravity if provided
    if gravity is not None:
        force_x += mass * gravity[0]
        force_y += mass * gravity[1]
    
    # Compute all pairwise interactions
    # Position and velocity differences
    dx = pos_x.unsqueeze(1) - pos_x.unsqueeze(0)  # (n, n)
    dy = pos_y.unsqueeze(1) - pos_y.unsqueeze(0)  # (n, n)
    dvx = vel_x.unsqueeze(1) - vel_x.unsqueeze(0)  # (n, n)
    dvy = vel_y.unsqueeze(1) - vel_y.unsqueeze(0)  # (n, n)
    r = torch.sqrt(dx**2 + dy**2 + 1e-10)
    
    # Pressure term: P_i/ρ_i² + P_j/ρ_j²
    pressure_i = (pressure / (density * density)).unsqueeze(1)  # (n, 1)
    pressure_j = (pressure / (density * density)).unsqueeze(0)  # (1, n)
    pressure_term = pressure_i + pressure_j  # (n, n)
    
    # Artificial viscosity (if enabled)
    visc_term = torch.zeros_like(r)
    if alpha_visc > 0:
        v_dot_r = dvx * dx + dvy * dy
        approaching = v_dot_r < 0
        
        # Sound speed estimate
        c_sound = 10.0 * torch.sqrt(torch.abs(pressure) / density + 1e-6)
        c_i = c_sound.unsqueeze(1)
        c_j = c_sound.unsqueeze(0)
        c_ij = 0.5 * (c_i + c_j)
        
        # Monaghan viscosity
        h_ref = h[0].item()  # Use average h
        mu_ij = h_ref * v_dot_r / (r * r + 0.01 * h_ref * h_ref)
        rho_ij = 0.5 * (density.unsqueeze(1) + density.unsqueeze(0))
        
        visc_term[approaching] = -alpha_visc * c_ij[approaching] * mu_ij[approaching] / rho_ij[approaching]
    
    # Kernel gradient
    grad_x, grad_y = cubic_spline_gradient_torch(dx, dy, r, h[0].item())
    
    # Total force contribution from each pair
    force_factor = -mass.unsqueeze(0) * (pressure_term + visc_term)
    
    # Zero out self-interactions
    mask = torch.eye(n_active, device=device).bool()
    force_factor[mask] = 0
    grad_x[mask] = 0
    grad_y[mask] = 0
    
    # Sum forces on each particle
    force_x = torch.sum(force_factor * grad_x, dim=1)
    force_y = torch.sum(force_factor * grad_y, dim=1)
    
    # Transfer back
    particles.force_x[:n_active] = force_x.cpu().numpy()
    particles.force_y[:n_active] = force_y.cpu().numpy()


@backend_function("compute_forces_neighbors")
@for_backend(Backend.GPU)
def compute_forces_torch_neighbors(particles: ParticleArrays, kernel, n_active: int,
                                  gravity: np.ndarray = None, alpha_visc: float = 0.1):
    """
    GPU force computation using neighbor lists (more efficient for large systems).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transfer to GPU
    pos_x = torch.from_numpy(particles.position_x[:n_active]).to(device)
    pos_y = torch.from_numpy(particles.position_y[:n_active]).to(device)
    vel_x = torch.from_numpy(particles.velocity_x[:n_active]).to(device)
    vel_y = torch.from_numpy(particles.velocity_y[:n_active]).to(device)
    mass = torch.from_numpy(particles.mass[:n_active]).to(device)
    density = torch.from_numpy(particles.density[:n_active]).to(device)
    pressure = torch.from_numpy(particles.pressure[:n_active]).to(device)
    h = torch.from_numpy(particles.smoothing_h[:n_active]).to(device)
    neighbor_ids = torch.from_numpy(particles.neighbor_ids[:n_active]).to(device)
    neighbor_distances = torch.from_numpy(particles.neighbor_distances[:n_active]).to(device)
    neighbor_count = torch.from_numpy(particles.neighbor_count[:n_active]).to(device)
    
    # Initialize forces
    force_x = torch.zeros(n_active, device=device)
    force_y = torch.zeros(n_active, device=device)
    
    # Add gravity
    if gravity is not None:
        force_x += mass * gravity[0]
        force_y += mass * gravity[1]
    
    # Process each particle
    for i in range(n_active):
        n_neighbors = neighbor_count[i]
        if n_neighbors == 0:
            continue
            
        # Get valid neighbors
        valid_neighbors = neighbor_ids[i, :n_neighbors]
        valid_distances = neighbor_distances[i, :n_neighbors]
        
        # Position/velocity differences
        dx = pos_x[i] - pos_x[valid_neighbors]
        dy = pos_y[i] - pos_y[valid_neighbors]
        dvx = vel_x[i] - vel_x[valid_neighbors]
        dvy = vel_y[i] - vel_y[valid_neighbors]
        
        # Pressure gradient term
        pressure_i = pressure[i] / (density[i] * density[i])
        pressure_j = pressure[valid_neighbors] / (density[valid_neighbors] * density[valid_neighbors])
        pressure_term = pressure_i + pressure_j
        
        # Viscosity
        visc_term = torch.zeros_like(valid_distances)
        if alpha_visc > 0:
            v_dot_r = dvx * dx + dvy * dy
            approaching = v_dot_r < 0
            
            if approaching.any():
                c_i = 10.0 * torch.sqrt(torch.abs(pressure[i]) / density[i] + 1e-6)
                c_j = 10.0 * torch.sqrt(torch.abs(pressure[valid_neighbors]) / density[valid_neighbors] + 1e-6)
                c_ij = 0.5 * (c_i + c_j)
                
                h_i = h[i].item()
                mu_ij = h_i * v_dot_r / (valid_distances**2 + 0.01 * h_i**2)
                rho_ij = 0.5 * (density[i] + density[valid_neighbors])
                
                visc_term[approaching] = -alpha_visc * c_ij[approaching] * mu_ij[approaching] / rho_ij[approaching]
        
        # Kernel gradient
        grad_x, grad_y = cubic_spline_gradient_torch(dx, dy, valid_distances, h[i].item())
        
        # Total force
        force_factor = -mass[valid_neighbors] * (pressure_term + visc_term)
        force_x[i] += torch.sum(force_factor * grad_x)
        force_y[i] += torch.sum(force_factor * grad_y)
    
    # Transfer back
    particles.force_x[:n_active] = force_x.cpu().numpy()
    particles.force_y[:n_active] = force_y.cpu().numpy()