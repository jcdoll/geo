"""
Optimized PyTorch GPU force computation for SPH.

Uses advanced PyTorch features for maximum performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from ..core.backend import backend_function, for_backend, Backend
from ..core.particles import ParticleArrays


# Pre-compile kernel gradient function with torch.jit.script
@torch.jit.script
def cubic_spline_gradient_jit(dx: torch.Tensor, dy: torch.Tensor, r: torch.Tensor, h: float) -> tuple:
    """JIT-compiled cubic spline kernel gradient."""
    q = r / h
    norm_2d = 10.0 / (7.0 * 3.14159265359 * h * h * h)
    
    grad_mag = torch.zeros_like(r)
    
    # Mask-based computation (no branches in GPU)
    mask1 = (q <= 1.0) & (r > 1e-10)
    mask2 = (q > 1.0) & (q <= 2.0) & (r > 1e-10)
    
    grad_mag = torch.where(mask1,
                          norm_2d * (-3 * q + 2.25 * q**2),
                          grad_mag)
    grad_mag = torch.where(mask2,
                          -norm_2d * 0.75 * (2 - q)**2,
                          grad_mag)
    
    # Apply direction
    factor = grad_mag / (r + 1e-10)
    grad_x = factor * dx
    grad_y = factor * dy
    
    return grad_x, grad_y


class ForceComputerGPU:
    """Persistent GPU force computer to avoid repeated memory transfers."""
    
    def __init__(self, max_particles: int = 100000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_particles = max_particles
        
        # Pre-allocate GPU memory
        self.pos_x_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.pos_y_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.vel_x_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.vel_y_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.mass_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.density_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.pressure_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.h_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.force_x_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        self.force_y_gpu = torch.zeros(max_particles, device=self.device, dtype=torch.float32)
        
    def compute_forces_all_pairs(self, particles: ParticleArrays, n_active: int,
                                gravity: np.ndarray = None, alpha_visc: float = 0.1):
        """All-pairs force computation optimized for GPU."""
        # Transfer only active particles
        self.pos_x_gpu[:n_active] = torch.from_numpy(particles.position_x[:n_active])
        self.pos_y_gpu[:n_active] = torch.from_numpy(particles.position_y[:n_active])
        self.vel_x_gpu[:n_active] = torch.from_numpy(particles.velocity_x[:n_active])
        self.vel_y_gpu[:n_active] = torch.from_numpy(particles.velocity_y[:n_active])
        self.mass_gpu[:n_active] = torch.from_numpy(particles.mass[:n_active])
        self.density_gpu[:n_active] = torch.from_numpy(particles.density[:n_active])
        self.pressure_gpu[:n_active] = torch.from_numpy(particles.pressure[:n_active])
        self.h_gpu[:n_active] = torch.from_numpy(particles.smoothing_h[:n_active])
        
        # Work with views of active particles only
        pos_x = self.pos_x_gpu[:n_active]
        pos_y = self.pos_y_gpu[:n_active]
        vel_x = self.vel_x_gpu[:n_active]
        vel_y = self.vel_y_gpu[:n_active]
        mass = self.mass_gpu[:n_active]
        density = self.density_gpu[:n_active]
        pressure = self.pressure_gpu[:n_active]
        h = self.h_gpu[:n_active]
        
        # Initialize forces with gravity
        if gravity is not None:
            self.force_x_gpu[:n_active] = mass * gravity[0]
            self.force_y_gpu[:n_active] = mass * gravity[1]
        else:
            self.force_x_gpu[:n_active] = 0
            self.force_y_gpu[:n_active] = 0
        
        # Batch compute all pairwise differences at once
        dx = pos_x.unsqueeze(1) - pos_x.unsqueeze(0)
        dy = pos_y.unsqueeze(1) - pos_y.unsqueeze(0)
        dvx = vel_x.unsqueeze(1) - vel_x.unsqueeze(0)
        dvy = vel_y.unsqueeze(1) - vel_y.unsqueeze(0)
        r = torch.sqrt(dx**2 + dy**2 + 1e-10)
        
        # Pressure term: P_i/ρ_i² + P_j/ρ_j²
        pressure_over_rho2 = pressure / (density * density)
        pressure_term = pressure_over_rho2.unsqueeze(1) + pressure_over_rho2.unsqueeze(0)
        
        # Artificial viscosity (optimized)
        visc_term = torch.zeros_like(r)
        if alpha_visc > 0:
            v_dot_r = dvx * dx + dvy * dy
            approaching = v_dot_r < 0
            
            if approaching.any():
                # Sound speed estimate
                c_sound = 10.0 * torch.sqrt(torch.abs(pressure) / density + 1e-6)
                c_ij = 0.5 * (c_sound.unsqueeze(1) + c_sound.unsqueeze(0))
                
                # Monaghan viscosity
                h_avg = h.mean().item()
                mu_ij = h_avg * v_dot_r / (r * r + 0.01 * h_avg * h_avg)
                rho_ij = 0.5 * (density.unsqueeze(1) + density.unsqueeze(0))
                
                visc_term = torch.where(approaching,
                                      -alpha_visc * c_ij * mu_ij / rho_ij,
                                      visc_term)
        
        # Kernel gradient (use average h for simplicity)
        h_avg = h.mean().item()
        grad_x, grad_y = cubic_spline_gradient_jit(dx, dy, r, h_avg)
        
        # Total force contribution from each pair
        force_factor = -mass.unsqueeze(0) * (pressure_term + visc_term)
        
        # Create mask for self-interactions more efficiently
        eye_mask = torch.eye(n_active, device=self.device, dtype=torch.bool)
        force_factor = force_factor.masked_fill(eye_mask, 0)
        grad_x = grad_x.masked_fill(eye_mask, 0)
        grad_y = grad_y.masked_fill(eye_mask, 0)
        
        # Sum forces on each particle
        self.force_x_gpu[:n_active] += torch.sum(force_factor * grad_x, dim=1)
        self.force_y_gpu[:n_active] += torch.sum(force_factor * grad_y, dim=1)
        
        # Copy back to CPU
        particles.force_x[:n_active] = self.force_x_gpu[:n_active].cpu().numpy()
        particles.force_y[:n_active] = self.force_y_gpu[:n_active].cpu().numpy()
    
    def compute_forces_neighbors(self, particles: ParticleArrays, n_active: int,
                               gravity: np.ndarray = None, alpha_visc: float = 0.1):
        """Neighbor-based force computation (more efficient for large systems)."""
        # Transfer data
        self.pos_x_gpu[:n_active] = torch.from_numpy(particles.position_x[:n_active])
        self.pos_y_gpu[:n_active] = torch.from_numpy(particles.position_y[:n_active])
        self.vel_x_gpu[:n_active] = torch.from_numpy(particles.velocity_x[:n_active])
        self.vel_y_gpu[:n_active] = torch.from_numpy(particles.velocity_y[:n_active])
        self.mass_gpu[:n_active] = torch.from_numpy(particles.mass[:n_active])
        self.density_gpu[:n_active] = torch.from_numpy(particles.density[:n_active])
        self.pressure_gpu[:n_active] = torch.from_numpy(particles.pressure[:n_active])
        self.h_gpu[:n_active] = torch.from_numpy(particles.smoothing_h[:n_active])
        
        neighbor_ids = torch.from_numpy(particles.neighbor_ids[:n_active]).to(self.device)
        neighbor_distances = torch.from_numpy(particles.neighbor_distances[:n_active]).to(self.device)
        neighbor_count = torch.from_numpy(particles.neighbor_count[:n_active]).to(self.device)
        
        # Initialize forces with gravity
        if gravity is not None:
            self.force_x_gpu[:n_active] = self.mass_gpu[:n_active] * gravity[0]
            self.force_y_gpu[:n_active] = self.mass_gpu[:n_active] * gravity[1]
        else:
            self.force_x_gpu[:n_active] = 0
            self.force_y_gpu[:n_active] = 0
        
        # Views for active particles
        pos_x = self.pos_x_gpu[:n_active]
        pos_y = self.pos_y_gpu[:n_active]
        vel_x = self.vel_x_gpu[:n_active]
        vel_y = self.vel_y_gpu[:n_active]
        mass = self.mass_gpu[:n_active]
        density = self.density_gpu[:n_active]
        pressure = self.pressure_gpu[:n_active]
        h = self.h_gpu[:n_active]
        
        # Create mask for valid neighbors
        max_neighbors = neighbor_ids.shape[1]
        neighbor_range = torch.arange(max_neighbors, device=self.device).unsqueeze(0)
        valid_mask = neighbor_range < neighbor_count.unsqueeze(1)
        
        # Set invalid neighbor IDs to 0 (will be masked out)
        neighbor_ids_safe = torch.where(valid_mask, neighbor_ids, 0)
        
        # Gather neighbor properties
        neighbor_pos_x = pos_x[neighbor_ids_safe]
        neighbor_pos_y = pos_y[neighbor_ids_safe]
        neighbor_vel_x = vel_x[neighbor_ids_safe]
        neighbor_vel_y = vel_y[neighbor_ids_safe]
        neighbor_mass = mass[neighbor_ids_safe]
        neighbor_density = density[neighbor_ids_safe]
        neighbor_pressure = pressure[neighbor_ids_safe]
        
        # Compute differences for all neighbors at once
        dx = pos_x.unsqueeze(1) - neighbor_pos_x
        dy = pos_y.unsqueeze(1) - neighbor_pos_y
        dvx = vel_x.unsqueeze(1) - neighbor_vel_x
        dvy = vel_y.unsqueeze(1) - neighbor_vel_y
        
        # Pressure gradient term
        pressure_over_rho2 = pressure / (density * density)
        neighbor_pressure_over_rho2 = neighbor_pressure / (neighbor_density * neighbor_density)
        pressure_term = pressure_over_rho2.unsqueeze(1) + neighbor_pressure_over_rho2
        
        # Artificial viscosity
        visc_term = torch.zeros_like(neighbor_distances)
        if alpha_visc > 0:
            v_dot_r = dvx * dx + dvy * dy
            approaching = (v_dot_r < 0) & valid_mask
            
            if approaching.any():
                c_sound = 10.0 * torch.sqrt(torch.abs(pressure) / density + 1e-6)
                neighbor_c_sound = 10.0 * torch.sqrt(torch.abs(neighbor_pressure) / neighbor_density + 1e-6)
                c_ij = 0.5 * (c_sound.unsqueeze(1) + neighbor_c_sound)
                
                h_expanded = h.unsqueeze(1).expand(-1, max_neighbors)
                mu_ij = h_expanded * v_dot_r / (neighbor_distances**2 + 0.01 * h_expanded**2)
                rho_ij = 0.5 * (density.unsqueeze(1) + neighbor_density)
                
                visc_term = torch.where(approaching,
                                      -alpha_visc * c_ij * mu_ij / rho_ij,
                                      visc_term)
        
        # Kernel gradient for all neighbors
        h_expanded = h.unsqueeze(1).expand(-1, max_neighbors)
        grad_x, grad_y = cubic_spline_gradient_jit(dx, dy, neighbor_distances, h.mean().item())
        
        # Mask out invalid neighbors
        grad_x = torch.where(valid_mask, grad_x, 0.0)
        grad_y = torch.where(valid_mask, grad_y, 0.0)
        
        # Total force
        force_factor = -neighbor_mass * (pressure_term + visc_term)
        force_factor = torch.where(valid_mask, force_factor, 0.0)
        
        # Sum contributions
        self.force_x_gpu[:n_active] += (force_factor * grad_x).sum(dim=1)
        self.force_y_gpu[:n_active] += (force_factor * grad_y).sum(dim=1)
        
        # Copy back
        particles.force_x[:n_active] = self.force_x_gpu[:n_active].cpu().numpy()
        particles.force_y[:n_active] = self.force_y_gpu[:n_active].cpu().numpy()


# Global instance to maintain GPU state
_force_computer = None


@backend_function("compute_forces")
@for_backend(Backend.GPU)
def compute_forces_torch_optimized(particles: ParticleArrays, kernel, n_active: int,
                                 gravity: np.ndarray = None, alpha_visc: float = 0.1):
    """Optimized GPU force computation."""
    global _force_computer
    
    # Create persistent computer on first use
    if _force_computer is None:
        _force_computer = ForceComputerGPU()
    
    # Use neighbor-based method for better scaling
    if particles.neighbor_count[0] > 0:  # Check if neighbors are available
        _force_computer.compute_forces_neighbors(particles, n_active, gravity, alpha_visc)
    else:
        _force_computer.compute_forces_all_pairs(particles, n_active, gravity, alpha_visc)