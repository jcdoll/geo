"""
Unified GPU implementation for SPH that keeps data on GPU.

This implementation minimizes CPU-GPU transfers by keeping all particle
data on the GPU and performing all computations there.
"""

import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from ..core.backend import backend_function, for_backend, Backend
from ..core.particles import ParticleArrays
from ..core.kernel_vectorized import CubicSplineKernel


# JIT-compiled kernel functions
@torch.jit.script
def cubic_spline_W(q: torch.Tensor, norm: float) -> torch.Tensor:
    """Cubic spline kernel value."""
    W = torch.zeros_like(q)
    
    mask1 = q <= 1.0
    mask2 = (q > 1.0) & (q <= 2.0)
    
    W = torch.where(mask1, 
                    norm * (1.0 - 1.5 * q**2 + 0.75 * q**3),
                    W)
    W = torch.where(mask2,
                    norm * 0.25 * (2.0 - q)**3,
                    W)
    
    return W


@torch.jit.script
def cubic_spline_grad_W(q: torch.Tensor, norm_grad: float) -> torch.Tensor:
    """Cubic spline kernel gradient magnitude."""
    grad = torch.zeros_like(q)
    
    mask1 = q <= 1.0
    mask2 = (q > 1.0) & (q <= 2.0)
    
    grad = torch.where(mask1,
                      norm_grad * (-3.0 * q + 2.25 * q**2),
                      grad)
    grad = torch.where(mask2,
                      -norm_grad * 0.75 * (2.0 - q)**2,
                      grad)
    
    return grad


class GPUUnifiedSPH:
    """Unified GPU SPH implementation with persistent memory."""
    
    def __init__(self, max_particles: int = 100000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_particles = max_particles
        
        # Pre-allocate all arrays on GPU
        self.gpu_data = {
            'position_x': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'position_y': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'velocity_x': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'velocity_y': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'force_x': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'force_y': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'mass': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'density': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'pressure': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'smoothing_h': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'material_type': torch.zeros(max_particles, device=self.device, dtype=torch.int32),
        }
        
        # Track what needs syncing
        self.needs_upload = set()
        self.needs_download = set()
        
    def upload_particles(self, particles: ParticleArrays, n_active: int, arrays: Optional[list] = None):
        """Upload particle data to GPU."""
        if arrays is None:
            arrays = ['position_x', 'position_y', 'velocity_x', 'velocity_y', 
                     'mass', 'smoothing_h', 'material_type']
        
        for name in arrays:
            if hasattr(particles, name):
                cpu_data = getattr(particles, name)[:n_active]
                self.gpu_data[name][:n_active].copy_(torch.from_numpy(cpu_data))
        
        # Clear upload flags
        self.needs_upload.difference_update(arrays)
    
    def download_particles(self, particles: ParticleArrays, n_active: int, arrays: Optional[list] = None):
        """Download particle data from GPU."""
        if arrays is None:
            arrays = list(self.needs_download)
        
        for name in arrays:
            if hasattr(particles, name) and name in self.gpu_data:
                gpu_data = self.gpu_data[name][:n_active]
                getattr(particles, name)[:n_active] = gpu_data.cpu().numpy()
        
        # Clear download flags
        self.needs_download.difference_update(arrays)
    
    def compute_density_gpu(self, n_active: int, use_neighbors: bool = False,
                           neighbor_ids: Optional[np.ndarray] = None,
                           neighbor_distances: Optional[np.ndarray] = None,
                           neighbor_count: Optional[np.ndarray] = None):
        """Compute density entirely on GPU."""
        # Get active particle views
        pos_x = self.gpu_data['position_x'][:n_active]
        pos_y = self.gpu_data['position_y'][:n_active]
        mass = self.gpu_data['mass'][:n_active]
        h = self.gpu_data['smoothing_h'][:n_active]
        
        if use_neighbors and neighbor_ids is not None:
            # Neighbor-based computation
            neighbor_ids_gpu = torch.from_numpy(neighbor_ids[:n_active]).to(self.device)
            neighbor_distances_gpu = torch.from_numpy(neighbor_distances[:n_active]).to(self.device)
            neighbor_count_gpu = torch.from_numpy(neighbor_count[:n_active]).to(self.device)
            
            # Self contribution
            h_avg = h.mean()
            norm_2d = 10.0 / (7.0 * np.pi * h_avg * h_avg)
            self.gpu_data['density'][:n_active] = mass * norm_2d
            
            # Neighbor contributions
            max_neighbors = neighbor_ids_gpu.shape[1]
            neighbor_range = torch.arange(max_neighbors, device=self.device).unsqueeze(0)
            valid_mask = neighbor_range < neighbor_count_gpu.unsqueeze(1)
            
            # Safe indexing
            neighbor_ids_safe = torch.where(valid_mask, neighbor_ids_gpu, 0)
            neighbor_masses = mass[neighbor_ids_safe]
            
            # Kernel values
            q = neighbor_distances_gpu / h_avg
            W = cubic_spline_W(q, norm_2d)
            W = torch.where(valid_mask, W, 0.0)
            
            # Sum contributions
            self.gpu_data['density'][:n_active] += (neighbor_masses * W).sum(dim=1)
            
        else:
            # All-pairs computation (for small systems)
            dx = pos_x.unsqueeze(1) - pos_x.unsqueeze(0)
            dy = pos_y.unsqueeze(1) - pos_y.unsqueeze(0)
            r = torch.sqrt(dx**2 + dy**2 + 1e-10)
            
            h_avg = h.mean()
            norm_2d = 10.0 / (7.0 * np.pi * h_avg * h_avg)
            
            q = r / h_avg
            W = cubic_spline_W(q, norm_2d)
            
            self.gpu_data['density'][:n_active] = torch.sum(mass.unsqueeze(0) * W, dim=1)
        
        self.needs_download.add('density')
    
    def compute_forces_gpu(self, n_active: int, gravity: Tuple[float, float] = (0.0, -9.81),
                          alpha_visc: float = 0.1, use_neighbors: bool = False,
                          neighbor_ids: Optional[np.ndarray] = None,
                          neighbor_distances: Optional[np.ndarray] = None,
                          neighbor_count: Optional[np.ndarray] = None):
        """Compute forces entirely on GPU."""
        # Get active views
        pos_x = self.gpu_data['position_x'][:n_active]
        pos_y = self.gpu_data['position_y'][:n_active]
        vel_x = self.gpu_data['velocity_x'][:n_active]
        vel_y = self.gpu_data['velocity_y'][:n_active]
        mass = self.gpu_data['mass'][:n_active]
        density = self.gpu_data['density'][:n_active]
        pressure = self.gpu_data['pressure'][:n_active]
        h = self.gpu_data['smoothing_h'][:n_active]
        
        # Initialize with gravity
        self.gpu_data['force_x'][:n_active] = mass * gravity[0]
        self.gpu_data['force_y'][:n_active] = mass * gravity[1]
        
        if use_neighbors and neighbor_ids is not None:
            # Neighbor-based forces
            neighbor_ids_gpu = torch.from_numpy(neighbor_ids[:n_active]).to(self.device)
            neighbor_distances_gpu = torch.from_numpy(neighbor_distances[:n_active]).to(self.device)
            neighbor_count_gpu = torch.from_numpy(neighbor_count[:n_active]).to(self.device)
            
            max_neighbors = neighbor_ids_gpu.shape[1]
            neighbor_range = torch.arange(max_neighbors, device=self.device).unsqueeze(0)
            valid_mask = neighbor_range < neighbor_count_gpu.unsqueeze(1)
            
            # Batch process all neighbors
            h_avg = h.mean()
            norm_grad = 10.0 / (7.0 * np.pi * h_avg * h_avg * h_avg)
            
            # Get neighbor properties
            neighbor_ids_safe = torch.where(valid_mask, neighbor_ids_gpu, 0)
            pos_x_j = pos_x[neighbor_ids_safe]
            pos_y_j = pos_y[neighbor_ids_safe]
            vel_x_j = vel_x[neighbor_ids_safe]
            vel_y_j = vel_y[neighbor_ids_safe]
            mass_j = mass[neighbor_ids_safe]
            density_j = density[neighbor_ids_safe]
            pressure_j = pressure[neighbor_ids_safe]
            
            # Compute differences
            dx = pos_x.unsqueeze(1) - pos_x_j
            dy = pos_y.unsqueeze(1) - pos_y_j
            dvx = vel_x.unsqueeze(1) - vel_x_j
            dvy = vel_y.unsqueeze(1) - vel_y_j
            
            # Use precomputed distances
            r = neighbor_distances_gpu
            q = r / h_avg
            
            # Kernel gradient
            grad_mag = cubic_spline_grad_W(q, norm_grad)
            grad_mag = torch.where(valid_mask & (r > 1e-6), grad_mag, 0.0)
            
            # Gradient vector
            grad_x = grad_mag * dx / (r + 1e-10)
            grad_y = grad_mag * dy / (r + 1e-10)
            
            # Pressure forces
            pressure_i_term = pressure.unsqueeze(1) / (density.unsqueeze(1) ** 2)
            pressure_j_term = pressure_j / (density_j ** 2 + 1e-10)
            pressure_term = pressure_i_term + pressure_j_term
            
            # Artificial viscosity
            if alpha_visc > 0:
                v_dot_r = dvx * dx + dvy * dy
                visc_mask = (v_dot_r < 0) & valid_mask
                
                c_i = 10.0 * torch.sqrt(torch.abs(pressure) / density + 1e-6)
                c_j = 10.0 * torch.sqrt(torch.abs(pressure_j) / (density_j + 1e-10) + 1e-6)
                c_ij = 0.5 * (c_i.unsqueeze(1) + c_j)
                h_ij = h_avg
                mu_ij = h_ij * v_dot_r / (r**2 + 0.01 * h_ij**2)
                rho_ij = 0.5 * (density.unsqueeze(1) + density_j)
                
                visc_term = torch.where(visc_mask,
                                       -alpha_visc * c_ij * mu_ij / (rho_ij + 1e-10),
                                       torch.zeros_like(mu_ij))
            else:
                visc_term = 0.0
            
            # Total SPH force
            force_factor = -mass_j * (pressure_term + visc_term)
            force_x = (force_factor * grad_x).sum(dim=1)
            force_y = (force_factor * grad_y).sum(dim=1)
            
            self.gpu_data['force_x'][:n_active] += force_x
            self.gpu_data['force_y'][:n_active] += force_y
            
        else:
            # All-pairs forces (for small systems)
            dx = pos_x.unsqueeze(1) - pos_x.unsqueeze(0)
            dy = pos_y.unsqueeze(1) - pos_y.unsqueeze(0)
            dvx = vel_x.unsqueeze(1) - vel_x.unsqueeze(0)
            dvy = vel_y.unsqueeze(1) - vel_y.unsqueeze(0)
            r = torch.sqrt(dx**2 + dy**2 + 1e-10)
            
            h_avg = h.mean()
            norm_grad = 10.0 / (7.0 * np.pi * h_avg * h_avg * h_avg)
            
            # Kernel gradient
            q = r / h_avg
            grad_mag = cubic_spline_grad_W(q, norm_grad)
            
            # Mask self-interaction and far particles
            mask = (r > 1e-6) & (q <= 2.0)
            grad_mag = torch.where(mask, grad_mag, 0.0)
            
            # Gradient vector
            grad_x = grad_mag * dx / r
            grad_y = grad_mag * dy / r
            
            # Pressure term
            pressure_i = (pressure / (density**2)).unsqueeze(1)
            pressure_j = (pressure / (density**2)).unsqueeze(0)
            pressure_term = pressure_i + pressure_j
            
            # SPH forces
            force_factor = -mass.unsqueeze(0) * pressure_term
            force_x = (force_factor * grad_x).sum(dim=1)
            force_y = (force_factor * grad_y).sum(dim=1)
            
            self.gpu_data['force_x'][:n_active] += force_x
            self.gpu_data['force_y'][:n_active] += force_y
        
        self.needs_download.add('force_x')
        self.needs_download.add('force_y')
    
    def integrate_gpu(self, n_active: int, dt: float, damping: float = 0.0):
        """Perform integration directly on GPU."""
        # Get views
        pos_x = self.gpu_data['position_x'][:n_active]
        pos_y = self.gpu_data['position_y'][:n_active]
        vel_x = self.gpu_data['velocity_x'][:n_active]
        vel_y = self.gpu_data['velocity_y'][:n_active]
        force_x = self.gpu_data['force_x'][:n_active]
        force_y = self.gpu_data['force_y'][:n_active]
        mass = self.gpu_data['mass'][:n_active]
        
        # Leapfrog integration
        accel_x = force_x / mass
        accel_y = force_y / mass
        
        vel_x += accel_x * dt
        vel_y += accel_y * dt
        
        # Apply damping
        if damping > 0:
            vel_x *= (1.0 - damping * dt)
            vel_y *= (1.0 - damping * dt)
        
        # Update positions
        pos_x += vel_x * dt
        pos_y += vel_y * dt
        
        self.needs_download.update(['position_x', 'position_y', 'velocity_x', 'velocity_y'])
    
    def compute_pressure_gpu(self, n_active: int, rest_density: float = 1000.0, 
                            sound_speed: float = 100.0, gamma: float = 7.0):
        """Compute pressure using Tait equation of state on GPU."""
        density = self.gpu_data['density'][:n_active]
        
        # Tait equation: P = B * ((rho/rho0)^gamma - 1)
        B = rest_density * sound_speed**2 / gamma
        pressure_ratio = (density / rest_density) ** gamma
        self.gpu_data['pressure'][:n_active] = B * (pressure_ratio - 1.0)
        
        self.needs_download.add('pressure')


# Global instance
_gpu_sph = None


def get_gpu_sph(max_particles: int = 100000) -> GPUUnifiedSPH:
    """Get or create unified GPU SPH instance."""
    global _gpu_sph
    if _gpu_sph is None:
        _gpu_sph = GPUUnifiedSPH(max_particles)
    return _gpu_sph


# Backend functions
@backend_function("compute_density")
@for_backend(Backend.GPU)
def compute_density_unified(particles: ParticleArrays, kernel: CubicSplineKernel, n_active: int):
    """Unified GPU density computation."""
    gpu = get_gpu_sph()
    
    # Upload necessary data if not already on GPU
    gpu.upload_particles(particles, n_active, ['position_x', 'position_y', 'mass', 'smoothing_h'])
    
    # Check if we have neighbors
    use_neighbors = particles.neighbor_count is not None and particles.neighbor_count[0] > 0
    
    # Compute on GPU
    gpu.compute_density_gpu(
        n_active, 
        use_neighbors=use_neighbors,
        neighbor_ids=particles.neighbor_ids if use_neighbors else None,
        neighbor_distances=particles.neighbor_distances if use_neighbors else None,
        neighbor_count=particles.neighbor_count if use_neighbors else None
    )
    
    # Download results
    gpu.download_particles(particles, n_active, ['density'])


@backend_function("compute_forces")
@for_backend(Backend.GPU)
def compute_forces_unified(particles: ParticleArrays, kernel: CubicSplineKernel, n_active: int,
                          gravity: np.ndarray = None, alpha_visc: float = 0.1):
    """Unified GPU force computation."""
    gpu = get_gpu_sph()
    
    if gravity is None:
        gravity = np.array([0.0, -9.81])
    
    # Upload necessary data
    gpu.upload_particles(particles, n_active, 
                        ['position_x', 'position_y', 'velocity_x', 'velocity_y', 
                         'mass', 'density', 'pressure', 'smoothing_h'])
    
    # Check if we have neighbors
    use_neighbors = particles.neighbor_count is not None and particles.neighbor_count[0] > 0
    
    # Compute on GPU
    gpu.compute_forces_gpu(
        n_active,
        gravity=tuple(gravity),
        alpha_visc=alpha_visc,
        use_neighbors=use_neighbors,
        neighbor_ids=particles.neighbor_ids if use_neighbors else None,
        neighbor_distances=particles.neighbor_distances if use_neighbors else None,
        neighbor_count=particles.neighbor_count if use_neighbors else None
    )
    
    # Download results
    gpu.download_particles(particles, n_active, ['force_x', 'force_y'])


@backend_function("integrate")
@for_backend(Backend.GPU)
def integrate_unified(particles: ParticleArrays, n_active: int, dt: float, damping: float = 0.0):
    """GPU integration."""
    gpu = get_gpu_sph()
    
    # Upload if needed
    gpu.upload_particles(particles, n_active,
                        ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'force_x', 'force_y', 'mass'])
    
    # Integrate on GPU
    gpu.integrate_gpu(n_active, dt, damping)
    
    # Download results
    gpu.download_particles(particles, n_active, 
                         ['position_x', 'position_y', 'velocity_x', 'velocity_y'])


@backend_function("compute_pressure")
@for_backend(Backend.GPU)
def compute_pressure_unified(particles: ParticleArrays, n_active: int,
                           rest_density: float = 1000.0, sound_speed: float = 100.0):
    """GPU pressure computation."""
    gpu = get_gpu_sph()
    
    # Upload density if needed
    gpu.upload_particles(particles, n_active, ['density'])
    
    # Compute pressure
    gpu.compute_pressure_gpu(n_active, rest_density, sound_speed)
    
    # Download results
    gpu.download_particles(particles, n_active, ['pressure'])