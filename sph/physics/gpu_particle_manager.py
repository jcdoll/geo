"""
GPU Particle Manager for persistent GPU memory.

Keeps particle data on GPU between frames to minimize transfers.
"""

import torch
import numpy as np
from typing import Optional, Dict
from ..core.particles import ParticleArrays
from ..core.backend import get_backend


class GPUParticleManager:
    """Manages particle data on GPU to minimize CPU-GPU transfers."""
    
    def __init__(self, max_particles: int = 100000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_particles = max_particles
        self.synced = False
        
        # GPU tensors for all particle properties
        self.gpu_arrays = {
            'position_x': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'position_y': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'velocity_x': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'velocity_y': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'force_x': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'force_y': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'mass': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'density': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'pressure': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'temperature': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'smoothing_h': torch.zeros(max_particles, device=self.device, dtype=torch.float32),
            'material_type': torch.zeros(max_particles, device=self.device, dtype=torch.int32),
            'phase_id': torch.zeros(max_particles, device=self.device, dtype=torch.int32),
        }
        
        # Track which arrays need syncing
        self.dirty_cpu = set()  # CPU arrays that need uploading
        self.dirty_gpu = set()  # GPU arrays that need downloading
        
    def upload_to_gpu(self, particles: ParticleArrays, n_active: int, 
                     arrays: Optional[list] = None):
        """Upload specific arrays or all arrays to GPU."""
        if arrays is None:
            arrays = list(self.gpu_arrays.keys())
        
        for name in arrays:
            if hasattr(particles, name):
                cpu_array = getattr(particles, name)[:n_active]
                self.gpu_arrays[name][:n_active].copy_(torch.from_numpy(cpu_array))
        
        # Mark as synced
        for name in arrays:
            if name in self.dirty_cpu:
                self.dirty_cpu.remove(name)
        
        self.synced = True
    
    def download_from_gpu(self, particles: ParticleArrays, n_active: int,
                         arrays: Optional[list] = None):
        """Download specific arrays or all arrays from GPU."""
        if arrays is None:
            arrays = list(self.gpu_arrays.keys())
        
        for name in arrays:
            if hasattr(particles, name) and name in self.dirty_gpu:
                gpu_array = self.gpu_arrays[name][:n_active]
                getattr(particles, name)[:n_active] = gpu_array.cpu().numpy()
        
        # Clear dirty flags
        for name in arrays:
            if name in self.dirty_gpu:
                self.dirty_gpu.remove(name)
    
    def get_gpu_view(self, name: str, n_active: int) -> torch.Tensor:
        """Get a view of active particles for a specific array."""
        return self.gpu_arrays[name][:n_active]
    
    def mark_dirty(self, arrays: list, on_gpu: bool = True):
        """Mark arrays as modified."""
        if on_gpu:
            self.dirty_gpu.update(arrays)
        else:
            self.dirty_cpu.update(arrays)
    
    def sync_positions(self, particles: ParticleArrays, n_active: int, to_gpu: bool = True):
        """Quick sync for positions only (common operation)."""
        if to_gpu:
            self.gpu_arrays['position_x'][:n_active].copy_(
                torch.from_numpy(particles.position_x[:n_active]))
            self.gpu_arrays['position_y'][:n_active].copy_(
                torch.from_numpy(particles.position_y[:n_active]))
        else:
            particles.position_x[:n_active] = self.gpu_arrays['position_x'][:n_active].cpu().numpy()
            particles.position_y[:n_active] = self.gpu_arrays['position_y'][:n_active].cpu().numpy()
    
    def integrate_on_gpu(self, n_active: int, dt: float, 
                        damping: float = 0.0, max_velocity: float = 100.0):
        """Perform integration directly on GPU."""
        # Get views
        pos_x = self.gpu_arrays['position_x'][:n_active]
        pos_y = self.gpu_arrays['position_y'][:n_active]
        vel_x = self.gpu_arrays['velocity_x'][:n_active]
        vel_y = self.gpu_arrays['velocity_y'][:n_active]
        force_x = self.gpu_arrays['force_x'][:n_active]
        force_y = self.gpu_arrays['force_y'][:n_active]
        mass = self.gpu_arrays['mass'][:n_active]
        
        # Leapfrog integration
        # v(t+dt/2) = v(t-dt/2) + a(t)*dt
        accel_x = force_x / mass
        accel_y = force_y / mass
        
        vel_x += accel_x * dt
        vel_y += accel_y * dt
        
        # Apply damping
        if damping > 0:
            vel_x *= (1.0 - damping * dt)
            vel_y *= (1.0 - damping * dt)
        
        # Velocity clamping
        vel_mag = torch.sqrt(vel_x**2 + vel_y**2)
        clamped = vel_mag > max_velocity
        if clamped.any():
            scale = max_velocity / (vel_mag + 1e-10)
            vel_x = torch.where(clamped, vel_x * scale, vel_x)
            vel_y = torch.where(clamped, vel_y * scale, vel_y)
        
        # x(t+dt) = x(t) + v(t+dt/2)*dt
        pos_x += vel_x * dt
        pos_y += vel_y * dt
        
        # Mark as dirty
        self.dirty_gpu.update(['position_x', 'position_y', 'velocity_x', 'velocity_y'])
    
    def apply_boundary_conditions_gpu(self, n_active: int, domain_min: tuple, domain_max: tuple,
                                    boundary_damping: float = 0.5):
        """Apply boundary conditions directly on GPU."""
        pos_x = self.gpu_arrays['position_x'][:n_active]
        pos_y = self.gpu_arrays['position_y'][:n_active]
        vel_x = self.gpu_arrays['velocity_x'][:n_active]
        vel_y = self.gpu_arrays['velocity_y'][:n_active]
        
        # X boundaries
        out_left = pos_x < domain_min[0]
        out_right = pos_x > domain_max[0]
        
        pos_x = torch.where(out_left, domain_min[0], pos_x)
        pos_x = torch.where(out_right, domain_max[0], pos_x)
        vel_x = torch.where(out_left | out_right, -vel_x * boundary_damping, vel_x)
        
        # Y boundaries
        out_bottom = pos_y < domain_min[1]
        out_top = pos_y > domain_max[1]
        
        pos_y = torch.where(out_bottom, domain_min[1], pos_y)
        pos_y = torch.where(out_top, domain_max[1], pos_y)
        vel_y = torch.where(out_bottom | out_top, -vel_y * boundary_damping, vel_y)
        
        self.dirty_gpu.update(['position_x', 'position_y', 'velocity_x', 'velocity_y'])


# Global GPU particle manager
_gpu_manager = None


def get_gpu_particle_manager(max_particles: int = 100000) -> Optional[GPUParticleManager]:
    """Get or create GPU particle manager if GPU backend is active."""
    global _gpu_manager
    
    if get_backend() != 'gpu':
        return None
    
    if _gpu_manager is None:
        _gpu_manager = GPUParticleManager(max_particles)
    
    return _gpu_manager