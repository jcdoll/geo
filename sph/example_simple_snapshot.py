#!/usr/bin/env python3
"""
SPH example that saves snapshots instead of interactive display.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Particle:
    """Simple particle for demonstration."""
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    density: float = 1000.0
    pressure: float = 0.0
    material: str = "water"

class SimpleKernel:
    """Simplified cubic spline kernel."""
    
    def __init__(self, h: float = 0.1):
        self.h = h
        self.norm_2d = 10.0 / (7.0 * np.pi * h * h)
    
    def W(self, r: float) -> float:
        """Kernel value at distance r."""
        q = r / self.h
        if q <= 1:
            return self.norm_2d * (1 - 1.5*q**2 + 0.75*q**3)
        elif q <= 2:
            return self.norm_2d * 0.25 * (2 - q)**3
        else:
            return 0.0
    
    def gradW(self, r_vec: np.ndarray) -> np.ndarray:
        """Kernel gradient vector."""
        r = np.linalg.norm(r_vec)
        if r < 1e-6:
            return np.zeros_like(r_vec)
        
        q = r / self.h
        if q <= 1:
            grad_mag = self.norm_2d * (-3*q + 2.25*q**2) / self.h
        elif q <= 2:
            grad_mag = -self.norm_2d * 0.75 * (2 - q)**2 / self.h
        else:
            return np.zeros_like(r_vec)
        
        return grad_mag * r_vec / r

def find_neighbors(particles: List[Particle], radius: float) -> List[List[int]]:
    """Simple O(N²) neighbor search for demonstration."""
    neighbors = [[] for _ in particles]
    
    for i, p_i in enumerate(particles):
        for j, p_j in enumerate(particles):
            if i != j:
                r = np.linalg.norm(p_i.position - p_j.position)
                if r < radius:
                    neighbors[i].append(j)
    
    return neighbors

def compute_density(particles: List[Particle], neighbors: List[List[int]], kernel: SimpleKernel):
    """SPH density summation."""
    for i, p_i in enumerate(particles):
        density = 0.0
        
        # Self contribution
        density += p_i.mass * kernel.W(0)
        
        # Neighbor contributions
        for j in neighbors[i]:
            p_j = particles[j]
            r = np.linalg.norm(p_i.position - p_j.position)
            density += p_j.mass * kernel.W(r)
        
        p_i.density = density

def tait_eos(density: float, rho0: float = 1000.0, B: float = 1e5, gamma: float = 7) -> float:
    """Tait equation of state for weakly compressible water."""
    return B * ((density/rho0)**gamma - 1)

def compute_forces(particles: List[Particle], neighbors: List[List[int]], kernel: SimpleKernel) -> List[np.ndarray]:
    """Compute SPH pressure forces."""
    forces = [np.zeros(2) for _ in particles]
    
    for i, p_i in enumerate(particles):
        for j in neighbors[i]:
            p_j = particles[j]
            
            # Pressure force (symmetric form)
            r_ij = p_i.position - p_j.position
            r_norm = np.linalg.norm(r_ij)
            
            if r_norm > 1e-6:
                # Pressure gradient
                pressure_term = p_i.pressure / (p_i.density**2) + p_j.pressure / (p_j.density**2)
                
                # Artificial viscosity for stability
                v_ij = p_i.velocity - p_j.velocity
                visc = 0.0
                if np.dot(v_ij, r_ij) < 0:  # Approaching
                    alpha = 0.01
                    visc = -alpha * kernel.h * np.dot(v_ij, r_ij) / (r_norm**2 + 0.01*kernel.h**2)
                
                force = -p_j.mass * (pressure_term + visc) * kernel.gradW(r_ij)
                forces[i] += force
    
    # Add gravity
    g = np.array([0, -9.81])
    for i, p in enumerate(particles):
        forces[i] += p.mass * g
    
    return forces

def integrate(particles: List[Particle], forces: List[np.ndarray], dt: float, bounds: Tuple[float, float, float, float]):
    """Simple Euler integration with boundary conditions."""
    xmin, xmax, ymin, ymax = bounds
    
    for i, p in enumerate(particles):
        # Update velocity
        p.velocity += forces[i] / p.mass * dt
        
        # Update position
        p.position += p.velocity * dt
        
        # Boundary conditions (simple reflection with damping)
        if p.position[0] < xmin:
            p.position[0] = xmin
            p.velocity[0] = -0.5 * p.velocity[0]
        elif p.position[0] > xmax:
            p.position[0] = xmax
            p.velocity[0] = -0.5 * p.velocity[0]
            
        if p.position[1] < ymin:
            p.position[1] = ymin
            p.velocity[1] = -0.5 * p.velocity[1]
        elif p.position[1] > ymax:
            p.position[1] = ymax
            p.velocity[1] = -0.5 * p.velocity[1]

def create_dam_break(nx: int = 10, ny: int = 20, spacing: float = 0.02) -> List[Particle]:
    """Create initial dam break configuration."""
    particles = []
    
    for i in range(nx):
        for j in range(ny):
            pos = np.array([0.1 + i * spacing, 0.1 + j * spacing])
            vel = np.zeros(2)
            
            p = Particle(
                position=pos,
                velocity=vel,
                mass=0.001,  # kg
                material="water"
            )
            particles.append(p)
    
    return particles

def simulate_and_snapshot():
    """Run simulation and save snapshots."""
    # Parameters
    kernel = SimpleKernel(h=0.04)
    dt = 0.0001
    steps = 500  # Reduced for quick test
    bounds = (0.0, 1.0, 0.0, 0.6)
    
    # Create particles
    particles = create_dam_break()
    print(f"Simulating {len(particles)} particles...")
    
    # Snapshots at specific times
    snapshot_steps = [0, 100, 250, 500]
    
    for step in range(steps + 1):
        # Find neighbors
        neighbors = find_neighbors(particles, 2 * kernel.h)
        
        # Compute density
        compute_density(particles, neighbors, kernel)
        
        # Compute pressure
        for p in particles:
            p.pressure = tait_eos(p.density)
        
        # Compute forces
        forces = compute_forces(particles, neighbors, kernel)
        
        # Integrate
        integrate(particles, forces, dt, bounds)
        
        # Save snapshot
        if step in snapshot_steps:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Extract positions and properties
            positions = np.array([p.position for p in particles])
            densities = np.array([p.density for p in particles])
            velocities = np.array([np.linalg.norm(p.velocity) for p in particles])
            
            # Particle positions colored by density
            scatter1 = ax1.scatter(
                positions[:, 0], positions[:, 1],
                c=densities, cmap='Blues',
                s=50, alpha=0.8,
                vmin=800, vmax=1200
            )
            
            ax1.set_xlim(bounds[0], bounds[1])
            ax1.set_ylim(bounds[2], bounds[3])
            ax1.set_aspect('equal')
            ax1.set_title(f'SPH Dam Break - Step {step} (t={step*dt:.3f}s)')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter1, ax=ax1, label='Density (kg/m³)')
            
            # Velocity field
            scatter2 = ax2.scatter(
                positions[:, 0], positions[:, 1],
                c=velocities, cmap='viridis',
                s=50, alpha=0.8,
                vmin=0, vmax=2.0
            )
            
            ax2.set_xlim(bounds[0], bounds[1])
            ax2.set_ylim(bounds[2], bounds[3])
            ax2.set_aspect('equal')
            ax2.set_title('Velocity Magnitude (m/s)')
            ax2.set_xlabel('X (m)')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='Velocity (m/s)')
            
            plt.tight_layout()
            plt.savefig(f'sph_dam_break_step_{step:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved snapshot at step {step}")
            
            # Also print particle stats
            print(f"  Particles visible: {len(positions)}")
            print(f"  Position range X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
            print(f"  Position range Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
            print(f"  Density range: [{densities.min():.0f}, {densities.max():.0f}] kg/m³")
            print(f"  Velocity range: [{velocities.min():.2f}, {velocities.max():.2f}] m/s")

if __name__ == "__main__":
    print("SPH Dam Break - Snapshot Version")
    print("================================")
    print()
    
    simulate_and_snapshot()
    
    print("\nDone! Check the PNG files for visualization.")