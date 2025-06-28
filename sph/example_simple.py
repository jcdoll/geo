#!/usr/bin/env python3
"""
Simple SPH example demonstrating basic concepts.

This is a minimal implementation to show how SPH handles particle
interactions without the complexity of grid-based pressure projection.
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
    
    def __init__(self, h: float = 1.0):
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
        for j in neighbors[i]:
            p_j = particles[j]
            r = np.linalg.norm(p_i.position - p_j.position)
            density += p_j.mass * kernel.W(r)
        
        # Self contribution
        density += p_i.mass * kernel.W(0)
        p_i.density = density

def tait_eos(density: float, rho0: float = 1000.0, B: float = 1e6, gamma: float = 7) -> float:
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
            pressure_term = p_i.pressure / (p_i.density**2) + p_j.pressure / (p_j.density**2)
            
            # Artificial viscosity for stability
            v_ij = p_i.velocity - p_j.velocity
            r_norm = np.linalg.norm(r_ij)
            if r_norm > 1e-6:
                visc = 0.0
                if np.dot(v_ij, r_ij) < 0:  # Approaching
                    alpha = 0.1
                    visc = -alpha * kernel.h * np.dot(v_ij, r_ij) / (r_norm**2 + 0.01*kernel.h**2)
                
                force = -p_j.mass * (pressure_term + visc) * kernel.gradW(r_ij)
                forces[i] += force
    
    # Add gravity
    g = np.array([0, -9.81])
    for i, p in enumerate(particles):
        forces[i] += p.mass * g
    
    return forces

def integrate(particles: List[Particle], forces: List[np.ndarray], dt: float):
    """Simple Euler integration."""
    for i, p in enumerate(particles):
        # Update velocity
        p.velocity += forces[i] / p.mass * dt
        
        # Update position
        p.position += p.velocity * dt
        
        # Boundary conditions (simple reflection)
        if p.position[1] < 0:
            p.position[1] = 0
            p.velocity[1] = -0.5 * p.velocity[1]  # Some damping

def create_dam_break(nx: int = 10, ny: int = 20, spacing: float = 0.5) -> List[Particle]:
    """Create initial dam break configuration."""
    particles = []
    
    for i in range(nx):
        for j in range(ny):
            pos = np.array([i * spacing, j * spacing + 1.0])
            vel = np.zeros(2)
            
            p = Particle(
                position=pos,
                velocity=vel,
                mass=0.1,  # kg
                material="water"
            )
            particles.append(p)
    
    return particles

def simulate():
    """Run simple SPH simulation."""
    # Parameters
    kernel = SimpleKernel(h=1.0)
    dt = 0.001
    steps = 1000
    
    # Create particles
    particles = create_dam_break()
    print(f"Simulating {len(particles)} particles...")
    
    # Visualization setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for step in range(steps):
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
        integrate(particles, forces, dt)
        
        # Visualize every 10 steps
        if step % 10 == 0:
            ax.clear()
            
            # Extract positions and color by density
            positions = np.array([p.position for p in particles])
            densities = np.array([p.density for p in particles])
            
            scatter = ax.scatter(
                positions[:, 0], positions[:, 1],
                c=densities, cmap='Blues',
                s=50, alpha=0.8,
                vmin=900, vmax=1100
            )
            
            ax.set_xlim(-1, 10)
            ax.set_ylim(-1, 10)
            ax.set_aspect('equal')
            ax.set_title(f'SPH Dam Break - Step {step}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            
            # Add colorbar on first frame
            if step == 0:
                plt.colorbar(scatter, ax=ax, label='Density (kg/m³)')
            
            plt.pause(0.01)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    print("Simple SPH Dam Break Example")
    print("============================")
    print("This demonstrates:")
    print("- No grid needed")
    print("- Natural free surface")
    print("- Stable with large density variations")
    print("- Simple physics implementation")
    print()
    
    simulate()