#!/usr/bin/env python3
"""
Fixed simple SPH example demonstrating basic concepts.

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

def simulate():
    """Run simple SPH simulation."""
    # Parameters
    kernel = SimpleKernel(h=0.04)
    dt = 0.0001
    steps = 2000
    bounds = (0.0, 1.0, 0.0, 0.6)  # Simulation domain
    
    # Create particles
    particles = create_dam_break()
    print(f"Simulating {len(particles)} particles...")
    
    # Visualization setup
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Storage for monitoring
    max_velocity = []
    avg_density = []
    
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
        integrate(particles, forces, dt, bounds)
        
        # Monitor
        velocities = [np.linalg.norm(p.velocity) for p in particles]
        densities = [p.density for p in particles]
        max_velocity.append(max(velocities))
        avg_density.append(np.mean(densities))
        
        # Visualize every 20 steps
        if step % 20 == 0:
            ax1.clear()
            ax2.clear()
            
            # Extract positions and properties
            positions = np.array([p.position for p in particles])
            densities = np.array([p.density for p in particles])
            velocities = np.array([np.linalg.norm(p.velocity) for p in particles])
            
            # Particle positions colored by density
            scatter1 = ax1.scatter(
                positions[:, 0], positions[:, 1],
                c=densities, cmap='Blues',
                s=30, alpha=0.8,
                vmin=800, vmax=1200
            )
            
            ax1.set_xlim(bounds[0], bounds[1])
            ax1.set_ylim(bounds[2], bounds[3])
            ax1.set_aspect('equal')
            ax1.set_title(f'SPH Dam Break - Step {step} (t={step*dt:.3f}s)')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.grid(True, alpha=0.3)
            
            # Velocity field
            scatter2 = ax2.scatter(
                positions[:, 0], positions[:, 1],
                c=velocities, cmap='viridis',
                s=30, alpha=0.8,
                vmin=0, vmax=2.0
            )
            
            ax2.set_xlim(bounds[0], bounds[1])
            ax2.set_ylim(bounds[2], bounds[3])
            ax2.set_aspect('equal')
            ax2.set_title('Velocity Magnitude (m/s)')
            ax2.set_xlabel('X (m)')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbars on first frame
            if step == 0:
                plt.colorbar(scatter1, ax=ax1, label='Density (kg/m³)')
                plt.colorbar(scatter2, ax=ax2, label='Velocity (m/s)')
            
            plt.tight_layout()
            plt.pause(0.01)
        
        # Progress
        if step % 100 == 0:
            print(f"Step {step}/{steps}, Max velocity: {max(velocities):.2f} m/s, "
                  f"Avg density: {np.mean(densities):.0f} kg/m³")
    
    plt.ioff()
    
    # Show final diagnostics
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 6))
    
    steps_array = np.arange(len(max_velocity)) * dt
    ax3.plot(steps_array, max_velocity)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Max Velocity (m/s)')
    ax3.set_title('Maximum Velocity vs Time')
    ax3.grid(True)
    
    ax4.plot(steps_array, avg_density)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Average Density (kg/m³)')
    ax4.set_title('Average Density vs Time')
    ax4.grid(True)
    ax4.set_ylim(900, 1100)
    
    plt.tight_layout()
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