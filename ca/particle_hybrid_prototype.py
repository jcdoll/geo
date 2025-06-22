"""Prototype of hybrid particle-grid simulation for geological physics.

Particles handle:
- Position, velocity, mass
- Material type
- Pressure (from local density)

Grid handles:
- Temperature field (fast diffusion)
- Density field (for visualization)
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from numba import jit, prange


class ParticleHybridSim:
    def __init__(self, width=128, height=128, n_particles=10000, cell_size=50.0):
        """Initialize hybrid particle-grid simulation."""
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.n_particles = n_particles
        
        # Grid properties (just temperature and density for coupling)
        self.grid_temp = np.full((height, width), 293.15)  # K
        self.grid_density = np.zeros((height, width))
        self.grid_material_count = np.zeros((height, width), dtype=np.int32)
        
        # Particle properties
        self.pos = np.zeros((n_particles, 2))  # x, y position
        self.vel = np.zeros((n_particles, 2))  # velocity
        self.mass = np.ones(n_particles) * 100.0  # kg per particle
        self.temp = np.full(n_particles, 293.15)  # K
        self.material = np.zeros(n_particles, dtype=np.int32)  # 0=air, 1=water, 2=rock
        self.density = np.zeros(n_particles)  # local density from SPH
        
        # SPH parameters
        self.h = cell_size * 1.5  # smoothing radius
        self.k_pressure = 1000.0  # pressure constant
        self.rest_density = {0: 1.2, 1: 1000.0, 2: 2700.0}  # kg/m³
        
        # Physics parameters
        self.gravity = np.array([0.0, 9.81])  # m/s²
        self.thermal_diffusivity = 1e-6  # m²/s (rock-like)
        
        # Initialize particle positions
        self._initialize_particles()
        
    def _initialize_particles(self):
        """Set up initial particle distribution."""
        # Create a water blob surrounded by air
        particles_per_type = {
            0: self.n_particles // 2,  # air
            1: self.n_particles // 3,  # water
            2: self.n_particles - (self.n_particles // 2 + self.n_particles // 3)  # rock
        }
        
        idx = 0
        # Air particles - top half
        for i in range(particles_per_type[0]):
            self.pos[idx] = [
                np.random.uniform(0, self.width * self.cell_size),
                np.random.uniform(0, self.height * self.cell_size / 2)
            ]
            self.material[idx] = 0
            self.mass[idx] = 1.0  # lighter
            idx += 1
            
        # Water particles - middle blob
        center = np.array([self.width * self.cell_size / 2, self.height * self.cell_size / 2])
        radius = self.width * self.cell_size / 4
        for i in range(particles_per_type[1]):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            self.pos[idx] = center + r * np.array([np.cos(angle), np.sin(angle)])
            self.material[idx] = 1
            self.mass[idx] = 100.0
            self.temp[idx] = 300.0  # slightly warm
            idx += 1
            
        # Rock particles - bottom
        for i in range(particles_per_type[2]):
            self.pos[idx] = [
                np.random.uniform(0, self.width * self.cell_size),
                np.random.uniform(self.height * self.cell_size * 0.7, self.height * self.cell_size)
            ]
            self.material[idx] = 2
            self.mass[idx] = 200.0
            self.temp[idx] = 350.0  # hot rock
            idx += 1
    
    @staticmethod
    @jit(nopython=True)
    def _compute_density_pressure(pos, mass, material, h, n_particles, rest_density, k_pressure):
        """Compute density and pressure for all particles using SPH."""
        density = np.zeros(n_particles)
        pressure = np.zeros(n_particles)
        
        # Simple cubic spline kernel
        def kernel(r, h):
            q = r / h
            if q <= 1.0:
                return (2 - q)**3 - 4 * (1 - q)**3 if q > 0.5 else 1.0
            return 0.0
        
        # Compute density
        for i in prange(n_particles):
            rho = 0.0
            for j in range(n_particles):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                r = np.sqrt(dx*dx + dy*dy)
                if r < h:
                    rho += mass[j] * kernel(r, h)
            density[i] = rho
            
            # Pressure from equation of state
            mat = material[i]
            rho_0 = rest_density[mat] if mat < len(rest_density) else 1000.0
            pressure[i] = k_pressure * (density[i] - rho_0)
        
        return density, pressure
    
    def compute_forces(self):
        """Compute all forces on particles - FAST version."""
        forces = np.zeros_like(self.vel)
        
        # Skip expensive SPH density calculation
        # Instead, use grid density as approximation
        t0 = time.perf_counter()
        
        # Sample density from grid for each particle
        for i in range(self.n_particles):
            gx = int(self.pos[i, 0] / self.cell_size)
            gy = int(self.pos[i, 1] / self.cell_size)
            
            if 0 <= gx < self.width and 0 <= gy < self.height:
                # Use grid density as local density estimate
                self.density[i] = max(self.grid_density[gy, gx], self.rest_density[self.material[i]])
            else:
                self.density[i] = self.rest_density[self.material[i]]
        
        t_density = time.perf_counter() - t0
        
        # Compute pressure from density
        pressure = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            mat = self.material[i]
            rho_0 = self.rest_density[mat]
            pressure[i] = self.k_pressure * (self.density[i] - rho_0)
        
        # Simple pressure gradient from grid
        t0 = time.perf_counter()
        for i in range(self.n_particles):
            gx = int(self.pos[i, 0] / self.cell_size)
            gy = int(self.pos[i, 1] / self.cell_size)
            
            if 1 <= gx < self.width-1 and 1 <= gy < self.height-1:
                # Estimate pressure gradient from neighboring cells
                p_right = self.k_pressure * (self.grid_density[gy, gx+1] - self.rest_density[self.material[i]])
                p_left = self.k_pressure * (self.grid_density[gy, gx-1] - self.rest_density[self.material[i]])
                p_up = self.k_pressure * (self.grid_density[gy-1, gx] - self.rest_density[self.material[i]])
                p_down = self.k_pressure * (self.grid_density[gy+1, gx] - self.rest_density[self.material[i]])
                
                # Pressure gradient force
                forces[i, 0] = -(p_right - p_left) / (2 * self.cell_size) * self.mass[i] / self.density[i]
                forces[i, 1] = -(p_down - p_up) / (2 * self.cell_size) * self.mass[i] / self.density[i]
        
        t_pressure = time.perf_counter() - t0
        
        # Gravity
        for i in range(self.n_particles):
            forces[i] += self.mass[i] * self.gravity
        
        return forces, t_density, t_pressure
    
    def particles_to_grid(self):
        """Transfer particle properties to grid."""
        # Reset grid
        self.grid_density.fill(0)
        self.grid_temp.fill(0)
        self.grid_material_count.fill(0)
        
        # Accumulate particle contributions
        for i in range(self.n_particles):
            # Find grid cell
            gx = int(self.pos[i, 0] / self.cell_size)
            gy = int(self.pos[i, 1] / self.cell_size)
            
            if 0 <= gx < self.width and 0 <= gy < self.height:
                self.grid_density[gy, gx] += self.mass[i]
                self.grid_temp[gy, gx] += self.temp[i] * self.mass[i]
                self.grid_material_count[gy, gx] += 1
        
        # Normalize temperature by mass
        mask = self.grid_density > 0
        self.grid_temp[mask] /= self.grid_density[mask]
        self.grid_temp[~mask] = 293.15  # ambient temperature
        
        # Convert density to kg/m³
        cell_volume = self.cell_size ** 2
        self.grid_density /= cell_volume
    
    def grid_to_particles(self):
        """Sample grid temperature back to particles."""
        for i in range(self.n_particles):
            # Find grid cell
            gx = int(self.pos[i, 0] / self.cell_size)
            gy = int(self.pos[i, 1] / self.cell_size)
            
            if 0 <= gx < self.width and 0 <= gy < self.height:
                # Simple nearest neighbor sampling (could do bilinear)
                self.temp[i] = self.grid_temp[gy, gx]
    
    def solve_heat_diffusion(self, dt):
        """Solve heat equation on grid using simple explicit method."""
        # Thermal diffusion: ∂T/∂t = α∇²T
        # For speed, use simple explicit update (stable for small dt)
        
        dx2 = self.cell_size ** 2
        alpha_dt_dx2 = self.thermal_diffusivity * dt / dx2
        
        # Ensure stability
        if alpha_dt_dx2 > 0.25:
            alpha_dt_dx2 = 0.25
            
        # Compute Laplacian and update
        laplacian = laplace(self.grid_temp) / dx2
        self.grid_temp += alpha_dt_dx2 * laplacian * dx2
    
    def update_particles(self, forces, dt):
        """Update particle velocities and positions."""
        # Update velocities
        for i in range(self.n_particles):
            if self.mass[i] > 0:
                self.vel[i] += forces[i] / self.mass[i] * dt
                
                # Simple damping based on material
                damping = {0: 0.99, 1: 0.98, 2: 0.95}[self.material[i]]
                self.vel[i] *= damping
        
        # Update positions
        self.pos += self.vel * dt
        
        # Boundary conditions (simple bounce)
        for i in range(self.n_particles):
            if self.pos[i, 0] < 0:
                self.pos[i, 0] = 0
                self.vel[i, 0] *= -0.5
            elif self.pos[i, 0] > self.width * self.cell_size:
                self.pos[i, 0] = self.width * self.cell_size
                self.vel[i, 0] *= -0.5
                
            if self.pos[i, 1] < 0:
                self.pos[i, 1] = 0
                self.vel[i, 1] *= -0.5
            elif self.pos[i, 1] > self.height * self.cell_size:
                self.pos[i, 1] = self.height * self.cell_size
                self.vel[i, 1] *= -0.5
    
    def step(self, dt=0.01):
        """Perform one simulation timestep."""
        times = {}
        
        # 1. Compute forces (pressure + gravity)
        t0 = time.perf_counter()
        forces, t_density, t_pressure = self.compute_forces()
        times['forces'] = time.perf_counter() - t0
        times['density'] = t_density
        times['pressure'] = t_pressure
        
        # 2. Update particles
        t0 = time.perf_counter()
        self.update_particles(forces, dt)
        times['particles'] = time.perf_counter() - t0
        
        # 3. Transfer to grid
        t0 = time.perf_counter()
        self.particles_to_grid()
        times['to_grid'] = time.perf_counter() - t0
        
        # 4. Solve heat diffusion
        t0 = time.perf_counter()
        self.solve_heat_diffusion(dt)
        times['heat'] = time.perf_counter() - t0
        
        # 5. Transfer back to particles
        t0 = time.perf_counter()
        self.grid_to_particles()
        times['from_grid'] = time.perf_counter() - t0
        
        times['total'] = sum(times.values()) - times['density'] - times['pressure']
        
        return times
    
    def visualize(self):
        """Create visualization of current state."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Particle plot
        colors = ['cyan', 'blue', 'brown']
        for mat in range(3):
            mask = self.material == mat
            ax1.scatter(self.pos[mask, 0], self.pos[mask, 1], 
                       c=colors[mat], s=5, alpha=0.6, 
                       label=['Air', 'Water', 'Rock'][mat])
        ax1.set_xlim(0, self.width * self.cell_size)
        ax1.set_ylim(0, self.height * self.cell_size)
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.set_title('Particles')
        
        # Temperature grid
        im = ax2.imshow(self.grid_temp, origin='lower', cmap='hot',
                       extent=[0, self.width * self.cell_size, 
                               0, self.height * self.cell_size])
        plt.colorbar(im, ax=ax2, label='Temperature (K)')
        ax2.set_title('Grid Temperature')
        
        plt.tight_layout()
        return fig


def test_performance():
    """Test the performance of the hybrid simulation."""
    print("HYBRID PARTICLE-GRID SIMULATION TEST")
    print("=" * 60)
    
    # Create simulation - use fewer particles for testing
    sim = ParticleHybridSim(width=64, height=64, n_particles=5000)
    
    print(f"\nSetup:")
    print(f"  Grid: {sim.width}×{sim.height}")
    print(f"  Particles: {sim.n_particles}")
    print(f"  Cell size: {sim.cell_size}m")
    
    # Warm up
    print("\nWarming up...")
    for _ in range(5):
        sim.step()
    
    # Time multiple steps
    print("\nTiming 10 steps...")
    all_times = []
    for i in range(10):
        times = sim.step()
        all_times.append(times)
        
        if i == 0:
            print(f"\nStep {i} breakdown:")
            print(f"  Density calc:  {times['density']*1000:6.2f} ms")
            print(f"  Pressure calc: {times['pressure']*1000:6.2f} ms")
            print(f"  Particles:     {times['particles']*1000:6.2f} ms")
            print(f"  To grid:       {times['to_grid']*1000:6.2f} ms")
            print(f"  Heat diffusion:{times['heat']*1000:6.2f} ms")
            print(f"  From grid:     {times['from_grid']*1000:6.2f} ms")
            print(f"  TOTAL:         {times['total']*1000:6.2f} ms")
    
    # Average times
    avg_times = {}
    for key in all_times[0]:
        avg_times[key] = np.mean([t[key] for t in all_times])
    
    print(f"\nAverage timing over 10 steps:")
    print(f"  Density calc:  {avg_times['density']*1000:6.2f} ms")
    print(f"  Pressure calc: {avg_times['pressure']*1000:6.2f} ms") 
    print(f"  Particles:     {avg_times['particles']*1000:6.2f} ms")
    print(f"  To grid:       {avg_times['to_grid']*1000:6.2f} ms")
    print(f"  Heat diffusion:{avg_times['heat']*1000:6.2f} ms")
    print(f"  From grid:     {avg_times['from_grid']*1000:6.2f} ms")
    print(f"  TOTAL:         {avg_times['total']*1000:6.2f} ms")
    print(f"\nFPS: {1.0/avg_times['total']:.1f}")
    
    # Check if pressure emerges correctly
    print(f"\nPressure check:")
    print(f"  Max particle density: {np.max(sim.density):.1f} kg/m³")
    print(f"  Max grid density: {np.max(sim.grid_density):.1f} kg/m³")
    
    # Visualize
    print("\nCreating visualization...")
    fig = sim.visualize()
    plt.savefig('particle_hybrid_test.png')
    print("Saved to particle_hybrid_test.png")
    
    return sim


if __name__ == "__main__":
    sim = test_performance()