"""Optimized particle-grid hybrid for <10ms performance.

Key optimizations:
1. Vectorized operations wherever possible
2. Grid-based pressure (no particle-particle interactions)
3. Simplified heat diffusion
4. Numba acceleration for critical loops
"""

import numpy as np
import time
from numba import jit, njit, prange


@njit(parallel=True)
def particles_to_grid_fast(pos, mass, temp, material, grid_density, grid_temp, grid_count, 
                           n_particles, width, height, cell_size):
    """Fast particle to grid transfer using numba."""
    # Reset grids
    grid_density.fill(0.0)
    grid_temp.fill(0.0)
    grid_count.fill(0)
    
    # Accumulate particles
    for i in prange(n_particles):
        gx = int(pos[i, 0] / cell_size)
        gy = int(pos[i, 1] / cell_size)
        
        if 0 <= gx < width and 0 <= gy < height:
            grid_density[gy, gx] += mass[i]
            grid_temp[gy, gx] += temp[i] * mass[i]
            grid_count[gy, gx] += 1
    
    # Normalize temperature
    for y in prange(height):
        for x in range(width):
            if grid_density[y, x] > 0:
                grid_temp[y, x] /= grid_density[y, x]
            else:
                grid_temp[y, x] = 293.15
                
    # Convert to density
    cell_volume = cell_size * cell_size
    grid_density /= cell_volume


@njit(parallel=True)
def compute_forces_from_grid(pos, mass, material, grid_pressure, 
                             n_particles, width, height, cell_size, gravity_y):
    """Compute forces using grid-based pressure."""
    forces = np.zeros((n_particles, 2))
    
    for i in prange(n_particles):
        gx = int(pos[i, 0] / cell_size)
        gy = int(pos[i, 1] / cell_size)
        
        # Gravity
        forces[i, 1] = mass[i] * gravity_y
        
        # Pressure gradient from grid
        if 1 <= gx < width-1 and 1 <= gy < height-1:
            # Central differences
            dp_dx = (grid_pressure[gy, gx+1] - grid_pressure[gy, gx-1]) / (2 * cell_size)
            dp_dy = (grid_pressure[gy+1, gx] - grid_pressure[gy-1, gx]) / (2 * cell_size)
            
            # Force = -∇P * Volume
            volume = mass[i] / 1000.0  # rough estimate
            forces[i, 0] -= dp_dx * volume
            forces[i, 1] -= dp_dy * volume
    
    return forces


@njit(parallel=True) 
def update_particles_fast(pos, vel, forces, mass, material, dt, n_particles, max_x, max_y):
    """Fast particle update with boundary conditions."""
    damping = np.array([0.99, 0.98, 0.95])  # air, water, rock
    
    for i in prange(n_particles):
        # Update velocity
        if mass[i] > 0:
            vel[i, 0] += forces[i, 0] / mass[i] * dt
            vel[i, 1] += forces[i, 1] / mass[i] * dt
            
            # Material damping
            damp = damping[material[i]]
            vel[i, 0] *= damp
            vel[i, 1] *= damp
        
        # Update position
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        
        # Boundaries
        if pos[i, 0] < 0:
            pos[i, 0] = 0
            vel[i, 0] *= -0.5
        elif pos[i, 0] > max_x:
            pos[i, 0] = max_x
            vel[i, 0] *= -0.5
            
        if pos[i, 1] < 0:
            pos[i, 1] = 0
            vel[i, 1] *= -0.5
        elif pos[i, 1] > max_y:
            pos[i, 1] = max_y
            vel[i, 1] *= -0.5


class FastParticleHybrid:
    """Optimized particle-grid hybrid simulation."""
    
    def __init__(self, width=128, height=128, n_particles=10000, cell_size=50.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.n_particles = n_particles
        
        # Grids
        self.grid_density = np.zeros((height, width), dtype=np.float32)
        self.grid_temp = np.full((height, width), 293.15, dtype=np.float32)
        self.grid_pressure = np.zeros((height, width), dtype=np.float32)
        self.grid_count = np.zeros((height, width), dtype=np.int32)
        
        # Particles  
        self.pos = np.zeros((n_particles, 2), dtype=np.float32)
        self.vel = np.zeros((n_particles, 2), dtype=np.float32)
        self.mass = np.ones(n_particles, dtype=np.float32) * 100.0
        self.temp = np.full(n_particles, 293.15, dtype=np.float32)
        self.material = np.zeros(n_particles, dtype=np.int32)
        
        # Constants
        self.gravity_y = 9.81
        self.k_pressure = 1000.0
        self.rest_density = np.array([1.2, 1000.0, 2700.0], dtype=np.float32)
        
        self._initialize_particles()
        
    def _initialize_particles(self):
        """Simple initialization for testing."""
        # Random distribution by material type
        n_air = self.n_particles // 3
        n_water = self.n_particles // 3
        n_rock = self.n_particles - n_air - n_water
        
        idx = 0
        domain_x = self.width * self.cell_size
        domain_y = self.height * self.cell_size
        
        # Air (top)
        for i in range(n_air):
            self.pos[idx] = [np.random.uniform(0, domain_x), 
                            np.random.uniform(0, domain_y * 0.4)]
            self.material[idx] = 0
            self.mass[idx] = 1.0
            idx += 1
            
        # Water (middle)
        for i in range(n_water):
            self.pos[idx] = [np.random.uniform(0, domain_x),
                            np.random.uniform(domain_y * 0.4, domain_y * 0.7)]
            self.material[idx] = 1
            self.mass[idx] = 100.0
            idx += 1
            
        # Rock (bottom)
        for i in range(n_rock):
            self.pos[idx] = [np.random.uniform(0, domain_x),
                            np.random.uniform(domain_y * 0.7, domain_y)]
            self.material[idx] = 2
            self.mass[idx] = 200.0
            idx += 1
    
    def compute_grid_pressure(self):
        """Simple pressure calculation from density."""
        # P = k * (ρ - ρ₀)
        # Use average rest density as reference
        rho_0 = 1000.0  # water density as reference
        self.grid_pressure = self.k_pressure * (self.grid_density - rho_0)
        
        # Clamp negative pressures
        self.grid_pressure = np.maximum(self.grid_pressure, 0.0)
    
    def simple_heat_diffusion(self, dt):
        """Simplified explicit heat diffusion."""
        # Only do a single iteration for speed
        alpha = 1e-6  # thermal diffusivity
        dx2 = self.cell_size ** 2
        
        # Stability limit
        max_dt = 0.25 * dx2 / alpha
        dt_heat = min(dt, max_dt)
        
        # Simple 5-point stencil
        temp_new = self.grid_temp.copy()
        temp_new[1:-1, 1:-1] = (
            self.grid_temp[1:-1, 1:-1] + 
            dt_heat * alpha / dx2 * (
                self.grid_temp[:-2, 1:-1] + self.grid_temp[2:, 1:-1] +
                self.grid_temp[1:-1, :-2] + self.grid_temp[1:-1, 2:] -
                4 * self.grid_temp[1:-1, 1:-1]
            )
        )
        self.grid_temp = temp_new
    
    def step(self, dt=0.01):
        """Single simulation step."""
        times = {}
        
        # 1. Particles to grid (~3ms)
        t0 = time.perf_counter()
        particles_to_grid_fast(
            self.pos, self.mass, self.temp, self.material,
            self.grid_density, self.grid_temp, self.grid_count,
            self.n_particles, self.width, self.height, self.cell_size
        )
        times['to_grid'] = time.perf_counter() - t0
        
        # 2. Compute grid pressure (~0.1ms)
        t0 = time.perf_counter()
        self.compute_grid_pressure()
        times['pressure'] = time.perf_counter() - t0
        
        # 3. Compute forces from grid (~2ms)
        t0 = time.perf_counter()
        forces = compute_forces_from_grid(
            self.pos, self.mass, self.material, self.grid_pressure,
            self.n_particles, self.width, self.height, self.cell_size,
            self.gravity_y
        )
        times['forces'] = time.perf_counter() - t0
        
        # 4. Update particles (~2ms)
        t0 = time.perf_counter()
        update_particles_fast(
            self.pos, self.vel, forces, self.mass, self.material,
            dt, self.n_particles, 
            self.width * self.cell_size, self.height * self.cell_size
        )
        times['particles'] = time.perf_counter() - t0
        
        # 5. Heat diffusion (~0.5ms)
        t0 = time.perf_counter()
        self.simple_heat_diffusion(dt)
        times['heat'] = time.perf_counter() - t0
        
        # 6. Sample temperature back (~1ms)
        t0 = time.perf_counter()
        for i in range(self.n_particles):
            gx = int(self.pos[i, 0] / self.cell_size)
            gy = int(self.pos[i, 1] / self.cell_size)
            if 0 <= gx < self.width and 0 <= gy < self.height:
                self.temp[i] = self.grid_temp[gy, gx]
        times['from_grid'] = time.perf_counter() - t0
        
        times['total'] = sum(times.values())
        return times


def test_optimized_performance():
    """Test the optimized simulation."""
    print("OPTIMIZED PARTICLE-GRID SIMULATION")
    print("=" * 60)
    
    # Test with target size
    sim = FastParticleHybrid(width=128, height=128, n_particles=10000)
    
    print(f"\nSetup:")
    print(f"  Grid: {sim.width}×{sim.height}")  
    print(f"  Particles: {sim.n_particles}")
    print(f"  Cell size: {sim.cell_size}m")
    
    # Warm up JIT
    print("\nWarming up JIT...")
    for _ in range(3):
        sim.step()
    
    # Time steps
    print("\nTiming 20 steps...")
    all_times = []
    for i in range(20):
        times = sim.step()
        all_times.append(times)
        
        if i == 0:
            print(f"\nFirst step breakdown:")
            for key, val in times.items():
                if key != 'total':
                    print(f"  {key:12s}: {val*1000:6.2f} ms")
            print(f"  {'TOTAL':12s}: {times['total']*1000:6.2f} ms")
    
    # Average
    avg_times = {}
    for key in all_times[0]:
        avg_times[key] = np.mean([t[key] for t in all_times])
    
    print(f"\nAverage timing:")
    for key, val in avg_times.items():
        if key != 'total':
            print(f"  {key:12s}: {val*1000:6.2f} ms")
    print(f"  {'TOTAL':12s}: {avg_times['total']*1000:6.2f} ms")
    print(f"\nFPS: {1.0/avg_times['total']:.1f}")
    
    # Check physics
    print(f"\nPhysics check:")
    print(f"  Max velocity: {np.max(np.abs(sim.vel)):.2f} m/s")
    print(f"  Max pressure: {np.max(sim.grid_pressure):.0f} Pa")
    print(f"  Temp range: {np.min(sim.grid_temp):.1f} - {np.max(sim.grid_temp):.1f} K")


if __name__ == "__main__":
    test_optimized_performance()