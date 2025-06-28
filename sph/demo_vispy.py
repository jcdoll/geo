#!/usr/bin/env python3
"""
Demo of fast GPU-accelerated SPH visualization using Vispy.

This demonstrates real-time rendering of large particle systems
with interactive controls and multiple visualization modes.
"""

import numpy as np
import time
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.spatial_hash_vectorized import VectorizedSpatialHash
from sph.physics.density_vectorized import compute_density_vectorized
from sph.physics.forces_vectorized import compute_forces_vectorized, tait_equation_of_state
from sph.core.integrator_vectorized import integrate_leapfrog_vectorized


def create_multi_material_scene(domain_size: tuple) -> tuple:
    """Create a scene with multiple materials."""
    particles = ParticleArrays.allocate(50000)
    idx = 0
    
    # Water blob (top left)
    water_center = (2.0, 4.0)
    water_radius = 0.8
    spacing = 0.04
    
    for i in range(-20, 21):
        for j in range(-20, 21):
            x = water_center[0] + i * spacing
            y = water_center[1] + j * spacing
            if (x - water_center[0])**2 + (y - water_center[1])**2 <= water_radius**2:
                particles.position_x[idx] = x
                particles.position_y[idx] = y
                particles.velocity_x[idx] = 0.0
                particles.velocity_y[idx] = 0.0
                particles.mass[idx] = 1.0 * spacing**2
                particles.density[idx] = 1000.0
                particles.smoothing_h[idx] = 1.3 * spacing
                particles.material_id[idx] = 0  # Water
                particles.temperature[idx] = 293.15  # 20°C
                idx += 1
    
    # Rock wall (bottom)
    for i in range(int(domain_size[0] / spacing)):
        for j in range(5):
            particles.position_x[idx] = i * spacing
            particles.position_y[idx] = j * spacing
            particles.velocity_x[idx] = 0.0
            particles.velocity_y[idx] = 0.0
            particles.mass[idx] = 2.5 * spacing**2
            particles.density[idx] = 2500.0
            particles.smoothing_h[idx] = 1.3 * spacing
            particles.material_id[idx] = 1  # Rock
            particles.temperature[idx] = 293.15
            idx += 1
    
    # Hot magma blob (right side)
    magma_center = (7.0, 3.0)
    magma_radius = 0.6
    
    for i in range(-15, 16):
        for j in range(-15, 16):
            x = magma_center[0] + i * spacing
            y = magma_center[1] + j * spacing
            if (x - magma_center[0])**2 + (y - magma_center[1])**2 <= magma_radius**2:
                particles.position_x[idx] = x
                particles.position_y[idx] = y
                particles.velocity_x[idx] = 0.0
                particles.velocity_y[idx] = -1.0  # Initial downward velocity
                particles.mass[idx] = 3.0 * spacing**2
                particles.density[idx] = 3000.0
                particles.smoothing_h[idx] = 1.3 * spacing
                particles.material_id[idx] = 2  # Magma
                particles.temperature[idx] = 1500.0  # Very hot!
                idx += 1
    
    return particles, idx


def run_vispy_demo():
    """Run the Vispy-based visualization demo."""
    try:
        from sph.visualization.vispy_renderer import SPHRenderer
    except ImportError:
        print("Vispy not installed. Install with: pip install vispy")
        return
    
    # Setup
    domain_size = (10.0, 6.0)
    particles, n_active = create_multi_material_scene(domain_size)
    
    print(f"Created scene with {n_active} particles")
    print("Materials: Water (blue), Rock (gray), Magma (red)")
    
    # Physics modules
    kernel = CubicSplineKernel(dim=2)
    spatial_hash = VectorizedSpatialHash(domain_size, 0.08)
    
    # Material properties
    rho0 = np.zeros(particles.mass.shape[0], dtype=np.float32)
    B = np.zeros(particles.mass.shape[0], dtype=np.float32)
    
    # Water
    mask = particles.material_id == 0
    rho0[mask] = 1000.0
    B[mask] = 2.2e5
    
    # Rock
    mask = particles.material_id == 1
    rho0[mask] = 2500.0
    B[mask] = 1e6
    
    # Magma
    mask = particles.material_id == 2
    rho0[mask] = 3000.0
    B[mask] = 5e5
    
    # Create renderer
    renderer = SPHRenderer(
        domain_size=domain_size,
        max_particles=100000,
        title='SPH Multi-Material Demo (Vispy)'
    )
    
    # Simulation state
    sim_time = 0.0
    dt = 5e-5
    step_count = 0
    
    def update_simulation(event):
        nonlocal sim_time, step_count
        
        # Physics update (multiple substeps for stability)
        substeps = 5
        for _ in range(substeps):
            # Spatial hash
            spatial_hash.build_vectorized(particles, n_active)
            spatial_hash.query_neighbors_vectorized(particles, n_active, 0.1)
            
            # Density
            compute_density_vectorized(particles, kernel, n_active)
            
            # Pressure
            particles.pressure[:n_active] = tait_equation_of_state(
                particles.density[:n_active],
                rho0[:n_active],
                B[:n_active]
            )
            
            # Forces
            compute_forces_vectorized(particles, kernel, n_active,
                                    gravity=np.array([0.0, -9.81]))
            
            # Integration
            integrate_leapfrog_vectorized(particles, n_active, dt,
                                        (0, domain_size[0], 0, domain_size[1]),
                                        damping=0.95)
            
            sim_time += dt
            step_count += 1
        
        # Update visualization
        renderer.update_particles(particles, n_active, sim_time)
        
        # Print stats occasionally
        if step_count % 100 == 0:
            max_v = np.max(np.sqrt(
                particles.velocity_x[:n_active]**2 + 
                particles.velocity_y[:n_active]**2
            ))
            print(f"Step {step_count}, Time {sim_time:.3f}s, Max velocity {max_v:.2f} m/s")
    
    # Run
    print("\nControls:")
    print("  D - Color by Density")
    print("  V - Color by Velocity") 
    print("  T - Color by Temperature")
    print("  P - Color by Pressure")
    print("  M - Color by Material")
    print("  Space - Pause/Resume")
    print("  R - Reset camera")
    print("  Mouse - Pan/Zoom")
    print("  ESC - Quit")
    
    renderer.run(update_simulation)


def benchmark_vispy_performance():
    """Benchmark rendering performance with different particle counts."""
    try:
        from sph.visualization.vispy_renderer import SPHRenderer
        from vispy import app
    except ImportError:
        print("Vispy not installed. Install with: pip install vispy")
        return
    
    print("\n=== Vispy Performance Benchmark ===")
    
    particle_counts = [1000, 5000, 10000, 50000, 100000]
    
    for n_particles in particle_counts:
        # Create random particles
        particles = ParticleArrays.allocate(n_particles)
        particles.position_x[:n_particles] = np.random.uniform(0, 10, n_particles)
        particles.position_y[:n_particles] = np.random.uniform(0, 6, n_particles)
        particles.velocity_x[:n_particles] = np.random.normal(0, 0.5, n_particles)
        particles.velocity_y[:n_particles] = np.random.normal(0, 0.5, n_particles)
        particles.density[:n_particles] = np.random.uniform(900, 1100, n_particles)
        particles.temperature[:n_particles] = np.random.uniform(273, 373, n_particles)
        particles.smoothing_h[:n_particles] = 0.05
        particles.material_id[:n_particles] = np.random.randint(0, 3, n_particles)
        
        # Create renderer
        renderer = SPHRenderer(title=f'Benchmark {n_particles} particles')
        
        # Measure frame time
        frame_times = []
        frame_count = 0
        
        def benchmark_frame(event):
            nonlocal frame_count
            t0 = time.perf_counter()
            
            # Simulate particle movement
            particles.position_x[:n_particles] += 0.01 * np.sin(frame_count * 0.1)
            particles.position_y[:n_particles] += 0.01 * np.cos(frame_count * 0.1)
            
            # Update renderer
            renderer.update_particles(particles, n_particles, frame_count * 0.016)
            
            frame_times.append(time.perf_counter() - t0)
            frame_count += 1
            
            if frame_count >= 100:
                avg_time = np.mean(frame_times[10:])  # Skip first frames
                fps = 1.0 / avg_time
                print(f"{n_particles:7d} particles: {fps:6.1f} FPS ({avg_time*1000:.2f} ms/frame)")
                renderer.canvas.close()
                app.quit()
        
        # Run benchmark
        timer = app.Timer(interval=0.0, connect=benchmark_frame, start=True)
        app.run()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_vispy_performance()
    else:
        run_vispy_demo()