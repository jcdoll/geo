#!/usr/bin/env python3
"""
Full planetary simulation demo with all physics enabled.

Demonstrates:
- Self-gravitating planet formation
- Material-aware physics
- Thermal evolution with heat transfer
- Phase transitions
- Layered planetary structure
"""

import numpy as np
import time
import sys
from sph.core import (
    ParticleArrays, CubicSplineKernel, VectorizedSpatialHash,
    integrate_leapfrog_vectorized, compute_adaptive_timestep
)
from sph.physics import (
    MaterialDatabase, MaterialType,
    compute_density_vectorized,
    compute_forces_vectorized,
    compute_gravity_direct_batched,
    update_temperature_full
)
from sph.scenarios import create_planet_earth_like, create_planet_simple
from sph.physics.gravity_vectorized import compute_center_of_mass


def material_aware_pressure(particles, material_db, n_active):
    """Compute pressure using material-specific EOS."""
    # Get material properties
    bulk_modulus = material_db.get_bulk_modulus_array(particles.material_id[:n_active])
    density_ref = material_db.get_density_ref_array(particles.material_id[:n_active])
    
    # Tait EOS with material properties
    gamma = 7.0
    particles.pressure[:n_active] = bulk_modulus * (
        (particles.density[:n_active] / density_ref)**gamma - 1
    )


def run_planet_simulation(scenario='simple', enable_viz=True):
    """Run full planet simulation with all physics."""
    
    print(f"\n=== SPH Planet Simulation ({scenario}) ===")
    
    # Create planet
    if scenario == 'earth':
        # Earth-like with layers (small scale)
        particles, n_active = create_planet_earth_like(
            radius_km=500,  # 500 km mini-Earth
            particle_spacing_km=10,  # 10 km spacing
            center=(5000, 5000)  # km
        )
        domain_size = (10000 * 1000, 10000 * 1000)  # meters
        G = 6.67430e-11  # Real gravity
        
    else:  # simple
        # Simple two-layer planet
        particles, n_active = create_planet_simple(
            radius=2000,  # 2 km radius
            particle_spacing=50,  # 50 m spacing
            center=(5000, 5000)  # m
        )
        domain_size = (10000, 10000)  # meters
        G = 6.67430e-11 * 1e6  # Scaled up for smaller system
    
    print(f"Created planet with {n_active} particles")
    
    # Initialize modules
    material_db = MaterialDatabase()
    kernel = CubicSplineKernel(dim=2)
    cell_size = np.mean(particles.smoothing_h[:n_active]) * 2
    spatial_hash = VectorizedSpatialHash(
        domain_size=(domain_size[0]/1000, domain_size[1]/1000),  # km for hash
        cell_size=cell_size/1000
    )
    
    # Count materials
    print("\nMaterial composition:")
    for mat_type in MaterialType:
        count = np.sum(particles.material_id[:n_active] == mat_type)
        if count > 0:
            mass = np.sum(particles.mass[particles.material_id[:n_active] == mat_type])
            print(f"  {material_db.get_properties(mat_type).name}: "
                  f"{count} particles, {mass/1e12:.1f} Mt")
    
    # Visualization setup
    if enable_viz:
        try:
            from sph.visualization.pygame_renderer import PygameRenderer
            renderer = PygameRenderer(
                domain_size=(domain_size[0]/1000, domain_size[1]/1000),  # km
                window_size=(800, 800),
                title=f"SPH Planet Simulation - {scenario}"
            )
            viz_available = True
        except ImportError:
            print("Pygame not available, running without visualization")
            viz_available = False
    else:
        viz_available = False
    
    # Simulation parameters
    sim_time = 0.0
    dt = 0.01 if scenario == 'simple' else 0.1
    max_steps = 10000
    step = 0
    
    # Statistics tracking
    stats_interval = 100
    energy_history = []
    
    # Main loop
    print(f"\nRunning simulation (dt={dt}s)...")
    t0 = time.time()
    
    while step < max_steps:
        # Build spatial hash (convert to km)
        particles_km = ParticleArrays.allocate(0)  # Dummy for interface
        particles_km.position_x = particles.position_x[:n_active] / 1000
        particles_km.position_y = particles.position_y[:n_active] / 1000
        particles_km.neighbor_ids = particles.neighbor_ids
        particles_km.neighbor_distances = particles.neighbor_distances
        particles_km.neighbor_count = particles.neighbor_count
        
        spatial_hash.build_vectorized(particles_km, n_active)
        spatial_hash.query_neighbors_vectorized(
            particles_km, n_active, cell_size/1000 * 2
        )
        
        # Copy neighbor info back and convert distances
        particles.neighbor_ids[:] = particles_km.neighbor_ids
        particles.neighbor_distances[:] = particles_km.neighbor_distances * 1000
        particles.neighbor_count[:] = particles_km.neighbor_count
        
        # Density
        compute_density_vectorized(particles, kernel, n_active)
        
        # Material-aware pressure
        material_aware_pressure(particles, material_db, n_active)
        
        # Forces (pressure + viscosity)
        compute_forces_vectorized(particles, kernel, n_active, gravity=None)
        
        # Self-gravity
        compute_gravity_direct_batched(particles, n_active, G=G, 
                                      softening=cell_size*0.5)
        
        # Thermal physics
        thermal_stats = update_temperature_full(
            particles, kernel, material_db, n_active, dt,
            enable_radiation=True,
            enable_transitions=True
        )
        
        # Integration
        integrate_leapfrog_vectorized(
            particles, n_active, dt,
            (0, domain_size[0], 0, domain_size[1]),
            damping=0.999  # Minimal damping
        )
        
        # Adaptive timestep
        if step % 50 == 0 and step > 0:
            dt_new = compute_adaptive_timestep(particles, n_active,
                                              cfl_factor=0.2)
            dt = np.clip(dt_new, dt*0.5, dt*2.0)
        
        # Statistics
        if step % stats_interval == 0:
            # Compute energies
            v_mag = np.sqrt(particles.velocity_x[:n_active]**2 + 
                           particles.velocity_y[:n_active]**2)
            kinetic_energy = 0.5 * np.sum(particles.mass[:n_active] * v_mag**2)
            
            # Center of mass
            cm_x, cm_y, total_mass = compute_center_of_mass(particles, n_active)
            
            print(f"\nStep {step}, t={sim_time:.1f}s, dt={dt:.3f}s")
            print(f"  Max velocity: {np.max(v_mag):.1f} m/s")
            print(f"  Temperature: {thermal_stats['min_temperature']:.0f} - "
                  f"{thermal_stats['max_temperature']:.0f} K")
            print(f"  Phase transitions: {thermal_stats['n_transitions']}")
            print(f"  Center of mass: ({cm_x/1000:.1f}, {cm_y/1000:.1f}) km")
            print(f"  Kinetic energy: {kinetic_energy/1e12:.3f} TJ")
            
            energy_history.append(kinetic_energy)
        
        # Visualization
        if viz_available and step % 10 == 0:
            # Convert to km for display
            particles_display = ParticleArrays.allocate(0)
            for attr in ['velocity_x', 'velocity_y', 'density', 'pressure',
                        'temperature', 'material_id']:
                setattr(particles_display, attr, getattr(particles, attr))
            particles_display.position_x = particles.position_x / 1000
            particles_display.position_y = particles.position_y / 1000
            
            renderer.update_particles(particles_display, n_active, sim_time)
            if not renderer.handle_events():
                break
        
        sim_time += dt
        step += 1
    
    # Final statistics
    elapsed = time.time() - t0
    print(f"\n=== Simulation Complete ===")
    print(f"Total time: {elapsed:.1f}s for {step} steps")
    print(f"Average: {elapsed/step*1000:.1f} ms/step, {step/elapsed:.1f} steps/s")
    print(f"Final simulation time: {sim_time:.1f}s")
    
    # Check stability
    if len(energy_history) > 2:
        energy_change = (energy_history[-1] - energy_history[0]) / energy_history[0]
        print(f"Energy change: {energy_change*100:.1f}%")
    
    if viz_available:
        renderer.close()


def run_benchmark():
    """Benchmark performance with different particle counts."""
    print("\n=== SPH Planet Benchmark ===")
    
    radii = [500, 1000, 2000]
    spacings = [100, 50, 25]
    
    for radius, spacing in zip(radii, spacings):
        particles, n_active = create_planet_simple(radius, spacing)
        
        # Initialize
        material_db = MaterialDatabase()
        kernel = CubicSplineKernel(dim=2)
        
        # Time one physics step
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            
            # Density
            compute_density_vectorized(particles, kernel, n_active)
            
            # Pressure
            material_aware_pressure(particles, material_db, n_active)
            
            # Forces
            compute_forces_vectorized(particles, kernel, n_active)
            
            # Gravity
            compute_gravity_direct_batched(particles, n_active)
            
            times.append(time.perf_counter() - t0)
        
        avg_time = np.mean(times[1:])  # Skip first
        fps = 1.0 / avg_time
        
        print(f"\n{n_active:5d} particles: {fps:5.1f} FPS ({avg_time*1000:.1f} ms/step)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--benchmark':
            run_benchmark()
        elif sys.argv[1] == '--earth':
            run_planet_simulation('earth', enable_viz=True)
        elif sys.argv[1] == '--no-viz':
            run_planet_simulation('simple', enable_viz=False)
    else:
        run_planet_simulation('simple', enable_viz=True)