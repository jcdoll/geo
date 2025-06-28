"""
Planet simulation with improved stability through cohesive forces and proper gas EOS.

Key improvements:
- Material-specific equation of state (ideal gas for air)
- Cohesive forces to hold solids together
- Improved pressure limits to prevent explosions
"""

import numpy as np
import time
import sys

# Core SPH components
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.integrator_vectorized import integrate_leapfrog_vectorized

# Physics
from sph.physics.density_vectorized import compute_density_vectorized
from sph.physics.forces_vectorized import compute_forces_vectorized
from sph.physics.gravity_vectorized import compute_gravity_direct_batched
from sph.physics.materials import MaterialDatabase, MaterialType
from sph.physics.thermal_vectorized import update_temperature_full
from sph.physics.cohesion import compute_cohesive_forces

# Scenarios and utilities
from sph.scenarios.planet import create_planet_simple, create_planet_earth_like
from sph.physics.gravity_vectorized import compute_center_of_mass
from sph.core.timestep import compute_adaptive_timestep

# Visualization (optional)
try:
    from sph.visualization.vispy_renderer import VispyRenderer
    viz_available = True
except ImportError:
    viz_available = False
    print("VisPy not available - running without visualization")


def material_aware_pressure_stable(particles, material_db, n_active):
    """
    Compute pressure using material-specific EOS with improved stability.
    
    - Uses ideal gas law for gases (air, water vapor)
    - Uses modified Tait equation for liquids/solids with clamping
    - Adds pressure floor to prevent particle clumping
    """
    # Constants
    R_air = 287.0  # Specific gas constant for air [J/(kg·K)]
    R_vapor = 461.5  # Specific gas constant for water vapor [J/(kg·K)]
    min_pressure = 100.0  # Minimum pressure floor [Pa]
    
    # Reset pressure
    particles.pressure[:n_active] = 0.0
    
    # Process by material type for efficiency
    for mat_type in MaterialType:
        # Find particles of this material
        mask = particles.material_id[:n_active] == mat_type.value
        if not np.any(mask):
            continue
        
        # Get indices
        indices = np.where(mask)[0]
        
        if mat_type in [MaterialType.AIR, MaterialType.WATER_VAPOR]:
            # Ideal gas law: P = ρRT
            R = R_air if mat_type == MaterialType.AIR else R_vapor
            particles.pressure[indices] = (
                particles.density[indices] * R * particles.temperature[indices]
            )
            
        elif mat_type == MaterialType.SPACE:
            # Space has minimal pressure
            particles.pressure[indices] = min_pressure * 0.01
            
        else:
            # Liquids and solids use Tait equation
            mat_props = material_db.get_properties(mat_type)
            bulk_modulus = mat_props.bulk_modulus
            density_ref = mat_props.density_ref
            
            # Reduce bulk modulus for stability
            if mat_type == MaterialType.WATER:
                B_stable = bulk_modulus * 0.01  # Much softer water
                gamma = 7.0
            else:
                B_stable = bulk_modulus * 0.001  # Very soft solids
                gamma = 3.0  # Lower exponent for solids
            
            # Clamp density ratio to prevent extreme pressures
            density_ratio = particles.density[indices] / density_ref
            density_ratio = np.clip(density_ratio, 0.5, 2.0)
            
            # Modified Tait equation
            particles.pressure[indices] = B_stable * (density_ratio**gamma - 1)
    
    # Apply pressure floor to all particles
    particles.pressure[:n_active] = np.maximum(particles.pressure[:n_active], min_pressure)
    
    # Additional clamping for extreme cases
    max_pressure = 1e9  # 1 GPa max
    particles.pressure[:n_active] = np.minimum(particles.pressure[:n_active], max_pressure)


def run_stable_planet_simulation(scenario='simple', enable_viz=True):
    """Run planet simulation with improved stability."""
    
    print(f"\n=== Stable SPH Planet Simulation ({scenario}) ===")
    
    # Create planet
    if scenario == 'earth':
        particles, n_active = create_planet_earth_like(
            radius_km=500,
            particle_spacing_km=10,
            center=(5000, 5000)
        )
        domain_size = (10000 * 1000, 10000 * 1000)
        G = 6.67430e-11
        time_scale = 1.0
    else:
        particles, n_active = create_planet_simple(
            radius=2000,
            particle_spacing=50,
            center=(0, 0)  # Centered at origin
        )
        domain_size = (10000, 10000)
        G = 6.67430e-11 * 1e6  # Scaled for smaller system
        time_scale = 0.001  # Faster evolution
    
    print(f"Created planet with {n_active} particles")
    
    # SPH setup
    material_db = MaterialDatabase()
    kernel = CubicSplineKernel(dim=2)
    
    # Spatial hash with centered domain
    domain_min = (-domain_size[0]/2, -domain_size[1]/2)
    cell_size = 100 if scenario == 'simple' else 20000
    
    import sph
    spatial_hash = sph.create_spatial_hash(domain_size, cell_size, domain_min=domain_min)
    
    # Initial conditions
    dt = 0.001 * time_scale
    sim_time = 0.0
    max_steps = 1000
    stats_interval = 50
    
    # Enable/disable physics
    enable_cohesion = True
    enable_self_gravity = True
    enable_thermal = True
    
    # Visualization
    renderer = None
    if viz_available and enable_viz:
        renderer = VispyRenderer(
            window_size=(1200, 1200),
            domain_size=domain_size,
            scale_factor=0.001 if scenario == 'earth' else 1.0,
            title="Stable SPH Planet"
        )
    
    # Statistics
    energy_history = []
    t0 = time.time()
    
    # Main simulation loop
    print("\nStarting simulation...")
    
    for step in range(max_steps):
        # Neighbor search
        particles_query = particles if scenario == 'simple' else None
        if particles_query is None:
            # For Earth scenario, convert to km for spatial hash
            particles_km = ParticleArrays.allocate(particles.max_particles)
            particles_km.position_x[:n_active] = particles.position_x[:n_active] / 1000
            particles_km.position_y[:n_active] = particles.position_y[:n_active] / 1000
            particles_query = particles_km
        
        spatial_hash.query_neighbors_vectorized(
            particles_query, n_active, cell_size * 2 / (1000 if scenario == 'earth' else 1)
        )
        
        if scenario == 'earth':
            # Copy neighbor info back and convert distances
            particles.neighbor_ids[:] = particles_km.neighbor_ids
            particles.neighbor_distances[:] = particles_km.neighbor_distances * 1000
            particles.neighbor_count[:] = particles_km.neighbor_count
        
        # Density
        compute_density_vectorized(particles, kernel, n_active)
        
        # Material-aware pressure with stability improvements
        material_aware_pressure_stable(particles, material_db, n_active)
        
        # Forces
        # 1. Pressure and viscosity forces
        compute_forces_vectorized(
            particles, kernel, n_active, 
            gravity=None,  # No external gravity for planet
            alpha_visc=0.5,  # Increased artificial viscosity for stability
            beta_visc=0.0
        )
        
        # 2. Cohesive forces (key for stability!)
        if enable_cohesion:
            compute_cohesive_forces(
                particles, kernel, n_active, material_db,
                cutoff_factor=1.5,
                temperature_softening=True
            )
        
        # 3. Self-gravity
        if enable_self_gravity:
            compute_gravity_direct_batched(
                particles, n_active, G=G,
                softening=cell_size * 2.0  # Increased softening for stability
            )
        
        # Thermal physics
        if enable_thermal:
            thermal_stats = update_temperature_full(
                particles, kernel, material_db, n_active, dt,
                enable_radiation=True,
                enable_transitions=False  # Disable for initial stability
            )
        
        # Integration with boundary conditions
        domain_bounds = (-domain_size[0]/2, domain_size[0]/2, 
                        -domain_size[1]/2, domain_size[1]/2)
        integrate_leapfrog_vectorized(
            particles, n_active, dt,
            domain_bounds,
            damping=0.995  # Slight damping for stability
        )
        
        # Adaptive timestep
        if step % 50 == 0 and step > 0:
            dt_new = compute_adaptive_timestep(
                particles, n_active, cfl_factor=0.1  # Conservative CFL
            )
            dt = np.clip(dt_new, dt*0.5, dt*2.0) * time_scale
        
        # Statistics
        if step % stats_interval == 0:
            # Compute energies
            v_mag = np.sqrt(particles.velocity_x[:n_active]**2 + 
                           particles.velocity_y[:n_active]**2)
            kinetic_energy = 0.5 * np.sum(particles.mass[:n_active] * v_mag**2)
            
            # Center of mass
            cm_x, cm_y, total_mass = compute_center_of_mass(particles, n_active)
            
            # Pressure stats by material
            print(f"\nStep {step}, t={sim_time:.3f}s, dt={dt*1000:.1f}ms")
            print(f"  Center of mass: ({cm_x:.1f}, {cm_y:.1f}) m")
            print(f"  Max velocity: {np.max(v_mag):.1f} m/s")
            
            # Material-specific pressure stats
            for mat_type in [MaterialType.AIR, MaterialType.WATER, MaterialType.ROCK]:
                mask = particles.material_id[:n_active] == mat_type.value
                if np.any(mask):
                    p_min = np.min(particles.pressure[:n_active][mask])
                    p_max = np.max(particles.pressure[:n_active][mask])
                    p_mean = np.mean(particles.pressure[:n_active][mask])
                    print(f"  {mat_type.name}: P = {p_min:.0f} - {p_max:.0f} Pa "
                          f"(mean {p_mean:.0f})")
            
            if enable_thermal and 'min_temperature' in thermal_stats:
                print(f"  Temperature: {thermal_stats['min_temperature']:.0f} - "
                      f"{thermal_stats['max_temperature']:.0f} K")
            
            energy_history.append(kinetic_energy)
        
        # Visualization
        if renderer and step % 10 == 0:
            particles_display = particles
            if scenario == 'earth':
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
    
    # Final statistics
    elapsed = time.time() - t0
    print(f"\n=== Simulation Complete ===")
    print(f"Total time: {elapsed:.1f}s for {step+1} steps")
    print(f"Average: {elapsed/(step+1)*1000:.1f} ms/step")
    
    # Check stability
    if len(energy_history) > 2:
        energy_change = (energy_history[-1] - energy_history[0]) / energy_history[0]
        print(f"Energy change: {energy_change*100:.1f}%")
        
        if abs(energy_change) < 0.1:
            print("✓ Simulation is stable!")
        else:
            print("⚠ Significant energy drift detected")
    
    if renderer:
        renderer.close()


if __name__ == "__main__":
    # Command line options
    scenario = 'simple'
    enable_viz = True
    
    if len(sys.argv) > 1:
        if '--earth' in sys.argv:
            scenario = 'earth'
        if '--no-viz' in sys.argv:
            enable_viz = False
    
    run_stable_planet_simulation(scenario, enable_viz)