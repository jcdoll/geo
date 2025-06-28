#!/usr/bin/env python3
"""
Minimal test of planet stability with cohesive forces and proper gas EOS.
"""

import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, '.')

from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.physics.density_vectorized import compute_density_vectorized
from sph.physics.forces_vectorized import compute_forces_vectorized
from sph.physics.gravity_vectorized import compute_gravity_direct_batched
from sph.physics.cohesion import compute_cohesive_forces
from sph.physics.materials import MaterialDatabase, MaterialType
from sph.core.integrator_vectorized import integrate_leapfrog_vectorized
import sph


def create_mini_planet(radius=500.0, spacing=50.0):
    """Create a tiny planet for testing."""
    particles = ParticleArrays.allocate(10000)
    n_active = 0
    
    # Generate particles in circular layers
    core_radius = radius * 0.2  # Uranium core
    rock_radius = radius * 0.7  # Rock layer
    water_radius = radius * 0.85  # Water layer
    
    # Fill with particles
    for y in np.arange(-radius, radius + spacing, spacing):
        for x in np.arange(-radius, radius + spacing, spacing):
            r = np.sqrt(x**2 + y**2)
            
            if r <= radius:
                particles.position_x[n_active] = x
                particles.position_y[n_active] = y
                
                # Assign material based on radius
                # Mass based on material density and particle volume
                # Volume ≈ spacing² for 2D (assuming unit thickness)
                particle_volume = spacing * spacing
                
                if r <= core_radius:
                    particles.material_id[n_active] = MaterialType.URANIUM.value
                    # Uranium density ~ 19000 kg/m³
                    particles.mass[n_active] = 19000.0 * particle_volume / 1000  # Scale down
                elif r <= rock_radius:
                    particles.material_id[n_active] = MaterialType.ROCK.value
                    # Rock density ~ 2700 kg/m³
                    particles.mass[n_active] = 2700.0 * particle_volume / 1000
                elif r <= water_radius:
                    particles.material_id[n_active] = MaterialType.WATER.value
                    # Water density ~ 1000 kg/m³
                    particles.mass[n_active] = 1000.0 * particle_volume / 1000
                else:
                    particles.material_id[n_active] = MaterialType.AIR.value
                    # Air density ~ 1.2 kg/m³
                    particles.mass[n_active] = 1.2 * particle_volume / 1000
                
                particles.temperature[n_active] = 300.0
                particles.smoothing_h[n_active] = spacing * 2.0
                n_active += 1
    
    return particles, n_active


def material_aware_pressure_fixed(particles, material_db, n_active):
    """Fixed pressure calculation."""
    R_air = 287.0  # J/(kg·K)
    min_pressure = 100.0  # Pa
    
    for i in range(n_active):
        mat_id = particles.material_id[i]
        
        if mat_id == MaterialType.AIR.value:
            # Ideal gas law for air
            particles.pressure[i] = particles.density[i] * R_air * particles.temperature[i]
        else:
            # Simplified Tait for others
            mat_props = material_db.get_properties(MaterialType(mat_id))
            B = mat_props.bulk_modulus * 0.001  # Much softer
            rho0 = mat_props.density_ref
            
            ratio = np.clip(particles.density[i] / rho0, 0.5, 2.0)
            particles.pressure[i] = B * (ratio**3 - 1)  # Lower exponent
    
    # Apply pressure floor
    particles.pressure[:n_active] = np.maximum(particles.pressure[:n_active], min_pressure)


def run_test():
    """Run planet stability test."""
    print("Creating mini planet...")
    particles, n_active = create_mini_planet(radius=500.0, spacing=50.0)
    print(f"Created {n_active} particles")
    
    # Count materials
    for mat in [MaterialType.URANIUM, MaterialType.ROCK, MaterialType.WATER, MaterialType.AIR]:
        count = np.sum(particles.material_id[:n_active] == mat.value)
        print(f"  {mat.name}: {count} particles")
    
    # Setup
    kernel = CubicSplineKernel(dim=2)
    material_db = MaterialDatabase()
    
    # Spatial hash
    domain_size = (2000.0, 2000.0)
    domain_min = (-1000.0, -1000.0)
    cell_size = 100.0
    spatial_hash = sph.create_spatial_hash(domain_size, cell_size, domain_min=domain_min)
    
    # Debug particle positions
    print(f"\nParticle position range:")
    print(f"  X: {np.min(particles.position_x[:n_active]):.1f} to {np.max(particles.position_x[:n_active]):.1f}")
    print(f"  Y: {np.min(particles.position_y[:n_active]):.1f} to {np.max(particles.position_y[:n_active]):.1f}")
    
    # Simulation parameters
    dt = 0.001
    G = 6.67430e-11 * 1e6  # Scaled gravity
    
    print("\nRunning simulation...")
    
    # Initial density computation
    search_radius = 2.0 * 100.0  # 2 * smoothing_h
    spatial_hash.build_vectorized(particles, n_active)
    spatial_hash.query_neighbors_vectorized(particles, n_active, search_radius)
    compute_density_vectorized(particles, kernel, n_active)
    
    # Check neighbor finding
    avg_neighbors = np.mean(particles.neighbor_count[:n_active])
    max_neighbors = np.max(particles.neighbor_count[:n_active])
    print(f"\nNeighbor stats: avg={avg_neighbors:.1f}, max={max_neighbors}")
    
    # Check initial densities
    print("\nInitial densities:")
    for mat in [MaterialType.URANIUM, MaterialType.ROCK, MaterialType.WATER, MaterialType.AIR]:
        mask = particles.material_id[:n_active] == mat.value
        if np.any(mask):
            densities = particles.density[:n_active][mask]
            print(f"  {mat.name}: {np.mean(densities):.1f} kg/m³ (expected {material_db.get_properties(mat).density_ref:.1f})")
    
    # Initial center of mass
    total_mass = np.sum(particles.mass[:n_active])
    cm_x0 = np.sum(particles.mass[:n_active] * particles.position_x[:n_active]) / total_mass
    cm_y0 = np.sum(particles.mass[:n_active] * particles.position_y[:n_active]) / total_mass
    
    for step in range(100):
        # Neighbor search (need to search within kernel support = 2*h)
        search_radius = 2.0 * 100.0  # 2 * smoothing_h
        spatial_hash.build_vectorized(particles, n_active)
        spatial_hash.query_neighbors_vectorized(particles, n_active, search_radius)
        
        # Density
        compute_density_vectorized(particles, kernel, n_active)
        
        # Pressure
        material_aware_pressure_fixed(particles, material_db, n_active)
        
        # Forces
        particles.force_x[:n_active] = 0.0
        particles.force_y[:n_active] = 0.0
        
        # Pressure forces
        compute_forces_vectorized(particles, kernel, n_active, gravity=None, alpha_visc=0.5)
        
        # Cohesive forces
        compute_cohesive_forces(particles, kernel, n_active, material_db, cutoff_factor=1.2)
        
        # Self-gravity
        compute_gravity_direct_batched(particles, n_active, G=G, softening=100.0)
        
        # Integration
        integrate_leapfrog_vectorized(
            particles, n_active, dt,
            (-1000, 1000, -1000, 1000),
            damping=0.99
        )
        
        if step % 20 == 0:
            # Check center of mass
            cm_x = np.sum(particles.mass[:n_active] * particles.position_x[:n_active]) / total_mass
            cm_y = np.sum(particles.mass[:n_active] * particles.position_y[:n_active]) / total_mass
            
            # Check air particles
            air_mask = particles.material_id[:n_active] == MaterialType.AIR.value
            if np.any(air_mask):
                air_pressures = particles.pressure[:n_active][air_mask]
                air_densities = particles.density[:n_active][air_mask]
                air_velocities = np.sqrt(particles.velocity_x[:n_active][air_mask]**2 + 
                                       particles.velocity_y[:n_active][air_mask]**2)
                print(f"\nStep {step}:")
                print(f"  CM drift: ({cm_x - cm_x0:.1f}, {cm_y - cm_y0:.1f})")
                print(f"  Air: ρ = {np.min(air_densities):.1f} - {np.max(air_densities):.1f} kg/m³")
                print(f"       P = {np.min(air_pressures):.0f} - {np.max(air_pressures):.0f} Pa")
                print(f"       v_max = {np.max(air_velocities):.1f} m/s")
    
    print("\n✓ Test complete!")


if __name__ == "__main__":
    run_test()