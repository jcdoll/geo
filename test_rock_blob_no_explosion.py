#!/usr/bin/env python3
"""
Test that rock blobs don't explode when created.
Uses the same pressure calculation as the visualizer.
"""

import numpy as np
import matplotlib.pyplot as plt
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.integrator_vectorized import integrate_leapfrog_vectorized
from sph.physics import MaterialType, MaterialDatabase
from sph.physics.density_vectorized import compute_density_vectorized
from sph.physics.forces_vectorized import compute_forces_vectorized
from sph.physics.pressure_overlap_prevention import (
    compute_pressure_simple_repulsive, 
    get_stable_bulk_modulus_improved
)


def create_rock_particles_like_tool(particles: ParticleArrays, tool_radius: float, 
                                   center: tuple = (0, 0)) -> int:
    """Create rock particles like the Material tool does."""
    # Same logic as visualizer
    base_spacing = 1.3
    
    if tool_radius < 5:
        spacing_multiplier = 1.0
    elif tool_radius < 10:
        spacing_multiplier = 1.5
    else:
        spacing_multiplier = 2.0
        
    spacing = base_spacing * spacing_multiplier
    n_radial = int(np.ceil(tool_radius / spacing))
    
    idx = 0
    max_particles = 50  # Same limit as visualizer
    
    for r in range(n_radial + 1):
        if r == 0:
            # Center particle
            particles.position_x[idx] = center[0]
            particles.position_y[idx] = center[1]
            particles.velocity_x[idx] = 0.0
            particles.velocity_y[idx] = 0.0
            particles.material_id[idx] = MaterialType.ROCK.value
            particles.smoothing_h[idx] = 1.3
            particles.mass[idx] = 2700.0 * np.pi * 1.3**2
            particles.density[idx] = 2700.0
            particles.pressure[idx] = 0.0
            particles.temperature[idx] = 288.0
            particles.force_x[idx] = 0.0
            particles.force_y[idx] = 0.0
            particles.neighbor_count[idx] = 0
            idx += 1
        else:
            # Ring of particles
            circumference = 2 * np.pi * r * spacing
            n_in_ring = max(6, int(circumference / spacing))
            
            for i in range(n_in_ring):
                if idx >= max_particles:
                    break
                    
                angle = 2 * np.pi * i / n_in_ring
                px = center[0] + r * spacing * np.cos(angle)
                py = center[1] + r * spacing * np.sin(angle)
                
                # Check if within tool radius
                dist_from_center = np.sqrt((px - center[0])**2 + (py - center[1])**2)
                if dist_from_center <= tool_radius:
                    particles.position_x[idx] = px
                    particles.position_y[idx] = py
                    particles.velocity_x[idx] = 0.0
                    particles.velocity_y[idx] = 0.0
                    particles.material_id[idx] = MaterialType.ROCK.value
                    particles.smoothing_h[idx] = 1.3
                    particles.mass[idx] = 2700.0 * np.pi * 1.3**2
                    particles.density[idx] = 2700.0
                    particles.pressure[idx] = 0.0
                    particles.temperature[idx] = 288.0
                    particles.force_x[idx] = 0.0
                    particles.force_y[idx] = 0.0
                    particles.neighbor_count[idx] = 0
                    idx += 1
                    
            if idx >= max_particles:
                break
    
    return idx


def run_explosion_test(tool_radius: float = 6.0, n_steps: int = 100, dt: float = 0.001):
    """Test if particles explode when created with given tool radius."""
    # Create particles
    max_particles = 200
    particles = ParticleArrays.allocate(max_particles)
    
    # Create rock blob
    n_active = create_rock_particles_like_tool(particles, tool_radius)
    print(f"Created {n_active} rock particles with {tool_radius}m tool")
    
    # Initialize physics
    kernel = CubicSplineKernel(dim=2)
    material_db = MaterialDatabase()
    
    # Create spatial hash
    domain_size = (100.0, 100.0)
    domain_min = (-50.0, -50.0)
    cell_size = 2.6
    spatial_hash = sph.create_spatial_hash(domain_size, cell_size, domain_min=domain_min)
    
    # Track metrics
    max_velocities = []
    avg_velocities = []
    max_forces = []
    positions_x = []
    positions_y = []
    
    # Run simulation
    for step in range(n_steps):
        # Update spatial hash
        spatial_hash.build_vectorized(particles, n_active)
        spatial_hash.query_neighbors_vectorized(particles, n_active, 2.6)
        
        # Compute density
        compute_density_vectorized(particles, kernel, n_active)
        
        # Compute pressure using SAME method as visualizer
        bulk_modulus = np.zeros(n_active, dtype=np.float32)
        for i in range(n_active):
            bulk_modulus[i] = get_stable_bulk_modulus_improved(particles.material_id[i])
        
        density_ref = np.full(n_active, 2700.0, dtype=np.float32)
        
        particles.pressure[:n_active] = compute_pressure_simple_repulsive(
            particles.density[:n_active],
            density_ref,
            bulk_modulus,
            gamma=7.0
        )
        
        # Reset forces
        particles.force_x[:n_active] = 0.0
        particles.force_y[:n_active] = 0.0
        
        # Compute pressure forces (no external gravity)
        compute_forces_vectorized(particles, kernel, n_active, 
                                 gravity=None, alpha_visc=0.1)
        
        # Add overlap prevention forces (same as visualizer)
        from sph.physics.overlap_forces import add_overlap_prevention_forces
        add_overlap_prevention_forces(particles, n_active, 
                                    overlap_distance=0.4, repulsion_strength=100.0)
        
        # Integrate
        domain_bounds = (-50.0, 50.0, -50.0, 50.0)
        integrate_leapfrog_vectorized(particles, n_active, dt, domain_bounds)
        
        # Record metrics
        vel_mag = np.sqrt(particles.velocity_x[:n_active]**2 + 
                         particles.velocity_y[:n_active]**2)
        force_mag = np.sqrt(particles.force_x[:n_active]**2 + 
                           particles.force_y[:n_active]**2)
        
        max_velocities.append(np.max(vel_mag))
        avg_velocities.append(np.mean(vel_mag))
        max_forces.append(np.max(force_mag))
        
        if step % 10 == 0:
            positions_x.append(particles.position_x[:n_active].copy())
            positions_y.append(particles.position_y[:n_active].copy())
            print(f"Step {step}: max_vel={max_velocities[-1]:.2f} m/s, "
                  f"avg_vel={avg_velocities[-1]:.2f} m/s, "
                  f"max_force={max_forces[-1]:.2e} N")
    
    # Check for explosion
    explosion_threshold = 50.0  # m/s
    exploded = np.max(max_velocities) > explosion_threshold
    
    return {
        'exploded': exploded,
        'max_velocities': max_velocities,
        'avg_velocities': avg_velocities,
        'max_forces': max_forces,
        'positions_x': positions_x,
        'positions_y': positions_y,
        'n_active': n_active
    }


def plot_test_results(results: dict, tool_radius: float):
    """Plot the test results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Rock Blob Test - Tool Radius {tool_radius}m - ' + 
                 ('EXPLODED!' if results['exploded'] else 'Stable'),
                 fontsize=16, color='red' if results['exploded'] else 'green')
    
    # Plot 1: Particle positions over time
    ax = axes[0, 0]
    # Initial positions
    ax.scatter(results['positions_x'][0], results['positions_y'][0], 
              alpha=0.3, s=20, label='Initial', color='blue')
    # Final positions
    if len(results['positions_x']) > 1:
        ax.scatter(results['positions_x'][-1], results['positions_y'][-1], 
                  alpha=0.3, s=20, label='Final', color='red')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Particle Positions')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    # Plot 2: Max velocity over time
    ax = axes[0, 1]
    ax.plot(results['max_velocities'], label='Max', color='red')
    ax.plot(results['avg_velocities'], label='Average', color='blue')
    ax.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='Explosion threshold')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Particle Velocities')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Max force over time
    ax = axes[1, 0]
    ax.plot(results['max_forces'])
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Force (N)')
    ax.set_title('Maximum Force')
    ax.set_yscale('log')
    ax.grid(True)
    
    # Plot 4: Text summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""Test Summary:
    
Tool radius: {tool_radius} m
Particles created: {results['n_active']}
Status: {'EXPLODED' if results['exploded'] else 'STABLE'}

Peak velocity: {np.max(results['max_velocities']):.1f} m/s
Final avg velocity: {results['avg_velocities'][-1]:.1f} m/s
Peak force: {np.max(results['max_forces']):.2e} N
"""
    ax.text(0.1, 0.5, summary, fontsize=12, verticalalignment='center',
            fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def main():
    """Test different tool radii."""
    tool_radii = [3.0, 6.0, 10.0]
    
    for radius in tool_radii:
        print(f"\n{'='*50}")
        print(f"Testing tool radius: {radius}m")
        print('='*50)
        
        results = run_explosion_test(tool_radius=radius, n_steps=100, dt=0.001)
        
        # Plot results
        fig = plot_test_results(results, radius)
        filename = f'rock_blob_test_{radius}m.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        
        # Summary
        if results['exploded']:
            print(f"⚠️  EXPLOSION DETECTED with {radius}m tool!")
        else:
            print(f"✓  Stable with {radius}m tool")


if __name__ == "__main__":
    main()