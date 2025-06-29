#!/usr/bin/env python3
"""
Test rock blob stability with cohesion forces.

Creates a small blob of rock particles and verifies they stay together
without exploding or drifting apart.
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
from sph.physics.cohesion_vectorized import compute_cohesive_forces_vectorized


def create_rock_blob(particles: ParticleArrays, n_particles: int, 
                     center: tuple = (0, 0), radius: float = 5.0) -> int:
    """Create a circular blob of rock particles."""
    spacing = 1.3
    idx = 0
    
    # Create particles in a hexagonal pattern
    for r in np.arange(0, radius, spacing):
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
            n_in_ring = int(2 * np.pi * r / spacing)
            for i in range(n_in_ring):
                angle = 2 * np.pi * i / n_in_ring
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                
                if idx >= n_particles:
                    break
                    
                particles.position_x[idx] = x
                particles.position_y[idx] = y
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
                
        if idx >= n_particles:
            break
    
    return idx


def run_stability_test(n_steps: int = 1000, dt: float = 0.001, 
                      enable_cohesion: bool = True, 
                      cohesion_factor: float = 1.5):
    """Run stability test and return metrics."""
    # Create particles
    max_particles = 200
    particles = ParticleArrays.allocate(max_particles)
    
    # Create rock blob
    n_active = create_rock_blob(particles, max_particles, radius=5.0)
    print(f"Created rock blob with {n_active} particles")
    
    # Initialize physics
    kernel = CubicSplineKernel(dim=2)
    material_db = MaterialDatabase()
    
    # Create spatial hash
    domain_size = (20.0, 20.0)
    domain_min = (-10.0, -10.0)
    cell_size = 2.6  # 2 * smoothing_h
    spatial_hash = sph.create_spatial_hash(domain_size, cell_size, domain_min=domain_min)
    
    # Track metrics
    positions_x = []
    positions_y = []
    velocities = []
    forces = []
    kinetic_energies = []
    center_of_mass_x = []
    center_of_mass_y = []
    max_distances = []
    
    # Initial metrics
    initial_com_x = np.mean(particles.position_x[:n_active])
    initial_com_y = np.mean(particles.position_y[:n_active])
    
    # Run simulation
    for step in range(n_steps):
        # Update spatial hash
        spatial_hash.build_vectorized(particles, n_active)
        spatial_hash.query_neighbors_vectorized(particles, n_active, 2.6)
        
        # Compute density
        compute_density_vectorized(particles, kernel, n_active)
        
        # Compute pressure (simple Tait equation)
        bulk_modulus = 50e9  # Rock bulk modulus
        density_ref = 2700.0
        gamma = 7.0
        particles.pressure[:n_active] = bulk_modulus / gamma * (
            (particles.density[:n_active] / density_ref)**gamma - 1.0
        )
        
        # Reset forces
        particles.force_x[:n_active] = 0.0
        particles.force_y[:n_active] = 0.0
        
        # Compute pressure forces (no external gravity)
        compute_forces_vectorized(particles, kernel, n_active, 
                                 gravity=None, alpha_visc=0.1)
        
        # Add cohesion if enabled
        if enable_cohesion:
            compute_cohesive_forces_vectorized(particles, kernel, n_active,
                                             material_db, cutoff_factor=cohesion_factor,
                                             temperature_softening=False)
        
        # Integrate
        domain_bounds = (-10.0, 10.0, -10.0, 10.0)  # (xmin, xmax, ymin, ymax)
        integrate_leapfrog_vectorized(particles, n_active, dt, domain_bounds)
        
        # Record metrics every 10 steps
        if step % 10 == 0:
            # Center of mass
            com_x = np.mean(particles.position_x[:n_active])
            com_y = np.mean(particles.position_y[:n_active])
            center_of_mass_x.append(com_x)
            center_of_mass_y.append(com_y)
            
            # Max distance from center
            dx = particles.position_x[:n_active] - com_x
            dy = particles.position_y[:n_active] - com_y
            distances = np.sqrt(dx**2 + dy**2)
            max_distances.append(np.max(distances))
            
            # Velocities
            vel_mag = np.sqrt(particles.velocity_x[:n_active]**2 + 
                            particles.velocity_y[:n_active]**2)
            velocities.append(np.mean(vel_mag))
            
            # Forces
            force_mag = np.sqrt(particles.force_x[:n_active]**2 + 
                              particles.force_y[:n_active]**2)
            forces.append(np.mean(force_mag))
            
            # Kinetic energy
            ke = 0.5 * np.sum(particles.mass[:n_active] * vel_mag**2)
            kinetic_energies.append(ke)
            
            # Store positions
            positions_x.append(particles.position_x[:n_active].copy())
            positions_y.append(particles.position_y[:n_active].copy())
        
        # Print progress
        if step % 100 == 0:
            drift = np.sqrt((com_x - initial_com_x)**2 + (com_y - initial_com_y)**2)
            print(f"Step {step}: COM drift={drift:.3f}, max_dist={distances.max():.3f}, "
                  f"avg_vel={vel_mag.mean():.3f}, avg_force={force_mag.mean():.1f}")
    
    return {
        'positions_x': positions_x,
        'positions_y': positions_y,
        'center_of_mass_x': center_of_mass_x,
        'center_of_mass_y': center_of_mass_y,
        'max_distances': max_distances,
        'velocities': velocities,
        'forces': forces,
        'kinetic_energies': kinetic_energies,
        'n_active': n_active
    }


def plot_results(results: dict, title: str):
    """Plot test results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Initial and final positions
    ax = axes[0, 0]
    ax.scatter(results['positions_x'][0], results['positions_y'][0], 
              alpha=0.5, s=10, label='Initial')
    ax.scatter(results['positions_x'][-1], results['positions_y'][-1], 
              alpha=0.5, s=10, label='Final')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Particle Positions')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    # Plot 2: Center of mass drift
    ax = axes[0, 1]
    com_drift = np.sqrt((np.array(results['center_of_mass_x']) - results['center_of_mass_x'][0])**2 +
                       (np.array(results['center_of_mass_y']) - results['center_of_mass_y'][0])**2)
    ax.plot(com_drift)
    ax.set_xlabel('Time steps (x10)')
    ax.set_ylabel('COM drift (m)')
    ax.set_title('Center of Mass Drift')
    ax.grid(True)
    
    # Plot 3: Max distance from COM
    ax = axes[0, 2]
    ax.plot(results['max_distances'])
    ax.set_xlabel('Time steps (x10)')
    ax.set_ylabel('Max distance (m)')
    ax.set_title('Maximum Distance from COM')
    ax.grid(True)
    
    # Plot 4: Average velocity
    ax = axes[1, 0]
    ax.plot(results['velocities'])
    ax.set_xlabel('Time steps (x10)')
    ax.set_ylabel('Avg velocity (m/s)')
    ax.set_title('Average Velocity')
    ax.grid(True)
    
    # Plot 5: Average force
    ax = axes[1, 1]
    ax.plot(results['forces'])
    ax.set_xlabel('Time steps (x10)')
    ax.set_ylabel('Avg force (N)')
    ax.set_title('Average Force Magnitude')
    ax.grid(True)
    
    # Plot 6: Kinetic energy
    ax = axes[1, 2]
    ax.plot(results['kinetic_energies'])
    ax.set_xlabel('Time steps (x10)')
    ax.set_ylabel('Kinetic energy (J)')
    ax.set_title('Total Kinetic Energy')
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def main():
    """Run stability tests with different configurations."""
    print("Testing rock blob stability...")
    
    # Test 1: With cohesion (current implementation)
    print("\n=== Test 1: With cohesion (cutoff=1.5) ===")
    results1 = run_stability_test(n_steps=1000, enable_cohesion=True, cohesion_factor=1.5)
    
    # Test 2: Without cohesion
    print("\n=== Test 2: Without cohesion ===")
    results2 = run_stability_test(n_steps=1000, enable_cohesion=False)
    
    # Test 3: With reduced cohesion
    print("\n=== Test 3: With reduced cohesion (cutoff=1.2) ===")
    results3 = run_stability_test(n_steps=1000, enable_cohesion=True, cohesion_factor=1.2)
    
    # Plot results
    fig1 = plot_results(results1, "Test 1: With Cohesion (cutoff=1.5)")
    fig2 = plot_results(results2, "Test 2: Without Cohesion")
    fig3 = plot_results(results3, "Test 3: With Reduced Cohesion (cutoff=1.2)")
    
    # Save plots
    fig1.savefig('rock_blob_stability_with_cohesion.png', dpi=150, bbox_inches='tight')
    fig2.savefig('rock_blob_stability_without_cohesion.png', dpi=150, bbox_inches='tight')
    fig3.savefig('rock_blob_stability_reduced_cohesion.png', dpi=150, bbox_inches='tight')
    
    print("\nPlots saved to rock_blob_stability_*.png")
    
    # Summary
    print("\n=== SUMMARY ===")
    for test_name, results in [("With cohesion (1.5)", results1), 
                               ("Without cohesion", results2),
                               ("With reduced cohesion (1.2)", results3)]:
        final_com_drift = np.sqrt((results['center_of_mass_x'][-1] - results['center_of_mass_x'][0])**2 +
                                 (results['center_of_mass_y'][-1] - results['center_of_mass_y'][0])**2)
        final_max_dist = results['max_distances'][-1]
        final_ke = results['kinetic_energies'][-1]
        
        print(f"\n{test_name}:")
        print(f"  Final COM drift: {final_com_drift:.3f} m")
        print(f"  Final max distance: {final_max_dist:.3f} m")
        print(f"  Final kinetic energy: {final_ke:.3e} J")
        
        # Check stability criteria
        stable = final_com_drift < 0.1 and final_ke < 1e3
        print(f"  Stable: {'YES' if stable else 'NO'}")


if __name__ == "__main__":
    main()