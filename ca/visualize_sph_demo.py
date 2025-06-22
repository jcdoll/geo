"""Visual demonstration of SPH particle simulation.

Shows:
1. Particle positions colored by material
2. Density field
3. Pressure field
4. Forces on particles
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from particle_hybrid_optimized import FastParticleHybrid
import matplotlib.patches as patches


def create_demo_setup(sim):
    """Create a clear demo setup with water blob falling."""
    # Clear and reset
    n_particles = 5000
    sim.n_particles = n_particles
    sim.pos = np.zeros((n_particles, 2), dtype=np.float32)
    sim.vel = np.zeros((n_particles, 2), dtype=np.float32)
    sim.mass = np.ones(n_particles, dtype=np.float32)
    sim.material = np.zeros(n_particles, dtype=np.int32)
    
    domain_x = sim.width * sim.cell_size
    domain_y = sim.height * sim.cell_size
    
    # Create a water blob in the air
    blob_center = [domain_x * 0.5, domain_y * 0.7]
    blob_radius = domain_x * 0.15
    
    # Water particles in blob
    n_water = n_particles // 2
    for i in range(n_water):
        # Random position within circle
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 1)) * blob_radius
        sim.pos[i] = [
            blob_center[0] + r * np.cos(angle),
            blob_center[1] + r * np.sin(angle)
        ]
        sim.material[i] = 1  # water
        sim.mass[i] = 100.0
        sim.vel[i, 1] = -50.0  # Initial downward velocity
    
    # Air particles around
    for i in range(n_water, n_particles):
        sim.pos[i] = [
            np.random.uniform(0, domain_x),
            np.random.uniform(0, domain_y)
        ]
        sim.material[i] = 0  # air
        sim.mass[i] = 0.1
        
    # Add a rock platform at bottom
    platform_y = domain_y * 0.2
    platform_particles = 500
    for i in range(n_particles - platform_particles, n_particles):
        sim.pos[i] = [
            np.random.uniform(domain_x * 0.3, domain_x * 0.7),
            np.random.uniform(0, platform_y)
        ]
        sim.material[i] = 2  # rock
        sim.mass[i] = 200.0
        sim.vel[i] = [0, 0]  # stationary


def visualize_sph_snapshot(sim, step_num=0):
    """Create a detailed visualization of SPH state."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create subplots
    ax1 = plt.subplot(2, 3, 1)  # Particles
    ax2 = plt.subplot(2, 3, 2)  # Density field
    ax3 = plt.subplot(2, 3, 3)  # Pressure field
    ax4 = plt.subplot(2, 3, 4)  # Particle velocities
    ax5 = plt.subplot(2, 3, 5)  # Pressure gradient
    ax6 = plt.subplot(2, 3, 6)  # Net forces
    
    domain_x = sim.width * sim.cell_size
    domain_y = sim.height * sim.cell_size
    
    # 1. Particle positions
    colors = {0: 'lightblue', 1: 'blue', 2: 'brown'}
    sizes = {0: 5, 1: 20, 2: 30}
    labels = {0: 'Air', 1: 'Water', 2: 'Rock'}
    
    for mat in [0, 1, 2]:
        mask = sim.material == mat
        if np.any(mask):
            ax1.scatter(sim.pos[mask, 0], sim.pos[mask, 1],
                       c=colors[mat], s=sizes[mat], alpha=0.6,
                       label=labels[mat], edgecolors='black', linewidth=0.5)
    
    ax1.set_xlim(0, domain_x)
    ax1.set_ylim(0, domain_y)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Particles (Step {step_num})')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Add blob circle
    if step_num == 0:
        circle = patches.Circle((domain_x * 0.5, domain_y * 0.7), 
                               domain_x * 0.15, fill=False, 
                               edgecolor='red', linewidth=2, linestyle='--')
        ax1.add_patch(circle)
    
    # 2. Density field
    im2 = ax2.imshow(sim.grid_density, origin='lower', cmap='viridis',
                     extent=[0, domain_x, 0, domain_y])
    ax2.set_title('Density (kg/m³)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('ρ')
    
    # 3. Pressure field
    im3 = ax3.imshow(sim.grid_pressure, origin='lower', cmap='plasma',
                     extent=[0, domain_x, 0, domain_y],
                     vmin=0, vmax=5000)
    ax3.set_title('Pressure (Pa)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('P')
    
    # 4. Particle velocities (arrows)
    # Subsample for clarity
    n_arrows = 200
    indices = np.random.choice(sim.n_particles, min(n_arrows, sim.n_particles), replace=False)
    
    # Scale velocities for visibility
    scale = 20.0
    ax4.quiver(sim.pos[indices, 0], sim.pos[indices, 1],
               sim.vel[indices, 0], sim.vel[indices, 1],
               scale=scale, alpha=0.6)
    
    # Color particles by material
    for mat in [0, 1, 2]:
        mask = sim.material == mat
        if np.any(mask):
            ax4.scatter(sim.pos[mask, 0], sim.pos[mask, 1],
                       c=colors[mat], s=10, alpha=0.3)
    
    ax4.set_xlim(0, domain_x)
    ax4.set_ylim(0, domain_y)
    ax4.set_aspect('equal')
    ax4.set_title('Particle Velocities')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    
    # Add velocity scale
    ax4.text(0.02, 0.98, f'Max |v| = {np.max(np.abs(sim.vel)):.1f} m/s',
             transform=ax4.transAxes, va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Pressure gradient
    grad_p_y = np.zeros_like(sim.grid_pressure)
    grad_p_y[1:-1, :] = (sim.grid_pressure[2:, :] - sim.grid_pressure[:-2, :]) / (2 * sim.cell_size)
    
    im5 = ax5.imshow(grad_p_y, origin='lower', cmap='RdBu_r',
                     extent=[0, domain_x, 0, domain_y],
                     vmin=-1000, vmax=1000)
    ax5.set_title('Pressure Gradient ∂P/∂y')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    cbar5 = plt.colorbar(im5, ax=ax5)
    cbar5.set_label('∂P/∂y (Pa/m)')
    
    # 6. Compare forces: gravity vs pressure gradient
    # Expected gravity force
    gravity_force = sim.grid_density * sim.gravity_y
    
    # Net force
    net_force = gravity_force - grad_p_y
    
    im6 = ax6.imshow(net_force, origin='lower', cmap='seismic',
                     extent=[0, domain_x, 0, domain_y],
                     vmin=-5000, vmax=5000)
    ax6.set_title('Net Force (ρg - ∂P/∂y)')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    cbar6 = plt.colorbar(im6, ax=ax6)
    cbar6.set_label('F_net (N/m³)')
    
    plt.tight_layout()
    return fig


def create_time_evolution():
    """Show evolution over time."""
    print("Creating SPH visualization...")
    
    # Create simulation
    sim = FastParticleHybrid(width=64, height=64, n_particles=5000)
    create_demo_setup(sim)
    
    # Time points to visualize
    times = [0, 10, 25, 50, 100]
    figs = []
    
    for target_step in times:
        # Run to target step
        while len(figs) < target_step:
            sim.step(dt=0.01)
        
        # Create snapshot
        print(f"Creating snapshot at step {target_step}...")
        fig = visualize_sph_snapshot(sim, target_step)
        fig.suptitle(f'SPH Particle Simulation - Step {target_step} (t = {target_step * 0.01:.2f}s)', 
                     fontsize=16)
        
        # Save
        filename = f'sph_visual_step_{target_step:03d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()
        
        figs.append(target_step)
    
    # Also create a simple particle-only view
    create_simple_view(sim)


def create_simple_view(sim):
    """Create a simple view showing just particles."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Run a bit more
    for _ in range(50):
        sim.step(0.01)
    
    domain_x = sim.width * sim.cell_size
    domain_y = sim.height * sim.cell_size
    
    # Three views at different times
    for idx, (ax, steps) in enumerate(zip(axes, [0, 25, 50])):
        if idx > 0:
            for _ in range(25):
                sim.step(0.01)
        
        # Plot particles
        colors = {0: 'lightblue', 1: 'blue', 2: 'brown'}
        sizes = {0: 10, 1: 30, 2: 40}
        
        for mat in [0, 1, 2]:
            mask = sim.material == mat
            if np.any(mask):
                ax.scatter(sim.pos[mask, 0], sim.pos[mask, 1],
                          c=colors[mat], s=sizes[mat], alpha=0.6,
                          edgecolors='black', linewidth=0.5)
        
        ax.set_xlim(0, domain_x)
        ax.set_ylim(0, domain_y)
        ax.set_aspect('equal')
        ax.set_title(f't = {steps * 0.01:.2f}s')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Add annotations
        if idx == 0:
            ax.annotate('Water blob\n(falling)', 
                       xy=(domain_x * 0.5, domain_y * 0.7),
                       xytext=(domain_x * 0.7, domain_y * 0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=12, ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.annotate('Rock platform', 
                       xy=(domain_x * 0.5, domain_y * 0.1),
                       xytext=(domain_x * 0.2, domain_y * 0.3),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2),
                       fontsize=12, ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('SPH Water Blob Falling onto Rock Platform', fontsize=16)
    plt.tight_layout()
    plt.savefig('sph_simple_view.png', dpi=150, bbox_inches='tight')
    print("Saved sph_simple_view.png")
    plt.close()


if __name__ == "__main__":
    create_time_evolution()
    print("\nVisualization complete!")
    print("\nWhat the visualization shows:")
    print("1. Particles represent discrete fluid elements")
    print("2. Density is computed from nearby particles") 
    print("3. Pressure P = k(ρ - ρ₀) creates repulsive forces")
    print("4. Water blob falls and spreads when hitting platform")
    print("\nKey observation: In uniform density regions,")
    print("pressure gradient is ~0, not ρg as needed for equilibrium!")