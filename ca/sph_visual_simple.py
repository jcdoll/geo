"""Simple SPH visualization showing the key concepts."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle


def create_sph_concept_figure():
    """Create a figure showing SPH concepts."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create 4 subplots showing SPH concepts
    ax1 = plt.subplot(2, 2, 1)  # SPH particles and kernels
    ax2 = plt.subplot(2, 2, 2)  # Density calculation
    ax3 = plt.subplot(2, 2, 3)  # Pressure forces
    ax4 = plt.subplot(2, 2, 4)  # Time evolution
    
    # 1. SPH Particles and Smoothing Kernels
    ax1.set_title('SPH Concept: Particles with Smoothing Kernels', fontsize=14)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    
    # Draw some particles
    particles = np.array([[3, 5], [4, 6], [5, 5], [4, 4], [6, 5.5]])
    particle_types = ['water', 'water', 'water', 'air', 'rock']
    colors = {'water': 'blue', 'air': 'lightblue', 'rock': 'brown'}
    
    # Draw smoothing radius for center particle
    center_idx = 0
    h = 2.0  # smoothing radius
    circle = Circle(particles[center_idx], h, fill=False, 
                   edgecolor='red', linewidth=2, linestyle='--',
                   label=f'Smoothing radius h={h}')
    ax1.add_patch(circle)
    
    # Draw particles
    for i, (pos, ptype) in enumerate(zip(particles, particle_types)):
        ax1.scatter(pos[0], pos[1], c=colors[ptype], s=200, 
                   edgecolors='black', linewidth=2, zorder=5)
        ax1.text(pos[0], pos[1], str(i+1), ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # Add kernel visualization
    x = np.linspace(0, 10, 100)
    y = 5 + 2 * np.exp(-((x-particles[center_idx, 0])**2) / (h/2)**2)
    ax1.plot(x, y, 'r-', alpha=0.5, linewidth=2, label='Kernel function W(r)')
    
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    
    # Add text
    ax1.text(0.5, 9.5, 'Each particle influences neighbors\nwithin radius h',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Density Calculation
    ax2.set_title('Density Calculation: ρᵢ = Σⱼ mⱼW(|rᵢ-rⱼ|)', fontsize=14)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    
    # Show density calculation for center particle
    for i, (pos, ptype) in enumerate(zip(particles, particle_types)):
        color = colors[ptype]
        if i == center_idx:
            ax2.scatter(pos[0], pos[1], c=color, s=300, 
                       edgecolors='red', linewidth=3, zorder=5)
        else:
            # Check if within radius
            dist = np.linalg.norm(pos - particles[center_idx])
            if dist < h:
                # Draw connection
                ax2.plot([particles[center_idx, 0], pos[0]], 
                        [particles[center_idx, 1], pos[1]], 
                        'k--', alpha=0.5)
                ax2.scatter(pos[0], pos[1], c=color, s=200, 
                           edgecolors='green', linewidth=2, zorder=4)
                # Add weight text
                weight = np.exp(-(dist**2) / (h/2)**2)
                mid_x = (particles[center_idx, 0] + pos[0]) / 2
                mid_y = (particles[center_idx, 1] + pos[1]) / 2
                ax2.text(mid_x, mid_y, f'w={weight:.2f}', 
                        fontsize=9, ha='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                ax2.scatter(pos[0], pos[1], c=color, s=100, alpha=0.3)
    
    # Add density value
    ax2.text(particles[center_idx, 0], particles[center_idx, 1] - 1.5, 
             'ρ = 850 kg/m³', ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    
    # 3. Pressure Forces
    ax3.set_title('Pressure Forces: F = -∇P = -k∇ρ', fontsize=14)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_aspect('equal')
    
    # Create a density gradient scenario
    water_particles = np.array([[2, 5], [3, 5], [4, 5], [5, 5], [6, 5]])
    air_particles = np.array([[7, 5], [8, 5]])
    
    # Draw particles
    for pos in water_particles:
        ax3.scatter(pos[0], pos[1], c='blue', s=200, 
                   edgecolors='black', linewidth=1)
    for pos in air_particles:
        ax3.scatter(pos[0], pos[1], c='lightblue', s=100,
                   edgecolors='black', linewidth=1)
    
    # Draw pressure gradient
    x_grad = np.linspace(1, 9, 20)
    pressure = 1000 * np.exp(-(x_grad - 4)**2 / 4)
    
    # Plot pressure as background
    X, Y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    P = 1000 * np.exp(-(X - 4)**2 / 4)
    contour = ax3.contourf(X, Y, P, levels=20, cmap='YlOrRd', alpha=0.5)
    
    # Add force arrows
    for i, pos in enumerate(water_particles):
        if i < len(water_particles) - 1:
            # Pressure gradient force (pointing right, away from high pressure)
            ax3.arrow(pos[0], pos[1], 0.5, 0, head_width=0.3, 
                     head_length=0.2, fc='red', ec='red', linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label('Pressure (Pa)')
    
    ax3.text(1, 8, 'High pressure\npushes particles\napart', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.text(7, 8, 'Low pressure\nregion', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.grid(True, alpha=0.3)
    
    # 4. Time Evolution
    ax4.set_title('Problem: In Uniform Density, ∇P = 0 not ρg!', fontsize=14)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.set_aspect('equal')
    
    # Show water column
    water_y = [2, 3, 4, 5, 6, 7]
    for y in water_y:
        for x in [3, 4, 5, 6, 7]:
            ax4.scatter(x, y, c='blue', s=150, edgecolors='black', linewidth=1)
    
    # Add gravity arrows
    for y in water_y[::2]:
        for x in [3, 5, 7]:
            ax4.arrow(x, y, 0, -0.4, head_width=0.2, head_length=0.1,
                     fc='green', ec='green', linewidth=1.5, alpha=0.7)
    
    # Add text annotations
    ax4.text(5, 8.5, 'Uniform water column', ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.text(1, 4, 'Gravity\nρg', ha='center', color='green', fontweight='bold')
    ax4.text(9, 4, 'Pressure\ngradient\n∇P ≈ 0!', ha='center', color='red', fontweight='bold')
    
    # Show the problem
    ax4.text(5, 0.5, 'Problem: With P = k(ρ-ρ₀), uniform density → ∇P = 0\nBut we need ∇P = ρg for equilibrium!',
             ha='center', fontsize=12, color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_sph_vs_grid_comparison():
    """Compare SPH and grid approaches."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grid approach
    ax1.set_title('Grid-Based Approach', fontsize=16)
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 10.5)
    ax1.set_aspect('equal')
    
    # Draw grid
    for i in range(11):
        ax1.axhline(i, color='gray', linewidth=0.5)
        ax1.axvline(i, color='gray', linewidth=0.5)
    
    # Fill some cells with water
    water_cells = [(3, 3), (4, 3), (5, 3), (6, 3),
                   (3, 4), (4, 4), (5, 4), (6, 4),
                   (3, 5), (4, 5), (5, 5), (6, 5)]
    
    for x, y in water_cells:
        rect = plt.Rectangle((x, y), 1, 1, facecolor='blue', alpha=0.6)
        ax1.add_patch(rect)
        ax1.text(x+0.5, y+0.5, 'W', ha='center', va='center', color='white', fontweight='bold')
    
    # Add air cells
    air_cells = [(3, 6), (4, 6), (5, 6), (6, 6)]
    for x, y in air_cells:
        rect = plt.Rectangle((x, y), 1, 1, facecolor='lightblue', alpha=0.6)
        ax1.add_patch(rect)
        ax1.text(x+0.5, y+0.5, 'A', ha='center', va='center', color='black')
    
    ax1.text(5, 8, 'Solve: ∇²P = ∇·(ρg)', ha='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='yellow'))
    ax1.text(5, 1, 'Problem: ∇P ≈ 0 in bulk water', ha='center', color='red', fontsize=12)
    
    ax1.set_xlabel('X (cells)')
    ax1.set_ylabel('Y (cells)')
    
    # SPH approach
    ax2.set_title('SPH (Particle) Approach', fontsize=16)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    
    # Draw particles
    np.random.seed(42)
    n_water = 50
    water_pos = np.random.uniform([3, 3], [7, 6], (n_water, 2))
    ax2.scatter(water_pos[:, 0], water_pos[:, 1], c='blue', s=100, 
               edgecolors='black', linewidth=1, alpha=0.8)
    
    n_air = 20
    air_pos = np.random.uniform([3, 6], [7, 7], (n_air, 2))
    ax2.scatter(air_pos[:, 0], air_pos[:, 1], c='lightblue', s=50,
               edgecolors='black', linewidth=1, alpha=0.8)
    
    # Show one particle's influence
    center = water_pos[25]
    circle = Circle(center, 1.5, fill=False, edgecolor='red', 
                   linewidth=2, linestyle='--')
    ax2.add_patch(circle)
    
    ax2.text(5, 8, 'P = k(ρ - ρ₀)', ha='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='yellow'))
    ax2.text(5, 1, 'Same problem: ∇P ≈ 0 in bulk', ha='center', color='red', fontsize=12)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Create concept figure
    print("Creating SPH concept visualization...")
    fig1 = create_sph_concept_figure()
    plt.savefig('sph_concepts.png', dpi=150, bbox_inches='tight')
    print("Saved sph_concepts.png")
    plt.close()
    
    # Create comparison figure
    print("Creating SPH vs Grid comparison...")
    fig2 = create_sph_vs_grid_comparison()
    plt.savefig('sph_vs_grid.png', dpi=150, bbox_inches='tight')
    print("Saved sph_vs_grid.png")
    plt.close()
    
    print("\nVisualization complete!")
    print("\nKey SPH concepts shown:")
    print("1. Particles carry properties (mass, velocity, etc)")
    print("2. Density computed from nearby particles within radius h")
    print("3. Pressure P = k(ρ - ρ₀) creates repulsive forces")
    print("4. Forces computed from pressure gradients")
    print("\nBUT: Same fundamental problem as grid!")
    print("In uniform density regions, ∇P = k∇ρ = 0")
    print("We need ∇P = ρg for hydrostatic equilibrium!")