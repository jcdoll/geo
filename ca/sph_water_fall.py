"""Simple visualization of water falling in SPH."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def create_water_fall_sequence():
    """Show water blob falling and splashing."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    # Common settings
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Draw ground
        ground = Rectangle((0, 0), 10, 1, facecolor='brown', alpha=0.8)
        ax.add_patch(ground)
        ax.text(5, 0.5, 'Ground', ha='center', va='center', color='white', fontweight='bold')
    
    # Time 1: Initial blob
    ax = axes[0]
    ax.set_title('t = 0.0s: Initial Water Blob', fontsize=14)
    
    # Draw water particles in circular blob
    np.random.seed(42)
    n_particles = 100
    center = [5, 7]
    radius = 1.5
    
    # Generate particles in circle
    angles = np.random.uniform(0, 2*np.pi, n_particles)
    radii = radius * np.sqrt(np.random.uniform(0, 1, n_particles))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    
    ax.scatter(x, y, c='blue', s=50, alpha=0.8, edgecolors='darkblue')
    
    # Add velocity arrows
    for i in range(0, n_particles, 10):
        ax.arrow(x[i], y[i], 0, -0.3, head_width=0.1, head_length=0.05,
                fc='red', ec='red', alpha=0.6)
    
    ax.text(5, 9, 'Water particles\nwith downward velocity', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Time 2: Falling
    ax = axes[1]
    ax.set_title('t = 0.3s: Falling', fontsize=14)
    
    # Particles have fallen and spread slightly
    y_fall = y - 2.5
    x_spread = x + np.random.normal(0, 0.2, n_particles)
    
    ax.scatter(x_spread, y_fall, c='blue', s=50, alpha=0.8, edgecolors='darkblue')
    
    # Show some particle interactions
    for i in range(5):
        j = i + 20
        if j < n_particles:
            ax.plot([x_spread[i], x_spread[j]], [y_fall[i], y_fall[j]], 
                   'gray', alpha=0.3, linewidth=1)
    
    ax.text(2, 7, 'Particles\ninteract via\npressure forces', ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Time 3: Impact
    ax = axes[2]
    ax.set_title('t = 0.5s: Impact!', fontsize=14)
    
    # Particles hit ground and spread
    y_impact = np.maximum(1.2, y_fall - 2)
    x_impact = x_spread + np.where(y_impact < 2, 
                                   np.random.normal(0, 1, n_particles), 
                                   0)
    
    # Color by speed
    speeds = np.sqrt((x_impact - x_spread)**2 + (y_impact - y_fall)**2)
    scatter = ax.scatter(x_impact, y_impact, c=speeds, s=60, 
                        cmap='Blues_r', alpha=0.8, edgecolors='darkblue',
                        vmin=0, vmax=2)
    
    # Show pressure waves
    circle1 = Circle([5, 1.5], 2, fill=False, edgecolor='red', 
                    linewidth=2, alpha=0.5, linestyle='--')
    ax.add_patch(circle1)
    
    ax.text(8, 3, 'High pressure\nat impact!', ha='center', color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Time 4: Spread
    ax = axes[3]
    ax.set_title('t = 1.0s: Equilibrium', fontsize=14)
    
    # Particles have spread into puddle
    x_final = np.clip(x_impact + np.random.normal(0, 0.5, n_particles), 1, 9)
    y_final = np.ones_like(x_final) * 1.3 + np.random.uniform(-0.1, 0.3, n_particles)
    
    ax.scatter(x_final, y_final, c='blue', s=50, alpha=0.8, edgecolors='darkblue')
    
    # Show equilibrium issue
    ax.text(5, 5, 'Problem: In this uniform puddle,\n∇P ≈ 0 instead of ρg!', 
            ha='center', color='red', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
    
    # Add arrow showing gravity but no pressure gradient
    ax.arrow(3, 2, 0, -0.5, head_width=0.2, head_length=0.1,
            fc='green', ec='green', linewidth=2)
    ax.text(3, 2.5, 'ρg', ha='center', color='green', fontweight='bold')
    
    ax.arrow(7, 2, 0, -0.1, head_width=0.2, head_length=0.05,
            fc='red', ec='red', linewidth=2)
    ax.text(7, 2.5, '∇P≈0', ha='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_sph_algorithm_flow():
    """Show the SPH algorithm flow."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'SPH Algorithm Flow', ha='center', fontsize=18, fontweight='bold')
    
    # Boxes for steps
    steps = [
        (5, 8, "1. Find Neighbors\nFor each particle, find all\nparticles within radius h"),
        (5, 6.5, "2. Compute Density\nρᵢ = Σⱼ mⱼ W(|rᵢ - rⱼ|)"),
        (5, 5, "3. Compute Pressure\nP = k(ρ - ρ₀)"),
        (5, 3.5, "4. Compute Forces\nF = -∇P + ρg"),
        (5, 2, "5. Update Velocity & Position\nv += F/ρ · dt, x += v · dt"),
    ]
    
    for x, y, text in steps:
        # Draw box
        box = Rectangle((x-2, y-0.6), 4, 1.2, 
                       facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11)
        
        # Draw arrow to next step
        if y > 2:
            ax.arrow(x, y-0.6, 0, -0.3, head_width=0.2, head_length=0.1,
                    fc='black', ec='black')
    
    # Add feedback arrow
    ax.arrow(7, 2, 0, 5.5, head_width=0.2, head_length=0.1,
            fc='gray', ec='gray', linewidth=2)
    ax.text(7.5, 5, 'Repeat', ha='left', va='center', color='gray', fontsize=12)
    
    # Add notes
    ax.text(1, 0.5, 'Key Issue: Step 3 gives ∇P = k∇ρ = 0 in uniform density!',
            color='red', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    return fig


if __name__ == "__main__":
    # Create water fall sequence
    print("Creating water fall visualization...")
    fig1 = create_water_fall_sequence()
    plt.savefig('sph_water_fall.png', dpi=150, bbox_inches='tight')
    print("Saved sph_water_fall.png")
    plt.close()
    
    # Create algorithm flow
    print("Creating SPH algorithm flow...")
    fig2 = create_sph_algorithm_flow()
    plt.savefig('sph_algorithm.png', dpi=150, bbox_inches='tight')
    print("Saved sph_algorithm.png")
    plt.close()
    
    print("\nDone! Created visualizations showing:")
    print("1. sph_concepts.png - Core SPH concepts")
    print("2. sph_vs_grid.png - Comparison with grid approach")
    print("3. sph_water_fall.png - Water blob falling sequence")
    print("4. sph_algorithm.png - SPH algorithm flow")
    
    print("\nThe visualizations show that SPH:")
    print("- Uses particles that carry mass, velocity, etc")
    print("- Computes density from nearby particles")
    print("- Creates pressure forces between particles")
    print("- Has the SAME hydrostatic problem as grids!")