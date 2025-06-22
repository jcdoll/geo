"""Explain what actually happens in SPH steady state."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches


def create_steady_state_explanation():
    """Show what actually happens vs ideal hydrostatic equilibrium."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Title
    fig.suptitle('SPH Narrow Container: What Actually Happens vs Ideal', fontsize=16, fontweight='bold')
    
    # === Top row: What Actually Happens ===
    # Stage 1: Initial fall
    ax = axes[0, 0]
    ax.set_title('Actual: Initial State', fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # Container
    container = Rectangle((4, 0), 2, 7, fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(container)
    
    # Falling particles
    np.random.seed(42)
    x = np.random.uniform(4.2, 5.8, 50)
    y = np.random.uniform(5, 8, 50)
    ax.scatter(x, y, c='blue', s=60, alpha=0.8, edgecolors='darkblue')
    
    # Velocity arrows
    for i in range(0, 50, 5):
        ax.arrow(x[i], y[i], 0, -0.5, head_width=0.1, head_length=0.05,
                fc='red', ec='red', alpha=0.6)
    
    ax.text(5, 9, 'Falling', ha='center', fontweight='bold')
    
    # Stage 2: Compression
    ax = axes[0, 1]
    ax.set_title('Actual: Compression Phase', fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    container = Rectangle((4, 0), 2, 7, fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(container)
    
    # Compressed particles at bottom
    y_compressed = []
    x_compressed = []
    # Bottom layers - very compressed
    for layer in range(6):
        n_layer = 10 - layer  # More particles in bottom layers
        x_layer = np.random.uniform(4.1, 5.9, n_layer)
        y_layer = np.random.uniform(0.2 + layer*0.3, 0.2 + (layer+1)*0.3, n_layer)
        x_compressed.extend(x_layer)
        y_compressed.extend(y_layer)
    
    x_compressed = np.array(x_compressed)
    y_compressed = np.array(y_compressed)
    
    # Color by local density (more red = higher density)
    density_colors = y_compressed.max() - y_compressed  # Higher at bottom
    scatter = ax.scatter(x_compressed, y_compressed, c=density_colors, 
                        cmap='Blues_r', s=60, alpha=0.8, edgecolors='darkblue')
    
    # Show compression forces
    ax.arrow(3, 1, 0.5, 0, head_width=0.2, head_length=0.1,
            fc='orange', ec='orange', linewidth=2)
    ax.arrow(7, 1, -0.5, 0, head_width=0.2, head_length=0.1,
            fc='orange', ec='orange', linewidth=2)
    ax.text(5, 0.5, 'High local density\n→ Repulsive forces', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Stage 3: Quasi-steady state
    ax = axes[0, 2]
    ax.set_title('Actual: Quasi-Steady State', fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    container = Rectangle((4, 0), 2, 7, fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(container)
    
    # More uniform distribution
    y_steady = []
    x_steady = []
    for layer in range(10):
        x_layer = np.random.uniform(4.2, 5.8, 5)
        y_layer = np.random.uniform(0.3 + layer*0.5, 0.3 + (layer+1)*0.5, 5)
        x_steady.extend(x_layer)
        y_steady.extend(y_layer)
    
    ax.scatter(x_steady[:50], y_steady[:50], c='blue', s=60, alpha=0.8, edgecolors='darkblue')
    
    # Small oscillations
    for i in [10, 20, 30]:
        ax.arrow(x_steady[i], y_steady[i], 0, 0.1, head_width=0.05, head_length=0.02,
                fc='red', ec='red', alpha=0.4)
        ax.arrow(x_steady[i], y_steady[i], 0, -0.1, head_width=0.05, head_length=0.02,
                fc='red', ec='red', alpha=0.4)
    
    ax.text(7, 5, 'Small oscillations\n& numerical noise', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
    
    # === Bottom row: Ideal Hydrostatic ===
    # Stage 1: Initial
    ax = axes[1, 0]
    ax.set_title('Ideal Hydrostatic: Initial', fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    container = Rectangle((4, 0), 2, 7, fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(container)
    
    # Same falling particles
    ax.scatter(x, y, c='blue', s=60, alpha=0.8, edgecolors='darkblue')
    
    # Stage 2: Perfect pressure gradient
    ax = axes[1, 1]
    ax.set_title('Ideal Hydrostatic: Perfect Gradient', fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    container = Rectangle((4, 0), 2, 7, fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(container)
    
    # Uniform distribution with pressure gradient
    y_ideal = np.linspace(0.5, 5.5, 50)
    x_ideal = np.random.uniform(4.2, 5.8, 50)
    
    # Color by hydrostatic pressure
    pressure_ideal = 1000 + 1000 * 9.81 * (5.5 - y_ideal)
    scatter2 = ax.scatter(x_ideal, y_ideal, c=pressure_ideal, cmap='YlOrRd',
                         s=60, alpha=0.8, edgecolors='darkblue',
                         vmin=1000, vmax=50000)
    
    # Show balanced forces
    for y_pos in [1, 2.5, 4]:
        # Gravity
        ax.arrow(3.5, y_pos, 0, -0.3, head_width=0.1, head_length=0.05,
                fc='green', ec='green', linewidth=1.5)
        # Pressure gradient
        ax.arrow(3.5, y_pos-0.05, 0, 0.3, head_width=0.1, head_length=0.05,
                fc='red', ec='red', linewidth=1.5)
    
    ax.text(2.5, 2.5, 'ρg', color='green', fontweight='bold', ha='center')
    ax.text(2.5, 3, '∇P', color='red', fontweight='bold', ha='center')
    
    # Stage 3: True equilibrium
    ax = axes[1, 2]
    ax.set_title('Ideal Hydrostatic: True Equilibrium', fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    container = Rectangle((4, 0), 2, 7, fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(container)
    
    # Same distribution, no motion
    ax.scatter(x_ideal, y_ideal, c=pressure_ideal, cmap='YlOrRd',
              s=60, alpha=0.8, edgecolors='darkblue',
              vmin=1000, vmax=50000)
    
    ax.text(5, 7, 'Perfectly stationary\n∇P = ρg everywhere', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Add comparison text
    ax.text(5, -0.5, 'TRUE HYDROSTATIC EQUILIBRIUM\n(Not achieved by SPH)', 
            ha='center', fontweight='bold', color='green')
    
    # Add explanation boxes
    axes[0, 1].text(5, -0.5, 'WHAT SPH ACTUALLY DOES\n(Quasi-equilibrium)', 
                    ha='center', fontweight='bold', color='red')
    
    plt.tight_layout()
    return fig


def create_force_balance_diagram():
    """Show the force balance issue more clearly."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # SPH Forces
    ax1.set_title('SPH in "Steady State"', fontsize=14)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    
    # Container with particles
    container = Rectangle((3, 1), 4, 6, fill=False, edgecolor='black', linewidth=3)
    ax1.add_patch(container)
    
    # Draw particle layers
    for layer in range(5):
        y = 2 + layer
        for x in np.linspace(3.5, 6.5, 7):
            ax1.scatter(x, y, c='blue', s=100, edgecolors='darkblue')
    
    # Force diagram for middle particle
    center_x, center_y = 5, 4
    
    # Gravity (large)
    ax1.arrow(center_x, center_y, 0, -1.0, head_width=0.2, head_length=0.1,
             fc='green', ec='green', linewidth=3)
    ax1.text(center_x-0.5, center_y-0.5, 'ρg\n(9810 N/m³)', ha='right', 
             color='green', fontweight='bold')
    
    # SPH pressure forces (small, from neighbors)
    # From above
    ax1.arrow(center_x, center_y+0.5, 0, -0.2, head_width=0.1, head_length=0.05,
             fc='red', ec='red', linewidth=2, alpha=0.7)
    # From below  
    ax1.arrow(center_x, center_y-0.5, 0, 0.2, head_width=0.1, head_length=0.05,
             fc='red', ec='red', linewidth=2, alpha=0.7)
    # From sides (cancel out)
    ax1.arrow(center_x-0.5, center_y, 0.1, 0, head_width=0.05, head_length=0.03,
             fc='orange', ec='orange', linewidth=1, alpha=0.5)
    ax1.arrow(center_x+0.5, center_y, -0.1, 0, head_width=0.05, head_length=0.03,
             fc='orange', ec='orange', linewidth=1, alpha=0.5)
    
    ax1.text(center_x+1, center_y, 'Small SPH\nforces', ha='left', color='red')
    
    # Net force arrow
    ax1.arrow(center_x+1.5, center_y, 0, -0.8, head_width=0.2, head_length=0.1,
             fc='purple', ec='purple', linewidth=3)
    ax1.text(center_x+2, center_y-0.5, 'Net ≠ 0!', ha='left', 
             color='purple', fontweight='bold', fontsize=12)
    
    # Add explanation
    ax1.text(5, 8, 'In uniform density:\n∇P ≈ 0, not ρg', ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # What really happens
    ax2.set_title('What Prevents Collapse', fontsize=14)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    
    # Container
    container2 = Rectangle((3, 0), 4, 7, fill=False, edgecolor='black', linewidth=3)
    ax2.add_patch(container2)
    
    # Bottom particles compressed
    # Very dense bottom layer
    for x in np.linspace(3.2, 6.8, 15):
        ax2.scatter(x, 0.3, c='darkblue', s=120, edgecolors='black', alpha=0.9)
    
    # Less dense layers above
    densities = [12, 10, 8, 6, 5]
    sizes = [100, 90, 80, 70, 60]
    for layer, (n, size) in enumerate(zip(densities, sizes)):
        y = 1 + layer * 0.8
        x_pos = np.linspace(3.5, 6.5, n)
        ax2.scatter(x_pos, [y]*n, c='blue', s=size, edgecolors='darkblue', 
                   alpha=0.8-layer*0.1)
    
    # Show forces at bottom
    ax2.arrow(5, 0.3, 0, 0.5, head_width=0.3, head_length=0.1,
             fc='red', ec='red', linewidth=3)
    ax2.text(5.5, 0.5, 'High density\n→ Big repulsion', ha='left', color='red')
    
    # Ground reaction
    ax2.arrow(4, -0.1, 0, 0.3, head_width=0.2, head_length=0.1,
             fc='brown', ec='brown', linewidth=2)
    ax2.arrow(6, -0.1, 0, 0.3, head_width=0.2, head_length=0.1,
             fc='brown', ec='brown', linewidth=2)
    ax2.text(5, -0.3, 'Ground reaction', ha='center', color='brown')
    
    # Explanation
    ax2.text(5, 8.5, 'Prevents total collapse:', ha='center', fontweight='bold')
    ax2.text(5, 8, '1. Ground provides reaction force', ha='center')
    ax2.text(5, 7.5, '2. Bottom particles get compressed', ha='center')
    ax2.text(5, 7, '3. High density → repulsion', ha='center')
    ax2.text(5, 6.5, '4. System oscillates/settles', ha='center')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Create steady state explanation
    print("Creating steady state explanation...")
    fig1 = create_steady_state_explanation()
    plt.savefig('sph_steady_state_explanation.png', dpi=150, bbox_inches='tight')
    print("Saved sph_steady_state_explanation.png")
    plt.close()
    
    # Create force balance diagram
    print("Creating force balance diagram...")
    fig2 = create_force_balance_diagram()
    plt.savefig('sph_force_balance.png', dpi=150, bbox_inches='tight')
    print("Saved sph_force_balance.png")
    plt.close()
    
    print("\nExplanation complete!")
    print("\nWhat actually happens in SPH:")
    print("1. Particles fall and compress at bottom")
    print("2. High compression → high local density")  
    print("3. High density → strong repulsive forces")
    print("4. System reaches quasi-equilibrium with:")
    print("   - Density gradient (higher at bottom)")
    print("   - Small oscillations")
    print("   - Numerical damping")
    print("\nIt LOOKS stable but isn't true hydrostatic equilibrium!")
    print("True equilibrium would have ∇P = ρg everywhere.")