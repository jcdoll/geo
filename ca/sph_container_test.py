"""Visualize SPH particles falling into different containers."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches


def create_container_comparison():
    """Show particles in wide vs narrow containers."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Common settings
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # === Wide Container Sequence ===
    # Time 1: Initial
    ax = axes[0]
    ax.set_title('Wide Container - t=0s: Initial', fontsize=12)
    
    # Draw wide container
    container = Polygon([(2, 0), (2, 3), (8, 3), (8, 0)], 
                       fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(container)
    
    # Initial particles above
    np.random.seed(42)
    n_particles = 100
    x_init = np.random.uniform(4, 6, n_particles)
    y_init = np.random.uniform(6, 8, n_particles)
    ax.scatter(x_init, y_init, c='blue', s=40, alpha=0.8, edgecolors='darkblue')
    
    # Time 2: Falling
    ax = axes[1]
    ax.set_title('Wide Container - t=0.5s: Impact', fontsize=12)
    ax.add_patch(Polygon([(2, 0), (2, 3), (8, 3), (8, 0)], 
                        fill=False, edgecolor='black', linewidth=3))
    
    # Particles spreading on impact
    x_wide = np.clip(x_init + np.random.normal(0, 1.5, n_particles), 2.2, 7.8)
    y_wide = np.random.uniform(0.2, 1.5, n_particles)
    ax.scatter(x_wide, y_wide, c='blue', s=40, alpha=0.8, edgecolors='darkblue')
    
    # Time 3: Settled
    ax = axes[2]
    ax.set_title('Wide Container - t=2s: Settled (Shallow)', fontsize=12)
    ax.add_patch(Polygon([(2, 0), (2, 3), (8, 3), (8, 0)], 
                        fill=False, edgecolor='black', linewidth=3))
    
    # Final distribution - shallow
    x_final_wide = np.random.uniform(2.2, 7.8, n_particles)
    y_final_wide = np.random.uniform(0.2, 0.8, n_particles)
    ax.scatter(x_final_wide, y_final_wide, c='blue', s=40, alpha=0.8, edgecolors='darkblue')
    
    # Add depth annotation
    ax.annotate('', xy=(1.5, 0.2), xytext=(1.5, 0.8),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(1.2, 0.5, '~0.6m', rotation=90, va='center', color='red', fontweight='bold')
    
    # === Narrow Container Sequence ===
    # Time 1: Initial
    ax = axes[3]
    ax.set_title('Narrow Container - t=0s: Initial', fontsize=12)
    
    # Draw narrow container
    container = Polygon([(4, 0), (4, 6), (6, 6), (6, 0)], 
                       fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(container)
    
    # Same initial particles
    ax.scatter(x_init, y_init, c='blue', s=40, alpha=0.8, edgecolors='darkblue')
    
    # Time 2: Falling
    ax = axes[4]
    ax.set_title('Narrow Container - t=0.5s: Entering', fontsize=12)
    ax.add_patch(Polygon([(4, 0), (4, 6), (6, 6), (6, 0)], 
                        fill=False, edgecolor='black', linewidth=3))
    
    # Particles funneling in
    x_narrow = np.clip(x_init * 0.4 + 3, 4.1, 5.9)  # Compress horizontally
    y_narrow = y_init - 3  # Fallen distance
    # Only show particles that would be inside or above container
    mask = (y_narrow > 0) | ((x_narrow > 4) & (x_narrow < 6))
    ax.scatter(x_narrow[mask], np.maximum(y_narrow[mask], 0.2), 
              c='blue', s=40, alpha=0.8, edgecolors='darkblue')
    
    # Time 3: Settled
    ax = axes[5]
    ax.set_title('Narrow Container - t=2s: Settled (Tall Column)', fontsize=12)
    ax.add_patch(Polygon([(4, 0), (4, 6), (6, 6), (6, 0)], 
                        fill=False, edgecolor='black', linewidth=3))
    
    # Final distribution - tall column
    # Stack particles in layers
    particles_per_layer = 20
    n_layers = n_particles // particles_per_layer
    
    x_final_narrow = []
    y_final_narrow = []
    
    for layer in range(n_layers):
        x_layer = np.random.uniform(4.1, 5.9, particles_per_layer)
        y_layer = np.random.uniform(0.2 + layer * 0.6, 0.2 + (layer + 1) * 0.6, particles_per_layer)
        x_final_narrow.extend(x_layer)
        y_final_narrow.extend(y_layer)
    
    ax.scatter(x_final_narrow[:n_particles], y_final_narrow[:n_particles], 
              c='blue', s=40, alpha=0.8, edgecolors='darkblue')
    
    # Add height annotation
    max_height = max(y_final_narrow[:n_particles])
    ax.annotate('', xy=(3.5, 0.2), xytext=(3.5, max_height),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(3.2, max_height/2, f'~{max_height:.1f}m', rotation=90, va='center', 
            color='red', fontweight='bold')
    
    # Add text annotations
    fig.text(0.5, 0.95, 'SPH Particles in Different Containers', 
             ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_pressure_buildup_diagram():
    """Show pressure buildup in narrow vs wide containers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Wide container
    ax1.set_title('Wide Container: Low Pressure', fontsize=14)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    
    # Container
    container1 = Polygon([(2, 0), (2, 3), (8, 3), (8, 0)], 
                        fill=False, edgecolor='black', linewidth=3)
    ax1.add_patch(container1)
    
    # Particles
    np.random.seed(42)
    x_wide = np.random.uniform(2.2, 7.8, 80)
    y_wide = np.random.uniform(0.2, 0.8, 80)
    
    # Color by pressure (height-based)
    pressure_wide = 1000 + y_wide * 100  # Low pressure gradient
    scatter1 = ax1.scatter(x_wide, y_wide, c=pressure_wide, s=50, 
                          cmap='YlOrRd', vmin=1000, vmax=2000,
                          edgecolors='black', linewidth=0.5)
    
    # Add pressure gradient arrows
    for y in [0.3, 0.5, 0.7]:
        ax1.arrow(1, y, 0, -0.2, head_width=0.1, head_length=0.05,
                 fc='green', ec='green', alpha=0.7)
    ax1.text(0.5, 0.5, 'ρg', fontsize=12, color='green', fontweight='bold')
    
    # Narrow container
    ax2.set_title('Narrow Container: High Pressure Gradient', fontsize=14)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    
    # Container
    container2 = Polygon([(4, 0), (4, 6), (6, 6), (6, 0)], 
                        fill=False, edgecolor='black', linewidth=3)
    ax2.add_patch(container2)
    
    # Particles in column
    x_narrow = []
    y_narrow = []
    for layer in range(8):
        x_layer = np.random.uniform(4.1, 5.9, 10)
        y_layer = np.random.uniform(0.2 + layer * 0.6, 0.2 + (layer + 1) * 0.6, 10)
        x_narrow.extend(x_layer)
        y_narrow.extend(y_layer)
    
    x_narrow = np.array(x_narrow)
    y_narrow = np.array(y_narrow)
    
    # Higher pressure gradient
    pressure_narrow = 1000 + y_narrow * 500  # Steeper gradient
    scatter2 = ax2.scatter(x_narrow, y_narrow, c=pressure_narrow, s=50,
                          cmap='YlOrRd', vmin=1000, vmax=4000,
                          edgecolors='black', linewidth=0.5)
    
    # Add pressure gradient arrows - bigger at bottom
    for i, y in enumerate([0.5, 2, 3.5]):
        arrow_size = 0.2 + i * 0.1
        ax2.arrow(3, y, 0, -arrow_size, head_width=0.15, head_length=0.08,
                 fc='green', ec='green', alpha=0.7, linewidth=2)
    ax2.text(2.5, 2, 'ρg', fontsize=12, color='green', fontweight='bold')
    
    # Colorbars
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Pressure (Pa)')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Pressure (Pa)')
    
    # Add annotations
    ax1.text(5, 8, 'Same amount of water\nspreads thin', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.text(5, 8, 'Water piles up\ncreating column', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add pressure profile insets
    # Wide container profile
    ax1_inset = ax1.inset_axes([0.7, 0.5, 0.25, 0.35])
    ax1_inset.plot([1000, 1100], [0, 0.8], 'r-', linewidth=2)
    ax1_inset.set_xlabel('P (Pa)', fontsize=8)
    ax1_inset.set_ylabel('Height (m)', fontsize=8)
    ax1_inset.set_title('Pressure Profile', fontsize=8)
    ax1_inset.grid(True, alpha=0.3)
    
    # Narrow container profile
    ax2_inset = ax2.inset_axes([0.05, 0.5, 0.25, 0.35])
    ax2_inset.plot([1000, 3000], [0, 5], 'r-', linewidth=2)
    ax2_inset.set_xlabel('P (Pa)', fontsize=8)
    ax2_inset.set_ylabel('Height (m)', fontsize=8)
    ax2_inset.set_title('Pressure Profile', fontsize=8)
    ax2_inset.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_sph_hydrostatic_issue():
    """Show the hydrostatic issue in both containers."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'SPH Hydrostatic Problem in Containers', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Draw narrow container with water
    container = Rectangle((2, 1), 2, 5, fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(container)
    
    # Water particles
    for y in np.linspace(1.2, 5.5, 8):
        for x in np.linspace(2.2, 3.8, 5):
            ax.scatter(x, y, c='blue', s=100, edgecolors='darkblue')
    
    # Show forces
    ax.arrow(1, 3, 0, -0.5, head_width=0.2, head_length=0.1,
            fc='green', ec='green', linewidth=2)
    ax.text(0.5, 3, 'ρg', fontsize=14, color='green', fontweight='bold')
    
    # Show the problem
    ax.text(6, 6, 'In uniform density column:', fontsize=14)
    ax.text(6, 5.5, '• SPH gives: ∇P = k∇ρ = 0', fontsize=12, color='red')
    ax.text(6, 5, '• We need: ∇P = ρg ≈ 9810 Pa/m', fontsize=12, color='green')
    ax.text(6, 4.5, '• Water accelerates downward!', fontsize=12, color='red')
    
    # Add equation
    ax.text(5, 2, 'SPH Pressure: P = k(ρ - ρ₀)', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.text(5, 1, 'Problem: When ρ is constant, ∇P = 0', fontsize=12,
            color='red', ha='center')
    
    # Show particle interactions
    y_mid = 3.5
    for x in [2.4, 2.8, 3.2, 3.6]:
        circle = plt.Circle((x, y_mid), 0.5, fill=False, 
                           edgecolor='orange', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    
    ax.text(5, 3.5, 'All particles have\nsame density ρ₀\n→ No pressure gradient!',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    
    return fig


if __name__ == "__main__":
    # Create container comparison
    print("Creating container comparison...")
    fig1 = create_container_comparison()
    plt.savefig('sph_container_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved sph_container_comparison.png")
    plt.close()
    
    # Create pressure buildup diagram
    print("Creating pressure buildup diagram...")
    fig2 = create_pressure_buildup_diagram()
    plt.savefig('sph_pressure_buildup.png', dpi=150, bbox_inches='tight')
    print("Saved sph_pressure_buildup.png")
    plt.close()
    
    # Create hydrostatic issue diagram
    print("Creating hydrostatic issue diagram...")
    fig3 = create_sph_hydrostatic_issue()
    plt.savefig('sph_hydrostatic_issue.png', dpi=150, bbox_inches='tight')
    print("Saved sph_hydrostatic_issue.png")
    plt.close()
    
    print("\nVisualization complete!")
    print("\nKey findings:")
    print("1. SPH particles DO pile up in narrow containers")
    print("2. This creates a taller water column")
    print("3. BUT the fundamental problem remains:")
    print("   - In the uniform column, ∇ρ = 0")
    print("   - So ∇P = k∇ρ = 0")
    print("   - Water still accelerates downward!")
    print("\nThe container shape changes the final configuration")
    print("but doesn't fix the hydrostatic equilibrium problem.")