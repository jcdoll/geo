#!/usr/bin/env python3
"""Test to diagnose the Ghost Fluid Method issue with space interfaces."""

import numpy as np
import matplotlib.pyplot as plt
from geo_game import GeoGame
from materials import MaterialType, MaterialDatabase

def test_gfm_space_interface():
    """Test GFM behavior at granite-space interface."""
    # Create small simulation for detailed analysis
    sim = GeoGame(width=10, height=10, cell_size=50.0)
    
    # Initialize everything as space
    sim.material_types[:] = MaterialType.SPACE
    sim._update_material_properties()
    
    # Add a small granite block
    sim.material_types[4:7, 4:7] = MaterialType.GRANITE
    sim._update_material_properties()
    
    # Get material database
    mat_db = MaterialDatabase()
    rho_granite = mat_db.get_properties(MaterialType.GRANITE).density
    rho_space = mat_db.get_properties(MaterialType.SPACE).density
    
    print(f"Granite density: {rho_granite} kg/m³")
    print(f"Space density: {rho_space} kg/m³")
    print(f"Density ratio: {rho_granite/rho_space:.2e}")
    
    # Detect interfaces
    horiz_interface, vert_interface = sim.fluid_dynamics.detect_interfaces()
    
    print("\nHorizontal interfaces (granite-space boundaries):")
    print(horiz_interface.astype(int))
    
    # Calculate forces before any steps
    fx, fy = sim.fluid_dynamics.compute_force_field()
    
    print("\nVertical force field (fy) at interfaces:")
    print(f"Min: {np.min(fy):.3e}, Max: {np.max(fy):.3e}")
    
    # Show force field around granite block
    print("\nForce field fy around granite block:")
    print(fy[3:8, 3:8])
    
    # Show gravity field
    print("\nGravity field gy around granite block:")
    print(sim.gravity_y[3:8, 3:8])
    
    # Calculate what the GFM correction would be
    rho_avg = 0.5 * (rho_granite + rho_space)
    gfm_force = -rho_avg * sim.gravity_y[5, 5]
    print(f"\nGFM correction force: {gfm_force:.3e} N/m³")
    print(f"This is essentially: -0.5 * {rho_granite} * {sim.gravity_y[5, 5]:.3f} = {gfm_force:.3e}")
    
    # Visualize
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    
    # Material layout
    ax1.imshow(sim.material_types == MaterialType.GRANITE, cmap='gray')
    ax1.set_title('Granite Block')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Horizontal interfaces
    ax2.imshow(horiz_interface, cmap='hot')
    ax2.set_title('Horizontal Interfaces')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Vertical force field
    im3 = ax3.imshow(fy, cmap='seismic', vmin=-1e5, vmax=1e5)
    ax3.set_title('Vertical Force Field (fy)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3, label='Force (N/m³)')
    
    # Gravity field
    im4 = ax4.imshow(sim.gravity_y, cmap='viridis')
    ax4.set_title('Gravity Field (gy)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im4, ax=ax4, label='Gravity (m/s²)')
    
    plt.tight_layout()
    plt.savefig('gfm_space_interface_analysis.png', dpi=150)
    print(f"\nSaved analysis to gfm_space_interface_analysis.png")

if __name__ == "__main__":
    test_gfm_space_interface()