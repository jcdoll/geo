"""Test simple rigid body container movement with fluid preservation"""

import numpy as np
import pytest
from scipy import ndimage
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


def test_simple_container_fall():
    """Test that a simple container preserves fluid temperature when falling"""
    # Create small simulation
    sim = GeoSimulation(width=20, height=20, cell_size=100)
    sim.external_gravity = (0, 5)  # Gentle gravity
    sim.enable_self_gravity = False
    sim.enable_solid_drag = True  # Keep damping for stability
    sim.debug_rigid_bodies = True
    
    # Disable all heat and fluid processes
    sim.enable_internal_heating = False
    sim.enable_solar_heating = False
    sim.enable_radiative_cooling = False
    sim.enable_heat_diffusion = False
    
    # Clear to space
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = sim.space_temperature
    
    # Create a simple U-shaped granite container at top
    # Container at (5, 10)
    y_base = 5
    x_center = 10
    
    # Bottom of U
    for x in range(x_center - 2, x_center + 3):
        sim.material_types[y_base + 2, x] = MaterialType.GRANITE
        sim.temperature[y_base + 2, x] = 300.0
    
    # Left wall
    for y in range(y_base, y_base + 3):
        sim.material_types[y, x_center - 2] = MaterialType.GRANITE
        sim.temperature[y, x_center - 2] = 300.0
        
    # Right wall  
    for y in range(y_base, y_base + 3):
        sim.material_types[y, x_center + 2] = MaterialType.GRANITE
        sim.temperature[y, x_center + 2] = 300.0
    
    # Put single magma cell inside
    sim.material_types[y_base + 1, x_center] = MaterialType.MAGMA
    sim.temperature[y_base + 1, x_center] = 1500.0
    
    # Floor at bottom
    sim.material_types[-1, :] = MaterialType.BASALT
    sim.temperature[-1, :] = 300.0
    
    sim._update_material_properties()
    
    # Check initial state
    magma_mask = sim.material_types == MaterialType.MAGMA
    initial_magma_temp = sim.temperature[magma_mask][0] if np.any(magma_mask) else 0
    print(f"\nInitial magma temperature: {initial_magma_temp:.1f}K")
    
    # Step once
    sim.step_forward(1.0)
    
    # Check magma temperature after movement
    magma_mask = sim.material_types == MaterialType.MAGMA
    if np.any(magma_mask):
        final_magma_temp = sim.temperature[magma_mask][0]
        print(f"Final magma temperature: {final_magma_temp:.1f}K")
        
        # Temperature should be preserved
        assert abs(final_magma_temp - initial_magma_temp) < 1.0, \
            f"Magma temperature changed: {initial_magma_temp:.1f}K -> {final_magma_temp:.1f}K"
        
        # Should NOT be space temperature
        assert final_magma_temp > 1000.0, \
            f"Magma cooled to near space temperature: {final_magma_temp:.1f}K"
            
        print("✓ Temperature preserved during movement!")
    else:
        print("✗ Magma was lost!")
        assert False, "Magma was lost during movement"


if __name__ == "__main__":
    test_simple_container_fall()