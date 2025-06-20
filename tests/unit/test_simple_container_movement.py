"""Test simple rigid body container movement with fluid preservation"""

import numpy as np
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


def test_container_diagnostics():
    """Diagnostic test to understand container fluid detection"""
    sim = GeoSimulation(width=15, height=15, cell_size=100)
    sim.debug_rigid_bodies = True
    
    # Create simple box
    # Box: 5x5 with 3x3 interior
    for y in range(5, 10):
        for x in range(5, 10):
            if y in [5, 9] or x in [5, 9]:  # Walls
                sim.material_types[y, x] = MaterialType.GRANITE
            else:  # Interior
                sim.material_types[y, x] = MaterialType.WATER
    
    sim.temperature[:] = 300.0
    sim._update_material_properties()
    
    # Test the flood fill detection
    from scipy import ndimage
    
    # Get rigid body mask
    labels, num_groups = sim.fluid_dynamics.identify_rigid_groups()
    group_mask = (labels == 1)  # First group
    
    h, w = sim.material_types.shape
    exterior_mask = np.zeros((h, w), dtype=bool)
    
    # Initialize edges
    for x in range(w):
        if not group_mask[0, x]:
            exterior_mask[0, x] = True
        if not group_mask[h-1, x]:
            exterior_mask[h-1, x] = True
    for y in range(h):
        if not group_mask[y, 0]:
            exterior_mask[y, 0] = True
        if not group_mask[y, w-1]:
            exterior_mask[y, w-1] = True
    
    # Flood fill
    structure = np.ones((3, 3), dtype=bool)
    exterior_mask = ndimage.binary_dilation(exterior_mask, structure=structure, 
                                           mask=~group_mask, iterations=-1)
    
    # Contained cells
    contained_mask = ~group_mask & ~exterior_mask
    contained_count = np.sum(contained_mask)
    
    print(f"\nRigid body cells: {np.sum(group_mask)}")
    print(f"Contained fluid cells: {contained_count}")
    print(f"Expected contained: 9 (3x3 interior)")
    
    # Visual debug - show the masks
    print("\nGroup mask (G=granite, .=empty):")
    for y in range(h):
        row = ""
        for x in range(w):
            if group_mask[y, x]:
                row += "G"
            elif contained_mask[y, x]:
                row += "W"
            else:
                row += "."
        if 4 <= y <= 10:  # Only show relevant rows
            print(f"{y:2d}: {row}")
    
    assert contained_count == 9, f"Expected 9 contained cells, got {contained_count}"
    print("\n✓ Container detection working correctly!")


if __name__ == "__main__":
    test_simple_container_fall()
    print("\n" + "="*50)
    test_container_diagnostics()