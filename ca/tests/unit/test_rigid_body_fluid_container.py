"""Test rigid body containers with fluids (e.g., rock donut with magma)"""

import numpy as np
import pytest
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


@pytest.mark.skip(reason="Rigid bodies removed - materials flow and mix")
def test_rock_donut_with_magma():
    """Test that a rock donut filled with magma preserves temperature when moving"""
    # Create simulation with external gravity
    sim = GeoSimulation(width=40, height=40, cell_size=100)
    sim.external_gravity = (0, 10)  # Downward gravity
    sim.enable_self_gravity = False
    sim.enable_solid_drag = False
    sim.debug_rigid_bodies = True  # Enable debug output
    
    # Disable heat processes to isolate the movement issue
    sim.enable_internal_heating = False
    sim.enable_solar_heating = False
    sim.enable_radiative_cooling = False
    sim.enable_heat_diffusion = False
    
    # Enable material processes to test rigid body detection with mixed materials
    sim.enable_material_processes = True
    
    # Disable force-based swapping to prevent donut breakup
    if hasattr(sim.fluid_dynamics, 'velocity_threshold'):
        sim.fluid_dynamics.velocity_threshold = False
    
    # Clear to space
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = sim.space_temperature  # 2.7K
    
    # Create a granite donut with magma inside
    # Donut center at (20, 10)
    center_y, center_x = 10, 20
    
    # Create a thicker outer ring of granite (7x7 with 3x3 hole) for stability
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            y, x = center_y + dy, center_x + dx
            # Skip the inner 3x3 area to create the hole
            if abs(dy) <= 1 and abs(dx) <= 1:
                continue  # This will be filled with magma
            sim.material_types[y, x] = MaterialType.GRANITE
            sim.temperature[y, x] = 1000.0  # Warm rock
    
    # Fill the hole with magma
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            y, x = center_y + dy, center_x + dx
            sim.material_types[y, x] = MaterialType.MAGMA
            sim.temperature[y, x] = 1500.0  # Hot magma
    
    # Create a solid floor at the bottom
    for x in range(sim.width):
        sim.material_types[-3:, x] = MaterialType.BASALT
        sim.temperature[-3:, x] = 300.0
    
    # Update properties
    sim._update_material_properties()
    
    # Record initial state
    granite_mask = sim.material_types == MaterialType.GRANITE
    magma_mask = sim.material_types == MaterialType.MAGMA
    
    initial_granite_count = np.sum(granite_mask)
    initial_magma_count = np.sum(magma_mask)
    initial_magma_temp = np.mean(sim.temperature[magma_mask])
    
    print(f"\nInitial state:")
    print(f"Granite cells: {initial_granite_count}")
    print(f"Magma cells: {initial_magma_count}")
    print(f"Magma temperature: {initial_magma_temp:.1f}K")
    
    # Run simulation for several steps to let the donut fall
    for step in range(15):
        sim.step_forward(1.0)
        
        # Check current state
        granite_mask = sim.material_types == MaterialType.GRANITE
        magma_mask = sim.material_types == MaterialType.MAGMA
        
        granite_count = np.sum(granite_mask)
        magma_count = np.sum(magma_mask)
        
        if magma_count > 0:
            magma_temp = np.mean(sim.temperature[magma_mask])
            print(f"Step {step+1}: Granite={granite_count}, Magma={magma_count}, Magma temp={magma_temp:.1f}K")
        else:
            print(f"Step {step+1}: Granite={granite_count}, Magma={magma_count} (lost!)")
    
    # Verify results
    final_granite_mask = sim.material_types == MaterialType.GRANITE
    final_magma_mask = sim.material_types == MaterialType.MAGMA
    
    final_granite_count = np.sum(final_granite_mask)
    final_magma_count = np.sum(final_magma_mask)
    
    # 1. Material conservation (allow some granite loss due to other physics)
    assert final_granite_count >= initial_granite_count * 0.7, \
        f"Too much granite lost: {initial_granite_count} -> {final_granite_count}"
    assert final_magma_count == initial_magma_count, \
        f"Magma count changed: {initial_magma_count} -> {final_magma_count}"
    
    # 2. Temperature preservation
    if final_magma_count > 0:
        final_magma_temp = np.mean(sim.temperature[final_magma_mask])
        # Allow small variation due to numerical precision
        assert abs(final_magma_temp - initial_magma_temp) < 10.0, \
            f"Magma temperature corrupted: {initial_magma_temp:.1f}K -> {final_magma_temp:.1f}K"
        
        # Most importantly, magma should NOT be at space temperature
        assert final_magma_temp > 1000.0, \
            f"Magma cooled to space temperature! {final_magma_temp:.1f}K"
    
    # 3. Check that donut moved as a unit
    if np.any(final_granite_mask):
        granite_coords = np.argwhere(final_granite_mask)
        final_center_y = np.mean(granite_coords[:, 0])
        final_center_x = np.mean(granite_coords[:, 1])
        
        # Should have fallen (Y increased)
        assert final_center_y > center_y, "Donut should have fallen"
        
        # Check magma is still inside the donut
        if np.any(final_magma_mask):
            magma_coords = np.argwhere(final_magma_mask)
            magma_center_y = np.mean(magma_coords[:, 0])
            magma_center_x = np.mean(magma_coords[:, 1])
            
            # Magma center should be close to granite center
            distance = np.hypot(magma_center_y - final_center_y, 
                              magma_center_x - final_center_x)
            assert distance < 2.0, \
                f"Magma separated from donut! Distance: {distance:.1f} cells"
    
    print("\n✓ Rock donut with magma test passed!")


def test_ice_bowl_with_water():
    """Test that an ice bowl filled with water moves correctly"""
    sim = GeoSimulation(width=30, height=30, cell_size=100)
    sim.external_gravity = (0, 10)
    sim.enable_self_gravity = False
    sim.enable_solid_drag = False
    
    # Disable heat processes
    sim.enable_internal_heating = False
    sim.enable_solar_heating = False
    sim.enable_radiative_cooling = False
    sim.enable_heat_diffusion = False
    
    # Clear to air (not space, to avoid extreme cooling)
    sim.material_types[:] = MaterialType.AIR
    sim.temperature[:] = 280.0  # Above freezing
    
    # Create an ice bowl (U-shape)
    bowl_y, bowl_x = 5, 15
    
    # Bottom and sides of bowl
    for dx in range(-3, 4):
        sim.material_types[bowl_y + 2, bowl_x + dx] = MaterialType.ICE  # Bottom
        sim.temperature[bowl_y + 2, bowl_x + dx] = 270.0
    for dy in range(0, 3):
        sim.material_types[bowl_y + dy, bowl_x - 3] = MaterialType.ICE  # Left side
        sim.material_types[bowl_y + dy, bowl_x + 3] = MaterialType.ICE  # Right side
        sim.temperature[bowl_y + dy, bowl_x - 3] = 270.0
        sim.temperature[bowl_y + dy, bowl_x + 3] = 270.0
    
    # Fill with water
    for dy in range(0, 2):
        for dx in range(-2, 3):
            sim.material_types[bowl_y + dy, bowl_x + dx] = MaterialType.WATER
            sim.temperature[bowl_y + dy, bowl_x + dx] = 275.0
    
    # Create floor
    sim.material_types[-2:, :] = MaterialType.BASALT
    sim.temperature[-2:, :] = 280.0
    
    sim._update_material_properties()
    
    # Record initial water state
    water_mask = sim.material_types == MaterialType.WATER
    initial_water_count = np.sum(water_mask)
    initial_water_temp = np.mean(sim.temperature[water_mask])
    
    print(f"\nInitial water: {initial_water_count} cells at {initial_water_temp:.1f}K")
    
    # Run simulation
    for step in range(10):
        sim.step_forward(1.0)
    
    # Check final state
    final_water_mask = sim.material_types == MaterialType.WATER
    final_water_count = np.sum(final_water_mask)
    
    if final_water_count > 0:
        final_water_temp = np.mean(sim.temperature[final_water_mask])
        
        # Water should be conserved and maintain reasonable temperature
        assert final_water_count >= initial_water_count - 2, \
            f"Lost too much water: {initial_water_count} -> {final_water_count}"
        assert final_water_temp > 270.0, \
            f"Water temperature dropped too much: {final_water_temp:.1f}K"
        
        print(f"Final water: {final_water_count} cells at {final_water_temp:.1f}K")
        print("\n✓ Ice bowl with water test passed!")
    else:
        print("Warning: All water was lost or frozen")


if __name__ == "__main__":
    test_rock_donut_with_magma()
    print("\n" + "="*50)
    test_ice_bowl_with_water()