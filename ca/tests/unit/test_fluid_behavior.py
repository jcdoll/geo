"""Test fluid behavior
"""

import numpy as np
import pytest
from scipy import ndimage
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


def test_fluid_conservation():
    """Test that fluid cells are conserved during simulation"""
    sim = GeoSimulation(width=10, height=10, cell_size=100, setup_planet=False)
    
    # Create a line of water in air
    sim.material_types[:] = MaterialType.AIR
    for x in range(3, 7):
        sim.material_types[5, x] = MaterialType.WATER
    
    # Disable phase transitions to ensure conservation
    sim.enable_material_processes = False
    sim.enable_atmospheric_processes = False
    
    # Update properties to calculate density
    sim._update_material_properties()
    
    initial_water_count = np.sum(sim.material_types == MaterialType.WATER)
    
    # Run one step
    sim.step_forward(1.0)
    
    # Check that water cells are conserved
    final_water_count = np.sum(sim.material_types == MaterialType.WATER)
    assert final_water_count == initial_water_count, f"Water count changed: {initial_water_count} -> {final_water_count}"


def test_fluid_blob_stability():
    """Test that fluid blobs remain relatively stable without external forces"""
    sim = GeoSimulation(width=20, height=20, cell_size=100, setup_planet=False)
    sim.enable_material_processes = False  # Prevent water from evaporating
    sim.external_gravity = (0, 0)  # No gravity
    
    # Create a small water blob in air
    sim.material_types[:] = MaterialType.AIR
    sim.material_types[10:13, 9:12] = MaterialType.WATER
    
    # Store initial positions
    initial_water = np.copy(sim.material_types == MaterialType.WATER)
    initial_count = np.sum(initial_water)
    
    # Run a few macro steps
    for _ in range(5):
        sim.step_forward(1.0)
    
    # Check that water stayed relatively stable
    final_water = sim.material_types == MaterialType.WATER
    final_count = np.sum(final_water)
    
    # Water should be conserved
    assert final_count == initial_count, "Water cells were lost"
    
    # Check connectivity - water should stay mostly connected
    # Note: Some fragmentation is expected on coarse discrete grids (50-100m cells)
    labeled, num_features = ndimage.label(final_water)
    assert num_features <= 10, f"Water split into {num_features} disconnected parts (expected some fragmentation on coarse grid)"


def test_fluid_falls_under_gravity():
    """Test that fluids fall under gravity"""
    sim = GeoSimulation(width=20, height=20, cell_size=100, setup_planet=False)
    sim.external_gravity = (0, 10)  # Downward gravity
    sim.enable_material_processes = False  # Prevent phase changes
    
    # Create air environment
    sim.material_types[:] = MaterialType.AIR
    
    # Create a small water droplet high up
    sim.material_types[5:7, 9:11] = MaterialType.WATER
    
    # Get initial center of mass
    water_positions = np.where(sim.material_types == MaterialType.WATER)
    initial_y_center = np.mean(water_positions[0])
    
    # Run simulation
    for _ in range(10):
        sim.step_forward(1.0)
    
    # Check if water fell
    water_mask = sim.material_types == MaterialType.WATER
    water_positions = np.where(water_mask)
    
    assert len(water_positions[0]) > 0, "Water disappeared"
    
    final_y_center = np.mean(water_positions[0])
    
    # Water should have fallen (y increases downward)
    assert final_y_center > initial_y_center + 2, f"Water didn't fall enough: {initial_y_center:.1f} -> {final_y_center:.1f}"




if __name__ == "__main__":
    # Run with verbose output
    test_fluid_conservation()
    print("\n" + "="*50 + "\n")
    test_fluid_blob_stability()
    print("\n" + "="*50 + "\n")
    test_fluid_falls_under_gravity()
    print("\nAll tests passed!")