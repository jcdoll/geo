"""Test surface tension force implementation"""

import numpy as np
import pytest
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


def test_surface_tension_forces_computed():
    """Test that surface tension forces are computed for water interfaces"""
    sim = GeoSimulation(width=10, height=10, cell_size=100, setup_planet=False)
    sim.enable_surface_tension = True
    
    # Create a line of water in space
    for x in range(3, 7):
        sim.material_types[5, x] = MaterialType.WATER
    
    # Update properties to calculate density
    sim._update_material_properties()
    
    # Compute forces
    fx, fy = sim.fluid_dynamics.compute_force_field()
    
    # Check that forces exist at water cells
    water_mask = sim.material_types == MaterialType.WATER
    
    # Print debug info
    print("\nWater locations:")
    print(np.where(water_mask))
    print("\nForce X at water cells:")
    print(fx[water_mask])
    print("\nForce Y at water cells:")
    print(fy[water_mask])
    
    # Surface tension should create non-zero forces at interface cells
    assert np.any(np.abs(fx[water_mask]) > 0) or np.any(np.abs(fy[water_mask]) > 0), \
        "No surface tension forces detected at water interfaces"
    
    # Check that interface cells have forces
    # Water at edges should have inward forces
    assert fx[5, 3] > 0, "Left edge water should have rightward force"
    assert fx[5, 6] < 0, "Right edge water should have leftward force"


def test_surface_tension_creates_cohesion():
    """Test that surface tension creates cohesive behavior in water droplets"""
    sim = GeoSimulation(width=20, height=20, cell_size=100, setup_planet=False)
    sim.enable_surface_tension = True
    sim.enable_material_processes = False  # Prevent water from evaporating
    
    # Create a small water blob
    sim.material_types[10:13, 9:12] = MaterialType.WATER
    
    # Store initial positions
    initial_water = np.copy(sim.material_types == MaterialType.WATER)
    
    # Run a few macro steps
    for _ in range(10):
        sim.step_forward(1.0)
    
    # Check that water stayed relatively cohesive
    final_water = sim.material_types == MaterialType.WATER
    
    # Count water cells that moved
    initial_count = np.sum(initial_water)
    final_count = np.sum(final_water)
    
    # Water should not disperse much
    assert final_count == initial_count, "Water cells were lost"
    
    # Check connectivity - water should stay mostly connected
    from scipy import ndimage
    labeled, num_features = ndimage.label(final_water)
    assert num_features <= 2, f"Water split into {num_features} disconnected parts"


def test_surface_tension_magnitude():
    """Test that surface tension forces have appropriate magnitude"""
    sim = GeoSimulation(width=10, height=10, cell_size=100, setup_planet=False)
    sim.enable_surface_tension = True
    
    # Create a single water cell surrounded by space
    sim.material_types[5, 5] = MaterialType.WATER
    sim._update_material_properties()
    
    # Get surface tension forces specifically
    fx_st, fy_st = sim.fluid_dynamics._compute_surface_tension_forces()
    
    # A single water cell surrounded by space should have zero net force
    # (it's symmetric in all directions)
    assert abs(fx_st[5, 5]) < 1e-10, f"Single water cell has non-zero X force: {fx_st[5, 5]}"
    assert abs(fy_st[5, 5]) < 1e-10, f"Single water cell has non-zero Y force: {fy_st[5, 5]}"
    
    # Create asymmetric case - water line
    sim.material_types[:, :] = MaterialType.SPACE
    sim.material_types[5, 3:7] = MaterialType.WATER
    sim._update_material_properties()
    
    fx_st, fy_st = sim.fluid_dynamics._compute_surface_tension_forces()
    
    # Edge cells should have outward forces (tension pulls outward)
    print(f"\nEdge forces: left={fx_st[5,3]:.2e}, right={fx_st[5,6]:.2e}")
    assert fx_st[5, 3] < 0, "Left edge should have leftward surface tension"
    assert fx_st[5, 6] > 0, "Right edge should have rightward surface tension"
    
    # Check magnitude is reasonable (should be significant but not huge)
    force_magnitude = max(abs(fx_st[5, 3]), abs(fx_st[5, 6]))
    assert 1e-10 < force_magnitude < 1e3, f"Surface tension magnitude {force_magnitude} out of expected range"


def test_surface_tension_vs_gravity():
    """Test that surface tension can compete with gravity for small droplets"""
    sim = GeoSimulation(width=20, height=20, cell_size=100, setup_planet=False)
    sim.enable_surface_tension = True
    sim.external_gravity = (0, 10)  # Downward gravity
    
    # Create a small water droplet high up
    sim.material_types[5:7, 9:11] = MaterialType.WATER
    
    # Run simulation
    for _ in range(5):
        sim.step_forward(1.0)
    
    # Check if water maintained some cohesion despite gravity
    water_mask = sim.material_types == MaterialType.WATER
    water_positions = np.where(water_mask)
    
    if len(water_positions[0]) > 0:
        # Calculate spread of water
        y_min, y_max = water_positions[0].min(), water_positions[0].max()
        x_min, x_max = water_positions[1].min(), water_positions[1].max()
        
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        
        # Water should not be too stretched vertically
        assert height <= 4, f"Water stretched too much vertically: {height} cells"
        assert width <= 4, f"Water spread too much horizontally: {width} cells"


if __name__ == "__main__":
    # Run with verbose output
    test_surface_tension_forces_computed()
    print("\n" + "="*50 + "\n")
    test_surface_tension_magnitude()
    print("\n" + "="*50 + "\n")
    test_surface_tension_creates_cohesion()
    print("\n" + "="*50 + "\n")
    test_surface_tension_vs_gravity()
    print("\nAll tests passed!") 