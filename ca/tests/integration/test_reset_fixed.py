"""Test reset functionality properly"""
import numpy as np
from geo_game import GeoGame
from materials import MaterialType


def test_reset_restores_initial_state():
    """Test that reset restores the initial state with planet"""
    # Create simulation with planet
    sim = GeoGame(width=20, height=20, cell_size=1.0, setup_planet=True)
    
    # Store initial state
    initial_materials = sim.material_types.copy()
    initial_temperature = sim.temperature.copy()
    initial_time = sim.time
    
    # Verify we have a planet (not all space)
    assert not np.all(sim.material_types == MaterialType.SPACE), "Should have a planet initially"
    
    # Make some changes
    # Add heat
    sim.add_heat_source(10, 10, 500.0, 2)
    # Run a few steps
    for _ in range(5):
        sim.step_forward(1.0)
    
    # Verify state has changed
    assert not np.array_equal(sim.material_types, initial_materials), "Materials should have changed"
    assert not np.array_equal(sim.temperature, initial_temperature), "Temperature should have changed"
    assert sim.time > initial_time, "Time should have advanced"
    
    # Reset
    sim.reset()
    
    # Verify state is restored
    assert np.array_equal(sim.material_types, initial_materials), "Materials should be restored"
    assert np.array_equal(sim.temperature, initial_temperature), "Temperature should be restored"
    assert sim.time == initial_time, "Time should be reset"
    
    # Verify grid properties are preserved
    assert sim.width == 20
    assert sim.height == 20
    assert sim.cell_size == 50.0
    
    print("Reset restore test passed")


def test_clear_to_space():
    """Test clearing simulation to empty space"""
    # Create simulation with planet
    sim = GeoGame(width=15, height=15, cell_size=1.0, setup_planet=True)
    
    # Verify we have a planet
    assert not np.all(sim.material_types == MaterialType.SPACE), "Should have a planet initially"
    
    # Clear to space by reinitializing without planet
    sim.__init__(sim.width, sim.height, cell_size=sim.cell_size, 
                 cell_depth=sim.cell_depth, setup_planet=False)
    
    # Verify everything is space
    assert np.all(sim.material_types == MaterialType.SPACE), "Should be all space after clear"
    assert np.all(sim.temperature == 0), "Temperature should be zero in space"
    assert sim.time == 0, "Time should be reset"
    
    print("Clear to space test passed")


def test_reset_without_planet():
    """Test reset functionality when starting without planet"""
    # Create empty simulation
    sim = GeoGame(width=10, height=10, cell_size=1.0, setup_planet=False)
    
    # Verify it's empty
    assert np.all(sim.material_types == MaterialType.SPACE), "Should be empty initially"
    
    # Add some materials manually
    sim.material_types[5:7, 5:7] = MaterialType.WATER
    sim.temperature[5:7, 5:7] = 300.0
    
    # Step forward
    sim.step_forward(1.0)
    
    # Reset
    sim.reset()
    
    # Should be empty again (setup_planet=False is preserved)
    assert np.all(sim.material_types == MaterialType.SPACE), "Should be empty after reset"
    assert np.all(sim.temperature == 0), "Temperature should be zero"
    
    print("Reset without planet test passed")


if __name__ == "__main__":
    test_reset_restores_initial_state()
    test_clear_to_space()
    test_reset_without_planet()
    print("All reset tests passed!")