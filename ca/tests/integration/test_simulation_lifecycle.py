"""Integration tests for simulation lifecycle and overall behavior"""

import numpy as np
import pytest
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


def test_simulation_initialization():
    """Test that simulation initializes correctly"""
    sim = GeoSimulation(width=20, height=30, cell_size=1.0, setup_planet=False)
    
    # Check basic properties
    assert sim.width == 20
    assert sim.height == 30
    assert sim.cell_size == 50.0
    
    # Check arrays are initialized
    assert sim.material_types.shape == (30, 20)
    assert sim.temperature.shape == (30, 20)
    # CA doesn't have pressure
    assert sim.density.shape == (30, 20)
    
    # Check default values (without planet)
    assert np.all(sim.material_types == MaterialType.SPACE)
    assert np.all(sim.temperature >= 0)  # Should have non-negative temperatures
    # CA doesn't calculate pressure
    assert np.all(sim.density >= 0)    # Density should be non-negative
    
    print(f"Simulation initialization test passed")
    print(f"  Grid size: {sim.width}x{sim.height}")
    print(f"  Cell size: {sim.cell_size} m")


def test_simulation_step_forward():
    """Test that simulation can step forward without errors"""
    sim = GeoSimulation(width=10, height=15, cell_size=100.0, setup_planet=False)
    
    # Add some materials for interesting dynamics
    sim.material_types[5:10, :] = MaterialType.WATER
    sim.material_types[10:, :] = MaterialType.GRANITE
    sim.temperature[5:10, :] = 275.0
    sim.temperature[10:, :] = 300.0
    
    # Update material properties
    sim._update_material_properties()
    
    # Record initial state
    initial_temp = sim.temperature.copy()
    # CA doesn't have pressure
    
    # Step forward multiple times
    dt = 1.0
    for step in range(5):
        try:
            sim.step_forward(dt)
        except Exception as e:
            pytest.fail(f"Simulation step {step} failed: {e}")
    
    # Check that something changed (simulation is active)
    temp_changed = not np.allclose(sim.temperature, initial_temp)
    # CA doesn't calculate pressure
    
    # At least one field should change
    assert temp_changed, "Temperature should evolve over time"
    
    print(f"Simulation step forward test passed")
    print(f"  Temperature changed: {temp_changed}")


def test_simulation_reset():
    """Test simulation reset functionality"""
    # Test with planet setup
    sim = GeoSimulation(width=8, height=12, cell_size=75.0, setup_planet=True)
    
    # Store initial state
    initial_materials = sim.material_types.copy()
    initial_temp = sim.temperature.copy()
    initial_time = sim.time
    
    # Verify we have a planet
    assert not np.all(sim.material_types == MaterialType.SPACE), "Should have a planet initially"
    
    # Modify simulation state
    sim.material_types[3:6, 2:5] = MaterialType.MAGMA
    sim.temperature[3:6, 2:5] = 1500.0
    sim.external_gravity = (1, 5)
    
    # Step forward to change state further
    for _ in range(3):
        sim.step_forward(1.0)
    
    # Verify state changed
    assert not np.array_equal(sim.material_types, initial_materials), "Materials should have changed"
    assert sim.time > initial_time, "Time should have advanced"
    
    # Reset simulation
    sim.reset()
    
    # Check that state is restored to initial planet configuration
    assert np.array_equal(sim.material_types, initial_materials), "Materials should restore to initial planet"
    assert np.array_equal(sim.temperature, initial_temp), "Temperature should restore to initial"
    assert sim.time == initial_time, "Time should reset to zero"
    
    # Grid properties should remain unchanged
    assert sim.width == 8
    assert sim.height == 12
    assert sim.cell_size == 75.0
    
    print(f"Simulation reset test passed")


def test_physics_integration():
    """Test that different physics modules work together"""
    sim = GeoSimulation(width=15, height=20, cell_size=100.0, setup_planet=False)
    
    # Enable multiple physics systems
    sim.external_gravity = (0, 10)
    sim.enable_self_gravity = False
    sim.enable_heat_diffusion = True
    sim.enable_material_processes = True
    sim.enable_atmospheric_processes = True
    
    # Create a complex scenario
    # Hot magma chamber in granite host
    sim.material_types[:] = MaterialType.GRANITE
    sim.temperature[:] = 300.0
    
    # Magma chamber
    chamber_y, chamber_x = 10, 7
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y, x = chamber_y + dy, chamber_x + dx
            if 0 <= y < sim.height and 0 <= x < sim.width:
                sim.material_types[y, x] = MaterialType.MAGMA
                sim.temperature[y, x] = 1600.0
    
    # Water on surface
    sim.material_types[:3, :] = MaterialType.WATER
    sim.temperature[:3, :] = 275.0
    
    # Air above water
    sim.material_types[0, :] = MaterialType.AIR
    sim.temperature[0, :] = 280.0
    
    sim._update_material_properties()
    
    # Record initial state
    initial_materials = sim.material_types.copy()
    initial_temp = sim.temperature.copy()
    
    # Run simulation for several steps
    dt = 1.0
    for step in range(10):
        sim.step_forward(dt)
    
    # Check that physics systems are interacting
    # Heat should diffuse (even small changes count)
    temp_change = np.max(np.abs(sim.temperature - initial_temp))
    heat_diffused = temp_change > 0.001  # Very small change is OK
    
    # Materials may transform
    materials_changed = not np.array_equal(sim.material_types, initial_materials)
    
    # CA doesn't calculate pressure
    pressure_developed = False
    
    # At least heat diffusion should occur
    assert heat_diffused, f"Heat diffusion should occur with temperature gradients. Max temp change: {temp_change}"
    
    print(f"Physics integration test passed")
    print(f"  Heat diffused: {heat_diffused}")
    print(f"  Materials changed: {materials_changed}")
    print(f"  Pressure developed: {pressure_developed}")
    print(f"  Final max temperature: {np.max(sim.temperature):.1f}K")


def test_conservation_laws():
    """Test that conservation laws are approximately satisfied"""
    sim = GeoSimulation(width=12, height=12, cell_size=100.0, setup_planet=False)
    
    # Disable processes that might create/destroy materials
    sim.enable_material_processes = False
    sim.enable_atmospheric_processes = False
    
    # Create closed system with different materials
    sim.material_types[:] = MaterialType.GRANITE  # Background
    sim.material_types[3:6, 3:6] = MaterialType.WATER  # Water blob
    sim.material_types[7:9, 7:9] = MaterialType.MAGMA  # Hot spot
    
    # Set temperatures
    sim.temperature[:] = 300.0
    sim.temperature[3:6, 3:6] = 275.0
    sim.temperature[7:9, 7:9] = 1400.0
    
    sim._update_material_properties()
    
    # Calculate initial totals
    initial_water_count = np.sum(sim.material_types == MaterialType.WATER)
    initial_magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
    initial_granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
    
    # Calculate initial total thermal energy (approximate)
    initial_thermal_energy = np.sum(sim.temperature * sim.density)
    
    # Run simulation
    dt = 1.0
    for step in range(5):
        sim.step_forward(dt)
    
    # Check conservation
    final_water_count = np.sum(sim.material_types == MaterialType.WATER)
    final_magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
    final_granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
    
    # Material counts should be conserved (no phase transitions enabled)
    assert final_water_count == initial_water_count, "Water count should be conserved"
    assert final_magma_count == initial_magma_count, "Magma count should be conserved"
    assert final_granite_count == initial_granite_count, "Granite count should be conserved"
    
    # Total cell count should be conserved
    total_cells = sim.width * sim.height
    counted_cells = final_water_count + final_magma_count + final_granite_count
    assert counted_cells == total_cells, "Total cell count should be conserved"
    
    print(f"Conservation laws test passed")
    print(f"  Water cells: {initial_water_count} -> {final_water_count}")
    print(f"  Magma cells: {initial_magma_count} -> {final_magma_count}")
    print(f"  Granite cells: {initial_granite_count} -> {final_granite_count}")


if __name__ == "__main__":
    test_simulation_initialization()
    test_simulation_step_forward()
    test_simulation_reset()
    test_physics_integration()
    test_conservation_laws()
    print("\nAll simulation lifecycle integration tests passed!")