"""
Simple test to verify the modular geological simulation engine works correctly.
"""

import numpy as np
from simulation_engine_new import GeologySimulation
from materials import MaterialType

def test_modular_simulation():
    """Test that the modular simulation engine works correctly"""
    print("Testing modular geological simulation engine...")
    
    # Create a small simulation
    sim = GeologySimulation(width=20, height=20, cell_size=100.0, quality=1)
    
    # Check initial state
    print(f"Initial simulation time: {sim.time}")
    print(f"Grid size: {sim.width}x{sim.height}")
    print(f"Cell size: {sim.cell_size}m")
    
    # Check that modules are initialized
    assert hasattr(sim, 'heat_transfer'), "HeatTransfer module not initialized"
    assert hasattr(sim, 'fluid_dynamics'), "FluidDynamics module not initialized"
    assert hasattr(sim, 'atmospheric_processes'), "AtmosphericProcesses module not initialized"
    assert hasattr(sim, 'material_processes'), "MaterialProcesses module not initialized"
    print("✓ All modules initialized correctly")
    
    # Check material properties are set
    non_space_mask = (sim.material_types != MaterialType.SPACE)
    assert np.any(non_space_mask), "No non-space materials found"
    
    # Check that densities are reasonable
    densities = sim.density[non_space_mask]
    assert np.all(densities > 0), "Some materials have zero density"
    assert np.all(densities < 10000), "Some materials have unreasonably high density"
    print(f"✓ Material densities range from {np.min(densities):.1f} to {np.max(densities):.1f} kg/m³")
    
    # Check temperatures are reasonable
    temperatures = sim.temperature[non_space_mask]
    assert np.all(temperatures > 0), "Some materials have zero temperature"
    assert np.all(temperatures < 2000), "Some materials have unreasonably high temperature"
    print(f"✓ Temperatures range from {np.min(temperatures):.1f} to {np.max(temperatures):.1f} K")
    
    # Test a simulation step
    initial_time = sim.time
    sim.step_forward(dt=0.1)
    assert sim.time > initial_time, "Simulation time did not advance"
    print(f"✓ Simulation step completed, time advanced to {sim.time}")
    
    # Test material painting
    sim.paint_material(10, 10, MaterialType.MAGMA, radius=2)
    magma_mask = (sim.material_types == MaterialType.MAGMA)
    assert np.any(magma_mask), "Magma painting failed"
    print("✓ Material painting works")
    
    # Test heat source addition
    sim.add_heat_source(5, 5, 1000.0, radius=1)
    assert sim.power_density[5, 5] > 0, "Heat source addition failed"
    print("✓ Heat source addition works")
    
    # Test statistics
    stats = sim.get_stats()
    required_keys = ['time', 'avg_temperature', 'max_temperature', 'min_temperature', 'thermal_fluxes']
    for key in required_keys:
        assert key in stats, f"Missing key in stats: {key}"
    print("✓ Statistics generation works")
    
    # Test visualization data
    materials, temps, pressures, power = sim.get_visualization_data()
    assert materials.shape == (20, 20), "Material grid shape incorrect"
    assert temps.shape == (20, 20), "Temperature grid shape incorrect"
    assert pressures.shape == (20, 20), "Pressure grid shape incorrect"
    assert power.shape == (20, 20), "Power density grid shape incorrect"
    print("✓ Visualization data generation works")
    
    print("\nModular simulation engine test completed successfully!")
    print("All core functionality is working correctly.")

if __name__ == "__main__":
    test_modular_simulation() 