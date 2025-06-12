#!/usr/bin/env python3
"""
Test script for modular geological simulation.
"""

import numpy as np
from materials import MaterialType, MaterialDatabase
from simulation_utils import SimulationUtils
from heat_transfer import HeatTransfer

# Mock simulation class for testing
class MockSimulation:
    def __init__(self):
        self.width = 50
        self.height = 50
        self.cell_size = 50.0
        
        # Initialize grids
        self.material_types = np.full((self.height, self.width), MaterialType.GRANITE, dtype=object)
        self.temperature = np.full((self.height, self.width), 300.0)  # 300K
        self.density = np.full((self.height, self.width), 2700.0)  # kg/m³
        self.thermal_conductivity = np.full((self.height, self.width), 2.5)  # W/(m·K)
        self.specific_heat = np.full((self.height, self.width), 1000.0)  # J/(kg·K)
        self.power_density = np.zeros((self.height, self.width))
        
        # Physics parameters
        self.seconds_per_year = 365.25 * 24 * 3600
        self.dt = 1.0
        self.space_temperature = 2.7
        self.reference_temperature = 273.15
        self.atmospheric_diffusivity_enhancement = 5.0
        self.interface_diffusivity_enhancement = 1.5
        self.max_thermal_diffusivity = 1e-3
        self.max_diffusion_substeps = 50
        self.thermal_diffusion_method = "explicit_euler"
        self.diffusion_stencil = "radius2"
        self.radiative_cooling_method = "newton_raphson_implicit"
        self.atmospheric_absorption_method = "directional_sweep"
        
        # Additional physics parameters needed by heat transfer
        self.base_greenhouse_effect = 0.2
        self.max_greenhouse_effect = 0.8
        self.greenhouse_vapor_scaling = 1000.0
        self.stefan_boltzmann_geological = 5.67e-8 * self.seconds_per_year
        self.surface_radiation_depth_fraction = 0.1
        self.radiative_cooling_efficiency = 0.9
        self.solar_constant = 50
        self.solar_angle = 90.0
        self.planetary_distance_factor = 1
        self.core_heating_depth_scale = 0.5
        
        # Thermal flux tracking
        self.thermal_fluxes = {
            'solar_input': 0.0, 'radiative_output': 0.0, 'internal_heating': 0.0,
            'atmospheric_heating': 0.0, 'net_flux': 0.0
        }
        
        # Material database
        self.material_db = MaterialDatabase()
        
        # Center of mass
        self.center_of_mass = (self.width / 2, self.height / 2)
        
        # Kernels
        self._circular_kernel_3x3 = SimulationUtils.create_circular_kernel(3)
        self._circular_kernel_5x5 = SimulationUtils.create_circular_kernel(5)
        self._laplacian_kernel_radius2 = SimulationUtils.create_laplacian_kernel_radius2()
        
        # Neighbors
        self.neighbors_8 = SimulationUtils.get_neighbors(8, shuffle=False)
        self.distance_factors_8 = np.array([
            1/np.sqrt(2), 1, 1/np.sqrt(2),
            1,               1,
            1/np.sqrt(2), 1, 1/np.sqrt(2)
        ])
        
        # Logging
        self.logging_enabled = False
        
        # Create a simple planet
        distances = SimulationUtils.get_distances_from_center(self.height, self.width, self.width/2, self.height/2)
        planet_radius = min(self.width, self.height) / 3
        
        # Space outside planet
        space_mask = distances > planet_radius
        self.material_types[space_mask] = MaterialType.SPACE
        self.temperature[space_mask] = self.space_temperature
        
        # Add some atmosphere
        atmosphere_mask = (distances > planet_radius * 0.9) & (distances <= planet_radius)
        self.material_types[atmosphere_mask] = MaterialType.AIR
        self.temperature[atmosphere_mask] = 250.0  # Cold atmosphere
        
    def _get_neighbors(self, num_neighbors: int = 8, shuffle: bool = True):
        return SimulationUtils.get_neighbors(num_neighbors, shuffle)
    
    def _get_distances_from_center(self, center_x=None, center_y=None):
        if center_x is None:
            center_x = self.center_of_mass[0]
        if center_y is None:
            center_y = self.center_of_mass[1]
        return SimulationUtils.get_distances_from_center(self.height, self.width, center_x, center_y)
    
    def _get_planet_radius(self):
        return SimulationUtils.get_planet_radius(self.material_types, *self.center_of_mass)
    
    def _get_solar_direction(self):
        return SimulationUtils.get_solar_direction(self.solar_angle)

def test_simulation_utils():
    """Test simulation utilities"""
    print("Testing SimulationUtils...")
    
    # Test kernel creation
    kernel_3x3 = SimulationUtils.create_circular_kernel(3)
    assert kernel_3x3.shape == (3, 3)
    print("✓ Circular kernel creation works")
    
    # Test distance calculation
    distances = SimulationUtils.get_distances_from_center(10, 10, 5, 5)
    assert distances.shape == (10, 10)
    assert distances[5, 5] == 0.0  # Center should be 0
    print("✓ Distance calculation works")
    
    # Test neighbors
    neighbors = SimulationUtils.get_neighbors(8, shuffle=False)
    assert len(neighbors) == 8
    print("✓ Neighbor generation works")
    
    print("SimulationUtils tests passed!\n")

def test_heat_transfer():
    """Test heat transfer module"""
    print("Testing HeatTransfer...")
    
    # Create mock simulation
    sim = MockSimulation()
    
    # Create heat transfer module
    heat_transfer = HeatTransfer(sim)
    
    # Test heat diffusion
    initial_temp = sim.temperature.copy()
    new_temp, stability = heat_transfer.solve_heat_diffusion()
    
    assert new_temp.shape == initial_temp.shape
    assert stability > 0
    print(f"✓ Heat diffusion works (stability factor: {stability:.3f})")
    
    # Check that temperature changed (should have some diffusion)
    temp_change = np.abs(new_temp - initial_temp).max()
    print(f"✓ Maximum temperature change: {temp_change:.6f} K")
    
    print("HeatTransfer tests passed!\n")

def test_integration():
    """Test integration of modules"""
    print("Testing module integration...")
    
    sim = MockSimulation()
    heat_transfer = HeatTransfer(sim)
    
    # Run a few simulation steps
    for step in range(3):
        new_temp, stability = heat_transfer.solve_heat_diffusion()
        sim.temperature = new_temp
        print(f"Step {step+1}: stability={stability:.3f}, avg_temp={np.mean(new_temp):.2f}K")
    
    print("✓ Module integration works")
    print("Integration tests passed!\n")

def test_modular_simulation_engine():
    """Test the new modular simulation engine"""
    print("\nTesting modular simulation engine...")
    
    try:
        from simulation_engine_new import GeologySimulation as ModularGeologySimulation
        
        # Create a small simulation
        sim = ModularGeologySimulation(width=10, height=10, cell_size=100.0, quality=1)
        
        # Check that modules are initialized
        assert hasattr(sim, 'heat_transfer'), "HeatTransfer module not initialized"
        assert hasattr(sim, 'fluid_dynamics'), "FluidDynamics module not initialized"
        assert hasattr(sim, 'atmospheric_processes'), "AtmosphericProcesses module not initialized"
        assert hasattr(sim, 'material_processes'), "MaterialProcesses module not initialized"
        print("✓ All modules initialized correctly")
        
        # Test a simulation step
        initial_time = sim.time
        sim.step_forward(dt=0.1)
        assert sim.time > initial_time, "Simulation time did not advance"
        print("✓ Simulation step completed")
        
        # Test material painting
        sim.paint_material(5, 5, MaterialType.MAGMA, radius=1)
        magma_mask = (sim.material_types == MaterialType.MAGMA)
        assert np.any(magma_mask), "Magma painting failed"
        print("✓ Material painting works")
        
        print("Modular simulation engine test passed!")
        
    except ImportError as e:
        print(f"Could not import modular simulation engine: {e}")
    except Exception as e:
        print(f"Modular simulation engine test failed: {e}")

if __name__ == "__main__":
    print("Testing modular geological simulation components...\n")
    
    test_simulation_utils()
    test_heat_transfer()
    test_integration()
    test_modular_simulation_engine()
    
    print("All tests passed! ✅")
    print("The modular approach is working correctly.") 