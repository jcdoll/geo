"""
Test to debug temperature stability issues in the planet scenario.
"""

import numpy as np
import pytest
from state import FluxState
from simulation import FluxSimulation
from scenarios import setup_scenario
from materials import MaterialDatabase, MaterialType


class TestTemperatureStability:
    """Test suite for temperature stability."""
    
    def test_planet_temperature_stability(self):
        """Test that planet scenario doesn't have temperature explosion."""
        # Create small planet for faster testing
        sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario="planet")
        
        # Track temperature statistics
        max_temps = []
        min_temps = []
        mean_temps = []
        
        # Run for 50 steps to catch any delayed explosion
        for step in range(50):
            sim.step_forward()
            
            # Get temperature stats
            temp = sim.state.temperature
            max_temp = np.max(temp)
            min_temp = np.min(temp)
            mean_temp = np.mean(temp)
            
            max_temps.append(max_temp)
            min_temps.append(min_temp)
            mean_temps.append(mean_temp)
            
            print(f"Step {step}: T_max={max_temp:.1f}K, T_min={min_temp:.1f}K, T_mean={mean_temp:.1f}K")
            
            # Check for temperature explosion
            assert max_temp < 1e6, f"Temperature exploded to {max_temp:.2e}K at step {step}"
            assert min_temp >= 0, f"Temperature went below absolute zero: {min_temp}K at step {step}"
            
        # Check that temperatures are reasonable after 20 steps
        # Planet scenario has uranium core at ~2288K initially
        final_max = max_temps[-1]
        initial_max = max_temps[0]
        
        # Temperature shouldn't increase dramatically from initial
        assert final_max < initial_max * 1.5, f"Temperature increased too much: {initial_max}K -> {final_max}K"
        
        # Absolute limit to catch runaway
        assert final_max < 5000, f"Temperature unreasonably high: {final_max}K"
        
    def test_planet_no_transport(self):
        """Test planet scenario with transport disabled."""
        sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario="planet")
        
        # Disable transport
        sim.enable_advection = False
        
        max_temps = []
        for step in range(20):
            sim.step_forward()
            max_temp = np.max(sim.state.temperature)
            max_temps.append(max_temp)
            print(f"Step {step} (no transport): T_max={max_temp:.1f}K")
            assert max_temp < 1e6, f"Temperature exploded without transport at step {step}"
            
    def test_planet_no_gravity(self):
        """Test planet scenario with gravity disabled."""
        sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario="planet")
        
        # Disable gravity
        sim.enable_gravity = False
        
        max_temps = []
        for step in range(20):
            sim.step_forward()
            max_temp = np.max(sim.state.temperature)
            max_temps.append(max_temp)
            print(f"Step {step} (no gravity): T_max={max_temp:.1f}K")
            assert max_temp < 1e6, f"Temperature exploded without gravity at step {step}"
            
    def test_planet_no_pressure(self):
        """Test planet scenario with pressure projection disabled."""
        sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario="planet")
        
        # Disable pressure projection
        sim.enable_momentum = False
        
        max_temps = []
        for step in range(20):
            sim.step_forward()
            max_temp = np.max(sim.state.temperature)
            max_temps.append(max_temp)
            print(f"Step {step} (no pressure): T_max={max_temp:.1f}K")
            assert max_temp < 1e6, f"Temperature exploded without pressure at step {step}"
            
    def test_planet_no_heat_transfer(self):
        """Test planet scenario with heat transfer disabled."""
        sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario="planet")
        
        # Disable heat transfer
        sim.enable_heat_transfer = False
        
        max_temps = []
        for step in range(20):
            sim.step_forward()
            max_temp = np.max(sim.state.temperature)
            max_temps.append(max_temp)
            print(f"Step {step} (no heat): T_max={max_temp:.1f}K")
            assert max_temp < 1e6, f"Temperature exploded without heat transfer at step {step}"
            
    def test_planet_no_solar(self):
        """Test planet scenario with solar heating disabled."""
        sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario="planet")
        
        # Disable solar heating
        sim.enable_solar_heating = False
        
        max_temps = []
        for step in range(20):
            sim.step_forward()
            max_temp = np.max(sim.state.temperature)
            max_temps.append(max_temp)
            print(f"Step {step} (no solar): T_max={max_temp:.1f}K")
            assert max_temp < 1e6, f"Temperature exploded without solar at step {step}"
            
    def test_simple_air_space_interface(self):
        """Test a simple air-space interface for stability."""
        # Create a simple test case
        nx, ny = 10, 10
        state = FluxState(nx, ny, dx=50.0)
        
        # Left half is air, right half is space
        state.vol_frac[MaterialType.AIR.value, :, :5] = 1.0
        state.vol_frac[MaterialType.SPACE.value, :, :5] = 0.0
        
        state.vol_frac[MaterialType.AIR.value, :, 5:] = 0.0
        state.vol_frac[MaterialType.SPACE.value, :, 5:] = 1.0
        
        # Set initial temperature
        state.temperature[:, :5] = 300.0  # Air at 300K
        state.temperature[:, 5:] = 2.7    # Space at cosmic background
        
        # Update material properties
        material_db = MaterialDatabase()
        state.update_mixture_properties(material_db)
        
        # Create heat transfer solver
        from heat_transfer import HeatTransfer
        heat_solver = HeatTransfer(state)
        
        # Run a few steps
        dt = 0.1
        for step in range(10):
            heat_solver.solve_heat_equation(dt)
            max_temp = np.max(state.temperature)
            min_temp = np.min(state.temperature)
            air_temp = np.max(state.temperature[:, :5])
            print(f"Step {step}: T_max={max_temp:.1f}K, T_min={min_temp:.1f}K, T_air_max={air_temp:.1f}K")
            
            # Temperature should not explode
            assert max_temp < 1000, f"Temperature exploded in simple test: {max_temp}K"
            assert min_temp >= 0, f"Temperature below zero: {min_temp}K"


if __name__ == "__main__":
    # Run the tests
    test = TestTemperatureStability()
    
    print("=== Testing basic planet scenario ===")
    test.test_planet_temperature_stability()
    
    print("\n=== Testing with transport disabled ===")
    test.test_planet_no_transport()
    
    print("\n=== Testing with gravity disabled ===")
    test.test_planet_no_gravity()
    
    print("\n=== Testing with pressure disabled ===")
    test.test_planet_no_pressure()
    
    print("\n=== Testing with heat transfer disabled ===")
    test.test_planet_no_heat_transfer()
    
    print("\n=== Testing with solar disabled ===") 
    test.test_planet_no_solar()
    
    print("\n=== Testing simple air-space interface ===")
    test.test_simple_air_space_interface()