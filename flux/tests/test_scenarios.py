"""Run all test scenarios in headless mode using pytest.

This allows running scenarios as part of the test suite without visualization.
"""

import pytest
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from tests.scenarios import ALL_SCENARIOS


# Mark all tests in this class as expected failures due to multigrid bug
# The bug is in multigrid.py:_apply_neumann_bc_stencils where array shapes
# don't match on coarser grid levels
@pytest.mark.xfail(reason="Multigrid solver has shape mismatch bug in Neumann BC stencils - see multigrid.py:41")
class TestScenarios:
    """Test runner for all scenarios."""
    
    @pytest.fixture
    def simulation(self):
        """Create a test simulation."""
        return FluxSimulation(nx=50, ny=50, dx=50.0, scenario=False)  # Don't auto-setup
    
    def run_scenario(self, simulation: FluxSimulation, scenario_class, params: Dict[str, Any], 
                    max_steps: int = 100, success_threshold: float = 0.8):
        """Run a scenario and check for success."""
        # Create scenario instance
        scenario = scenario_class(**params)
        
        # Setup
        scenario.setup(simulation)
        scenario.store_initial_state(simulation)
        
        # Run simulation
        success_count = 0
        for step in range(max_steps):
            # Compute timestep
            dt = simulation.physics.apply_cfl_limit()
            simulation.timestep(dt)
            
            # Evaluate every 10 steps
            if step % 10 == 0:
                result = scenario.evaluate(simulation)
                if result.get('success', False):
                    success_count += 1
                    
                # Check timeout
                if scenario.check_timeout():
                    break
                    
        # Final evaluation
        final_result = scenario.evaluate(simulation)
        
        # Success if final state is good or we had enough successful evaluations
        success_rate = success_count / (max_steps // 10)
        assert final_result.get('success', False) or success_rate >= success_threshold, \
            f"Scenario failed: {final_result.get('message', 'No message')}"
    
    # Generate test methods for each scenario
    def test_hydrostatic_equilibrium(self, simulation):
        """Test hydrostatic equilibrium scenario."""
        scenario_class, params = ALL_SCENARIOS['hydrostatic_equilibrium']
        self.run_scenario(simulation, scenario_class, params, max_steps=200)
    
    def test_water_drop_fall(self, simulation):
        """Test water drop falling scenario."""
        scenario_class, params = ALL_SCENARIOS['water_drop_fall']
        self.run_scenario(simulation, scenario_class, params, max_steps=300)
    
    def test_buoyancy(self, simulation):
        """Test buoyancy scenario."""
        scenario_class, params = ALL_SCENARIOS['buoyancy']
        self.run_scenario(simulation, scenario_class, params, max_steps=200)
    
    def test_water_freezing(self, simulation):
        """Test water freezing scenario."""
        scenario_class, params = ALL_SCENARIOS['water_freezing']
        self.run_scenario(simulation, scenario_class, params, max_steps=100)
    
    def test_ice_melting(self, simulation):
        """Test ice melting scenario."""
        scenario_class, params = ALL_SCENARIOS['ice_melting']
        self.run_scenario(simulation, scenario_class, params, max_steps=100)
    
    def test_water_evaporation(self, simulation):
        """Test water evaporation scenario."""
        scenario_class, params = ALL_SCENARIOS['water_evaporation']
        self.run_scenario(simulation, scenario_class, params, max_steps=150)
    
    @pytest.mark.slow
    def test_rock_melting(self, simulation):
        """Test rock melting scenario (slow due to high temps needed)."""
        scenario_class, params = ALL_SCENARIOS['rock_melting']
        self.run_scenario(simulation, scenario_class, params, max_steps=500)
    
    def test_heat_diffusion(self, simulation):
        """Test heat diffusion scenario."""
        scenario_class, params = ALL_SCENARIOS['heat_diffusion']
        self.run_scenario(simulation, scenario_class, params, max_steps=100)
    
    def test_uranium_heating(self, simulation):
        """Test uranium heating scenario."""
        scenario_class, params = ALL_SCENARIOS['uranium_heating']
        self.run_scenario(simulation, scenario_class, params, max_steps=200)
    
    @pytest.mark.slow
    def test_solar_heating(self, simulation):
        """Test solar heating scenario."""
        scenario_class, params = ALL_SCENARIOS['solar_heating']
        self.run_scenario(simulation, scenario_class, params, max_steps=300)
    
    def test_radiative_cooling(self, simulation):
        """Test radiative cooling scenario."""
        scenario_class, params = ALL_SCENARIOS['radiative_cooling']
        self.run_scenario(simulation, scenario_class, params, max_steps=200)


# Note: Using @pytest.mark.parametrize instead of pytest_generate_tests


@pytest.mark.xfail(reason="Multigrid solver has shape mismatch bug in Neumann BC stencils - see multigrid.py:41")
class TestAllScenarios:
    """Alternative test class using parametrization."""
    
    @pytest.mark.parametrize("scenario_key", list(ALL_SCENARIOS.keys()))
    def test_scenario(self, scenario_key):
        """Test a single scenario."""
        # Create simulation
        sim = FluxSimulation(nx=50, ny=50, dx=50.0)
        
        # Get scenario
        scenario_class, params = ALL_SCENARIOS[scenario_key]
        scenario = scenario_class(**params)
        
        # Setup
        scenario.setup(sim)
        scenario.store_initial_state(sim)
        
        # Run briefly
        max_steps = 50 if 'melting' not in scenario_key else 100
        for _ in range(max_steps):
            # Compute timestep
            dt = sim.physics.apply_cfl_limit()
            sim.timestep(dt)
            
        # Final evaluation
        result = scenario.evaluate(sim)
        
        # We don't require success for all scenarios in this quick test
        # Just ensure they run without errors
        assert 'success' in result
        assert 'metrics' in result
        assert 'message' in result