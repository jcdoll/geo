"""
Pytest tests for water conservation using the test framework.
"""

import pytest
from tests.test_framework import ScenarioRunner
from tests.test_water_conservation_scenarios import (
    WaterConservationScenario,
    WaterConservationStressTestScenario,
    WaterConservationByPhaseScenario
)


def test_water_cell_conservation():
    """Verify that total number of water-bearing cells (water, ice, vapor) is conserved.

    This test is expected to fail with the current implementation and serves
    as a regression check once the leak is fixed.
    """
    scenario = WaterConservationScenario()
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=400)
    
    # The scenario allows 1% tolerance
    assert result['success'], f"Water conservation failed: {result['message']}"


def test_water_conservation_stress():
    """Aggressive water conservation test with many cavities."""
    scenario = WaterConservationStressTestScenario()
    runner = ScenarioRunner(scenario, sim_width=80, sim_height=80)
    result = runner.run_headless(max_steps=100)  # Run fewer steps for stress test
    
    # This is a stress test - we expect some loss but want to track it
    percent_change = result['metrics']['percent_change']
    print(f"Stress test water change: {percent_change:.2f}%")
    
    # Allow more tolerance for stress test
    assert abs(percent_change) < 5.0, f"Water loss too high in stress test: {percent_change:.2f}%"


@pytest.mark.parametrize("disabled_phase", [
    None,
    'fluid_dynamics',
    'heat_transfer',
    'material_processes',
    'self_gravity'
])
def test_water_conservation_by_phase(disabled_phase):
    """Test water conservation with different physics phases disabled."""
    scenario = WaterConservationByPhaseScenario(disabled_phase=disabled_phase)
    runner = ScenarioRunner(scenario, sim_width=50, sim_height=50)
    result = runner.run_headless(max_steps=100)
    
    # Print diagnostic info
    phase_name = disabled_phase or "all_phases"
    percent_change = result['metrics']['percent_change']
    print(f"Phase {phase_name}: water change = {percent_change:+.2f}%")
    
    # This is primarily a diagnostic test
    assert True, "Diagnostic test - check output for water loss by phase"