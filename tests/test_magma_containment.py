"""
Pytest tests for magma containment using the test framework scenarios.

This replaces the old test_magma_containment.py with a cleaner approach.
"""

import pytest
from tests.test_framework import ScenarioRunner, ModuleDisabler
from tests.test_magma_containment_scenarios import (
    MagmaContainmentScenario,
    MagmaContainmentNoPhysicsScenario,
    MagmaContainmentHeatOnlyScenario,
    MagmaContainmentFluidOnlyScenario,
    MagmaContainmentMaterialOnlyScenario,
    MagmaContainmentGravityOnlyScenario,
    MagmaBindingForceScenario
)


class TestMagmaContainment:
    """Pytest suite for magma containment tests."""
    
    def test_baseline_no_physics(self):
        """Test that magma doesn't move with all physics disabled."""
        scenario = MagmaContainmentNoPhysicsScenario('small')
        runner = ScenarioRunner(scenario)
        result = runner.run_headless(max_steps=10)
        assert result['success'], f"Baseline test failed: {result['message']}"
        
    def test_heat_transfer_only(self):
        """Test that heat transfer alone doesn't cause magma movement."""
        scenario = MagmaContainmentHeatOnlyScenario('small')
        runner = ScenarioRunner(scenario)
        result = runner.run_headless(max_steps=10)
        assert result['success'], f"Heat transfer test failed: {result['message']}"
        
    def test_fluid_dynamics_only(self):
        """Test that fluid dynamics respects binding forces."""
        scenario = MagmaContainmentFluidOnlyScenario('small')
        runner = ScenarioRunner(scenario)
        result = runner.run_headless(max_steps=10)
        assert result['success'], f"Fluid dynamics test failed: {result['message']}"
        
    def test_material_processes_only(self):
        """Test that material processes alone don't cause movement."""
        scenario = MagmaContainmentMaterialOnlyScenario('small')
        runner = ScenarioRunner(scenario)
        result = runner.run_headless(max_steps=10)
        assert result['success'], f"Material processes test failed: {result['message']}"
        
    def test_self_gravity_only(self):
        """Test that self-gravity alone doesn't cause movement."""
        scenario = MagmaContainmentGravityOnlyScenario('small')
        runner = ScenarioRunner(scenario)
        result = runner.run_headless(max_steps=10)
        assert result['success'], f"Self-gravity test failed: {result['message']}"
        
    def test_full_simulation(self):
        """Test magma containment with all physics enabled."""
        scenario = MagmaContainmentScenario('small')
        runner = ScenarioRunner(scenario)
        result = runner.run_headless(max_steps=20)
        assert result['success'], f"Full simulation failed: {result['message']}"
        
    def test_binding_forces(self):
        """Test that binding forces are properly configured."""
        scenario = MagmaBindingForceScenario('small')
        runner = ScenarioRunner(scenario)
        result = runner.run_headless(max_steps=5)
        assert result['success'], f"Binding force test failed: {result['message']}"
        

# Simple functional tests
def test_magma_containment_small():
    """Test small scenario magma containment."""
    scenario = MagmaContainmentScenario('small')
    runner = ScenarioRunner(scenario)
    result = runner.run_headless(max_steps=30)
    assert result['success'], f"Small scenario failed: {result['message']}"


def test_magma_containment_large():
    """Test large scenario magma containment."""
    scenario = MagmaContainmentScenario('large')
    runner = ScenarioRunner(scenario, sim_width=80, sim_height=80)
    result = runner.run_headless(max_steps=30)
    assert result['success'], f"Large scenario failed: {result['message']}"


@pytest.mark.slow
def test_magma_containment_long_term():
    """Long-term magma containment test."""
    scenario = MagmaContainmentScenario('large')
    runner = ScenarioRunner(scenario, sim_width=80, sim_height=80)
    result = runner.run_headless(max_steps=100)
    assert result['success'], f"Long-term test failed: {result['message']}" 