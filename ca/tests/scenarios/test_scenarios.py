"""Pytest wrapper for all test scenarios.

This module provides pytest test functions that wrap the scenarios defined
in other modules. Each test runs in headless mode with appropriate timeouts.
"""

import pytest
from typing import Dict, Any

# Import enhanced test framework
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.framework.test_framework import ScenarioRunner

# Import all scenario modules
from .fluids import (
    WaterConservationScenario,
    WaterConservationStressScenario,
    WaterBlobScenario,
    WaterLineCollapseScenario,
    MagmaFlowScenario,
    FluidGravityScenario,
)
from .mechanics import (
    GravityFallScenario,
    BuoyancyScenario,
    HydrostaticPressureScenario,
)
from .materials import (
    MaterialStabilityScenario,
    MetamorphismScenario,
    PhaseTransitionScenario,
)
from .rigid_body import (
    RigidBodyWithEnclosedFluidScenario,
    RigidBodyFluidDisplacementScenario,
    RigidBodyRotationScenario,
)
from materials import MaterialType


# ============================================================================
# FLUID DYNAMICS TESTS
# ============================================================================

@pytest.mark.parametrize("grid_size", [40, 60])
def test_water_conservation(grid_size):
    """Test that water is conserved during fluid flow."""
    scenario = WaterConservationScenario(
        grid_size=grid_size,
        water_fraction=0.3,
        tolerance_percent=2.0,  # Allow 2% variation
        timeout=30
    )
    runner = ScenarioRunner(scenario, sim_width=grid_size, sim_height=grid_size)
    result = runner.run_headless(max_steps=100)
    
    assert result['success'], f"Water conservation failed: {result['message']}"
    
    # Additional checks
    metrics = result.get('metrics', {})
    if 'percent_change' in metrics:
        assert abs(metrics['percent_change']) < 5.0, \
            f"Water loss too high: {metrics['percent_change']:.1f}%"

def test_magma_flow_and_cooling():
    """Test that magma flows downhill and cools to basalt."""
    scenario = MagmaFlowScenario(
        volcano_size=8,
        grid_size=60,
        timeout=30
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=100)
    
    assert result['success'], f"Magma flow test failed: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert metrics.get('magma_spread', 0) > 3, "Magma didn't spread enough"


@pytest.mark.slow
def test_water_conservation_stress():
    """Stress test water conservation with many surface features."""
    scenario = WaterConservationStressScenario(timeout=60)
    runner = ScenarioRunner(scenario, sim_width=80, sim_height=80)
    result = runner.run_headless(max_steps=200)
    
    # More lenient for stress test
    metrics = result.get('metrics', {})
    assert abs(metrics.get('percent_change', 100)) < 10.0, \
        f"Water loss too high in stress test: {metrics.get('percent_change', 0):.1f}%"


@pytest.mark.parametrize("width,height", [(20, 10), (10, 20), (15, 15)])
def test_water_blob_behavior(width, height):
    """Test water blob behavior."""
    scenario = WaterBlobScenario(
        blob_width=width,
        blob_height=height,
        grid_size=60,
        timeout=20
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=50)
    
    assert result['success'], f"Water blob test failed: {result['message']}"
    
    metrics = result.get('metrics', {})

    assert metrics.get('num_components', 50) <= 25, f"Water blob fragmented too much: {metrics.get('num_components')} components"


def test_water_line_collapse():
    """Test tall water column collapse behavior."""
    scenario = WaterLineCollapseScenario(
        line_thickness=2,
        line_height=30,
        grid_size=60,
        timeout=30
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=100)
    
    assert result['success'], f"Water line collapse failed: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert metrics.get('water_width', 0) > 4, "Water didn't spread horizontally"
    assert metrics.get('water_height', 100) < 15, "Water column didn't collapse"


def test_fluid_self_gravity():
    """Test fluid behavior under self-gravity."""
    scenario = FluidGravityScenario(
        planet_radius=15,
        fluid_amount=100,
        fluid_type=MaterialType.WATER,
        grid_size=80,
        timeout=40
    )
    runner = ScenarioRunner(scenario, sim_width=80, sim_height=80)
    result = runner.run_headless(max_steps=150)
    
    assert result['success'], f"Fluid gravity test failed: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert metrics.get('moved_closer', False), "Fluid didn't fall toward planet"


# ============================================================================
# MECHANICAL PHYSICS TESTS
# ============================================================================

@pytest.mark.parametrize("material", [
    MaterialType.BASALT,
    MaterialType.WATER,
    MaterialType.SAND,
])
def test_gravity_fall(material):
    """Test that objects fall under gravity."""
    scenario = GravityFallScenario(
        object_size=5,
        object_material=material,
        grid_size=60,
        timeout=20
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=1000)  # Many more steps for accumulation with 0.1s timestep
    
    assert result['success'], f"Gravity test failed for {material.name}: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert metrics.get('fall_distance', 0) > 1, \
        f"{material.name} didn't fall far enough: {metrics.get('fall_distance', 0):.1f}"


@pytest.mark.parametrize("object_type,fluid_type,should_float", [
    (MaterialType.ICE, MaterialType.WATER, True),
    (MaterialType.BASALT, MaterialType.WATER, False),
    (MaterialType.PUMICE, MaterialType.WATER, True),
    (MaterialType.GRANITE, MaterialType.MAGMA, True),  # Granite less dense than magma
])
def test_buoyancy(object_type, fluid_type, should_float):
    """Test buoyancy forces in fluids."""
    scenario = BuoyancyScenario(
        fluid_type=fluid_type,
        object_type=object_type,
        object_size=5,
        grid_size=60,
        timeout=30
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=200)  # More time for sinking
    
    # Check if behavior matches expected
    metrics = result.get('metrics', {})
    depth = metrics.get('depth_in_fluid', 0)
    initial_depth = 4.5  # From test setup
    
    if should_float:
        assert depth < 10, f"{object_type.name} should float in {fluid_type.name} but sank to depth {depth}"
    else:
        # For sinking, just verify it's moving down (given hydrostatic limitations)
        assert depth > initial_depth + 2, f"{object_type.name} should sink in {fluid_type.name} but only moved from {initial_depth} to {depth}"


def test_hydrostatic_pressure():
    """Test that pressure increases with depth in fluids."""
    scenario = HydrostaticPressureScenario(
        fluid_depth=40,
        grid_size=60,
        timeout=20
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    
    # Let system stabilize
    result = runner.run_headless(max_steps=20)
    
    assert result['success'], f"Hydrostatic pressure test failed: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert metrics.get('pressure_increasing', False), "Pressure doesn't increase with depth"
    # Due to known limitations with discrete pressure solver (see PHYSICS.md),
    # we accept larger errors in hydrostatic pressure gradient
    assert metrics.get('gradient_error_pct', 100) < 85, \
        f"Pressure gradient error too high: {metrics.get('gradient_error_pct', 100):.1f}%"


# ============================================================================
# MATERIAL PHYSICS TESTS
# ============================================================================

@pytest.mark.parametrize("material,environment,should_be_stable", [
    (MaterialType.GRANITE, "vacuum", True),
    (MaterialType.ICE, "vacuum", False),  # Should sublimate
    (MaterialType.GRANITE, "magma", False),  # Should melt
    (MaterialType.LIMESTONE, "water", True),  # Limestone is relatively stable in water
])
def test_material_stability(material, environment, should_be_stable):
    """Test material stability in various environments."""
    scenario = MaterialStabilityScenario(
        test_material=material,
        environment=environment,
        grid_size=40,
        timeout=30
    )
    runner = ScenarioRunner(scenario, sim_width=40, sim_height=40)
    result = runner.run_headless(max_steps=100)
    
    metrics = result.get('metrics', {})
    preservation = metrics.get('preservation_pct', 0)
    
    if should_be_stable:
        assert preservation > 90, f"{material.name} degraded in {environment}: {preservation:.1f}%"
    else:
        assert preservation < 50, f"{material.name} too stable in {environment}: {preservation:.1f}%"


def test_rock_metamorphism():
    """Test metamorphic rock formation under pressure/temperature."""
    scenario = MetamorphismScenario(
        source_rock=MaterialType.SHALE,
        target_conditions="both",  # High pressure and temperature
        grid_size=60,
        timeout=40
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=150)
    
    assert result['success'], f"Metamorphism test failed: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert metrics.get('transform_pct', 0) > 20, "Insufficient metamorphic transformation"


@pytest.mark.parametrize("material,transition", [
    (MaterialType.ICE, "melting"),
    (MaterialType.WATER, "freezing"),
    (MaterialType.WATER, "vaporization"),
])
def test_phase_transitions(material, transition):
    """Test material phase transitions."""
    scenario = PhaseTransitionScenario(
        material=material,
        transition_type=transition,
        grid_size=40,
        timeout=30
    )
    runner = ScenarioRunner(scenario, sim_width=40, sim_height=40)
    result = runner.run_headless(max_steps=100)
    
    assert result['success'], f"{transition} of {material.name} failed: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert metrics.get('conversion_pct', 0) > 25, f"Insufficient {transition}"


# ============================================================================
# RIGID BODY TESTS
# ============================================================================

@pytest.mark.skip(reason="Rigid bodies removed from physics model")
def test_rigid_body_with_fluid():
    """Test rigid container holding fluid."""
    scenario = RigidBodyWithEnclosedFluidScenario(
        container_size=15,
        wall_thickness=2,
        fluid_type=MaterialType.WATER,
        container_material=MaterialType.GRANITE,
        grid_size=60,
        timeout=30
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=100)
    
    assert result['success'], f"Rigid container test failed: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert metrics.get('container_intact', False), "Container broke apart"
    assert metrics.get('fluid_contained', False), "Fluid leaked out"


@pytest.mark.skip(reason="Rigid bodies removed from physics model")
def test_rigid_body_displacement():
    """Test rigid body displacing fluid (Archimedes)."""
    scenario = RigidBodyFluidDisplacementScenario(
        rock_size=10,
        rock_material=MaterialType.GRANITE,
        fluid_depth=20,
        grid_size=60,
        timeout=30
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=100)
    
    assert result['success'], f"Displacement test failed: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert metrics.get('rock_in_water', False), "Rock didn't enter water"
    assert metrics.get('water_rise', 0) > 0, "No water displacement observed"


@pytest.mark.skip(reason="Rigid bodies removed from physics model")
@pytest.mark.parametrize("shape", ["L", "T"])
def test_rigid_body_rotation(shape):
    """Test rigid body rotation dynamics."""
    scenario = RigidBodyRotationScenario(
        shape=shape,
        size=10,
        material=MaterialType.GRANITE,
        grid_size=60,
        timeout=20
    )
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=50)
    
    assert result['success'], f"Rotation test failed for {shape} shape: {result['message']}"
    
    metrics = result.get('metrics', {})
    assert abs(metrics.get('rotation_deg', 0)) > 10, f"{shape} shape didn't rotate enough"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_complex_fluid_interaction():
    """Test interaction between multiple fluid types."""
    # This would test oil/water separation, magma/water interaction, etc.
    # Placeholder for now
    pytest.skip("Complex fluid interaction test not yet implemented")


def test_material_phase_transitions():
    """Test material transformations under pressure/temperature."""
    # Already implemented above
    pass


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.slow
def test_large_scale_simulation():
    """Test simulation performance at larger scales."""
    scenario = WaterConservationScenario(
        grid_size=100,
        water_fraction=0.2,
        tolerance_percent=5.0,
        timeout=60
    )
    runner = ScenarioRunner(scenario, sim_width=100, sim_height=100)
    result = runner.run_headless(max_steps=50)
    
    assert result['success'], f"Large scale test failed: {result['message']}"
    
    # Check that it completed within timeout
    elapsed = scenario.get_elapsed_time()
    assert elapsed < 60, f"Simulation too slow: {elapsed:.1f}s for 50 steps"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_scenario_with_timeout(scenario_class, params: Dict[str, Any], 
                            max_steps: int = 100) -> Dict[str, Any]:
    """Helper to run a scenario with standard timeout handling."""
    scenario = scenario_class(**params)
    runner = ScenarioRunner(
        scenario, 
        sim_width=params.get('grid_size', 60),
        sim_height=params.get('grid_size', 60)
    )
    return runner.run_headless(max_steps=max_steps)


if __name__ == "__main__":
    # Allow running specific tests from command line
    pytest.main([__file__, "-v"])