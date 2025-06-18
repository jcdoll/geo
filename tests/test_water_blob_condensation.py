"""
Pytest tests for water blob condensation using the test framework.
"""

import pytest
from tests.test_framework import ScenarioRunner
from tests.test_water_blob_scenario import WaterBlobCondensationScenario


def test_water_condenses_to_circular_blob():
    """Test that a water bar condenses into a circular blob."""
    # Use default parameters (30x4 bar)
    scenario = WaterBlobCondensationScenario()
    runner = ScenarioRunner(scenario, sim_width=40, sim_height=40)
    
    # Run for 120 steps as in the original test
    result = runner.run_headless(max_steps=120)
    
    # Check that it succeeded
    assert result['success'], f"Water blob test failed: {result['message']}"
    
    # Additional assertion on the aspect ratio
    aspect_ratio = result['metrics']['aspect_ratio']
    assert aspect_ratio < 1.6, f"Water blob still elongated (ratio {aspect_ratio:.2f})"


def test_water_blob_different_sizes():
    """Test water condensation with different initial bar sizes."""
    test_cases = [
        (20, 2),  # Thin bar
        (40, 6),  # Wide bar
        (10, 10), # Square (should already be ~circular)
    ]
    
    for width, height in test_cases:
        scenario = WaterBlobCondensationScenario(bar_width=width, bar_height=height)
        runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
        
        result = runner.run_headless(max_steps=150)
        
        # Square should already be circular, others should converge
        if width == height:
            assert result['metrics']['aspect_ratio'] < 1.2, f"Square didn't stay circular"
        else:
            assert result['success'], f"Test failed for {width}x{height}: {result['message']}" 