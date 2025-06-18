"""
Pytest tests for chunk settling using the test framework.
"""

import pytest
from tests.test_framework import ScenarioRunner
from tests.test_chunk_settle_scenarios import (
    ChunkSettleScenario,
    MultiChunkSettleScenario
)
from materials import MaterialType


def test_chunk_settle_respects_terminal_velocity():
    """Unsupported solid chunk should fall at most `terminal_settle_velocity` cells per settle pass."""
    # Create scenario with terminal velocity of 3
    scenario = ChunkSettleScenario(
        chunk_material=MaterialType.BASALT,
        chunk_size=1,
        terminal_velocity=3.0
    )
    
    # Use small grid for this test
    runner = ScenarioRunner(scenario, sim_width=10, sim_height=10)
    
    # Run one step (chunk settling happens in fluid dynamics)
    result = runner.run_headless(max_steps=1)
    
    # Check that chunk moved by exactly terminal velocity
    fall_distance = result['metrics']['fall_distance']
    assert fall_distance == 3.0, f"Chunk should fall exactly 3 cells, fell {fall_distance}"
    assert result['success'], result['message']


def test_chunk_settle_inf_velocity_falls_all_the_way():
    """With terminal velocity = inf the chunk should fall until the first non-fluid cell or grid edge."""
    # Create scenario with infinite terminal velocity
    scenario = ChunkSettleScenario(
        chunk_material=MaterialType.BASALT,
        chunk_size=1,
        terminal_velocity=float('inf')
    )
    
    # Use small grid
    runner = ScenarioRunner(scenario, sim_width=10, sim_height=10)
    
    # Run one step
    result = runner.run_headless(max_steps=1)
    
    # Check that chunk fell to bottom
    current_y = result['metrics']['current_y']
    expected_y = 9  # Bottom row
    assert current_y == expected_y, f"Chunk should be at bottom (y={expected_y}), but is at y={current_y}"
    assert result['success'], result['message']


def test_multi_chunk_settling():
    """Test multiple chunks of different materials falling."""
    scenario = MultiChunkSettleScenario(terminal_velocity=5.0)
    runner = ScenarioRunner(scenario, sim_width=20, sim_height=20)
    
    # Run a few steps to let chunks fall
    result = runner.run_headless(max_steps=3)
    
    # All chunks should have fallen
    assert result['success'], result['message']
    
    # Check individual fall distances
    metrics = result['metrics']
    for material in ['basalt', 'granite', 'ice']:
        fall_key = f'{material}_fall'
        if fall_key in metrics:
            assert metrics[fall_key] > 0, f"{material} should have fallen"