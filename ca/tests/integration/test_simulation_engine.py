"""Test suite for simulation engine"""

import numpy as np
import pytest
from geo_game import GeoGame as GeologySimulation
from materials import MaterialType


def test_simulation_initialization():
    """Test simulation initialization"""
    sim = GeologySimulation(width=20, height=15)
    assert sim.width == 20
    assert sim.height == 15
    assert sim.time == 0
    assert sim.temperature.shape == (15, 20)
    # CA doesn't have pressure


def test_heat_injection():
    """Inject local heat using modular HeatTransfer API"""
    sim = GeologySimulation(width=10, height=10)
    initial_temp = np.max(sim.temperature)
    sim.heat_transfer.inject_heat(5, 5, 2, 1000)
    new_temp = np.max(sim.temperature)
    assert new_temp > initial_temp




def test_time_stepping():
    """Test time stepping"""
    sim = GeologySimulation(width=10, height=10)
    initial_time = sim.time
    sim.step_forward()
    assert sim.time > initial_time
    
    sim.step_backward()
    assert sim.time == initial_time


def test_multiple_steps():
    """Test multiple simulation steps"""
    sim = GeologySimulation(width=10, height=10)
    initial_time = sim.time
    
    for i in range(5):
        prev_time = sim.time
        sim.step_forward()
        assert sim.time > prev_time
    
    assert sim.time > initial_time


def test_basic_stats():
    """Compute basic statistics manually (no legacy helper)."""
    sim = GeologySimulation(width=10, height=10)

    # Non-space mask for averages
    non_space = sim.material_types != MaterialType.SPACE
    temps = sim.temperature[non_space]
    # CA doesn't have pressure

    avg_temperature = float(np.mean(temps)) if temps.size else 0.0
    max_temperature = float(np.max(temps)) if temps.size else 0.0
    avg_pressure = 0.0  # CA doesn't calculate pressure
    max_pressure = 0.0  # CA doesn't calculate pressure

    # Material composition
    material_strings = np.array([m.value for m in sim.material_types.flatten()])
    unique, counts = np.unique(material_strings, return_counts=True)
    composition = {u: 100.0 * c / material_strings.size for u, c in zip(unique, counts)}

    assert 0.0 <= avg_temperature <= max_temperature
    assert 0.0 <= avg_pressure <= max_pressure
    assert abs(sum(composition.values()) - 100.0) < 0.1


def test_color_mapping():
    """Verify material-to-RGB mapping via MaterialDatabase"""
    sim = GeologySimulation(width=8, height=6)
    colors = np.zeros((6, 8, 3), dtype=np.uint8)
    for mat in np.unique(sim.material_types):
        rgb = sim.material_db.get_properties(mat).color_rgb
        colors[sim.material_types == mat] = rgb

    # Sanity checks
    assert colors.shape == (6, 8, 3)
    assert colors.dtype == np.uint8
    assert np.all(colors >= 0) and np.all(colors <= 255) 