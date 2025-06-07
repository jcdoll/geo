"""Test suite for simulation engine"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_engine import GeologySimulation


def test_simulation_initialization():
    """Test simulation initialization"""
    sim = GeologySimulation(width=20, height=15)
    assert sim.width == 20
    assert sim.height == 15
    assert sim.time == 0
    assert sim.temperature.shape == (15, 20)
    assert sim.pressure.shape == (15, 20)


def test_heat_source():
    """Test adding heat sources"""
    sim = GeologySimulation(width=10, height=10)
    initial_temp = np.max(sim.temperature)
    sim.add_heat_source(5, 5, 2, 1000)
    new_temp = np.max(sim.temperature)
    assert new_temp > initial_temp


def test_tectonic_stress():
    """Test applying tectonic stress"""
    sim = GeologySimulation(width=10, height=10)
    initial_pressure = np.max(sim.pressure)
    sim.apply_tectonic_stress(5, 5, 2, 100)
    new_pressure = np.max(sim.pressure)
    assert new_pressure > initial_pressure


def test_time_stepping():
    """Test time stepping"""
    sim = GeologySimulation(width=10, height=10)
    initial_time = sim.time
    sim.step_forward()
    assert sim.time > initial_time
    
    success = sim.step_backward()
    assert success
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


def test_statistics():
    """Test simulation statistics"""
    sim = GeologySimulation(width=10, height=10)
    stats = sim.get_stats()
    
    required_keys = ['time', 'avg_temperature', 'max_temperature', 
                    'avg_pressure', 'max_pressure', 'material_composition', 
                    'dt', 'history_length']
    
    for key in required_keys:
        assert key in stats
    
    assert len(stats['material_composition']) > 0
    
    # Material composition should sum to ~100%
    total = sum(stats['material_composition'].values())
    assert abs(total - 100.0) < 0.1


def test_visualization_data():
    """Test getting visualization data"""
    sim = GeologySimulation(width=8, height=6)
    colors, temp, pressure = sim.get_visualization_data()
    
    assert colors.shape == (6, 8, 3)
    assert temp.shape == (6, 8)
    assert pressure.shape == (6, 8)
    assert colors.dtype == np.uint8
    assert np.all(colors >= 0) and np.all(colors <= 255) 