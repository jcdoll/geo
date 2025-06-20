"""
Magma containment tests with integrated scenarios.

This file combines test scenarios and pytest tests for better organization.
"""

import numpy as np
import pytest
from typing import Dict, Any, Optional

from tests.framework.test_framework import TestScenario, ScenarioRunner
from materials import MaterialType
from geo_game import GeoGame


# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

class MagmaContainmentScenario(TestScenario):
    """Test scenario for magma containment by surrounding rock."""
    
    def __init__(self, scenario: str = 'small', **kwargs):
        """Initialize with scenario size.
        
        Args:
            scenario: 'small' or 'large' test configuration
        """
        super().__init__(scenario=scenario, **kwargs)
        self.scenario = scenario
        
    def get_name(self) -> str:
        return f"magma_containment_{self.scenario}"
        
    def get_description(self) -> str:
        return f"Tests if magma remains contained when surrounded by solid rock ({self.scenario} scenario)"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up magma surrounded by rock."""
        # Clear to space
        sim.material_types.fill(MaterialType.SPACE)
        sim.temperature.fill(2.7)  # Space temperature
        
        if self.scenario == 'small':
            self._setup_small_scenario(sim)
        else:
            self._setup_large_scenario(sim)
            
        # Update properties
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def _setup_small_scenario(self, sim: GeoGame):
        """Create small test: 5x5 magma core in basalt."""
        center_y, center_x = 25, 25
        
        # Magma core (5x5)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y, x = center_y + dy, center_x + dx
                sim.material_types[y, x] = MaterialType.MAGMA
                sim.temperature[y, x] = 1300.0 + 273.15
                
        # Basalt shell (15x15)
        for dy in range(-7, 8):
            for dx in range(-7, 8):
                y, x = center_y + dy, center_x + dx
                if abs(dy) > 2 or abs(dx) > 2:
                    sim.material_types[y, x] = MaterialType.BASALT
                    sim.temperature[y, x] = 800.0 + 273.15
                    
    def _setup_large_scenario(self, sim: GeoGame):
        """Create large test: circular magma pocket."""
        center_y, center_x = sim.height // 2, sim.width // 2
        
        # Circular magma pocket
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dy*dy + dx*dx <= 9:
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < sim.height and 0 <= x < sim.width:
                        sim.material_types[y, x] = MaterialType.MAGMA
                        sim.temperature[y, x] = 1300.0 + 273.15
                        
        # Basalt shell
        for dy in range(-12, 13):
            for dx in range(-12, 13):
                if dy*dy + dx*dx > 9 and dy*dy + dx*dx <= 144:
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < sim.height and 0 <= x < sim.width:
                        sim.material_types[y, x] = MaterialType.BASALT
                        sim.temperature[y, x] = 800.0 + 273.15
                        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate containment status."""
        # Get current magma positions
        magma_mask = (sim.material_types == MaterialType.MAGMA)
        magma_count = np.sum(magma_mask)
        
        if magma_count == 0:
            return {
                'success': False,
                'metrics': {'magma_count': 0, 'expansion': 0},
                'message': 'No magma found!'
            }
            
        magma_positions = set(zip(*np.where(magma_mask)))
        
        # Calculate bounds
        min_y = min(pos[0] for pos in magma_positions)
        max_y = max(pos[0] for pos in magma_positions)
        min_x = min(pos[1] for pos in magma_positions)
        max_x = max(pos[1] for pos in magma_positions)
        
        # Compare to initial bounds
        initial_positions = self.initial_state['material_positions'].get(MaterialType.MAGMA, set())
        if not initial_positions:
            return {
                'success': False,
                'metrics': {'magma_count': magma_count},
                'message': 'No initial magma positions stored'
            }
            
        initial_min_y = min(pos[0] for pos in initial_positions)
        initial_max_y = max(pos[0] for pos in initial_positions)
        initial_min_x = min(pos[1] for pos in initial_positions)
        initial_max_x = max(pos[1] for pos in initial_positions)
        
        # Calculate expansion
        expansion = max(
            initial_min_y - min_y,
            max_y - initial_max_y,
            initial_min_x - min_x,
            max_x - initial_max_x
        )
        
        # Success criteria
        contained = expansion <= 2
        
        return {
            'success': contained,
            'metrics': {
                'magma_count': magma_count,
                'expansion': expansion,
                'bounds': f"({min_y},{max_y})x({min_x},{max_x})"
            },
            'message': f"Magma {'contained' if contained else 'NOT contained'}: expansion={expansion} cells"
        }


class MagmaNoPhysicsScenario(MagmaContainmentScenario):
    """Baseline test with all physics disabled."""
    
    def get_name(self) -> str:
        return f"magma_no_physics_{self.scenario}"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up scenario and disable all physics."""
        super().setup(sim)
        
        # Disable all physics
        sim.enable_heat_diffusion = False
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        sim.fluid_dynamics.calculate_planetary_pressure = lambda: None
        sim.fluid_dynamics.apply_unified_kinematics = lambda dt: None
        sim.material_processes.apply_metamorphism = lambda: None
        sim.material_processes.apply_phase_transitions = lambda: None
        sim.material_processes.apply_weathering = lambda: None
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """With no physics, there should be NO movement."""
        result = super().evaluate(sim)
        result['success'] = result['metrics']['expansion'] == 0
        return result


class GraniteVacuumScenario(TestScenario):
    """Test that granite doesn't spontaneously melt in vacuum."""
    
    def get_name(self) -> str:
        return "granite_vacuum_stability"
        
    def get_description(self) -> str:
        return "Granite in pure vacuum should not convert to magma"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up granite blob in vacuum."""
        # Wipe to pure space
        sim.material_types[:, :] = MaterialType.SPACE
        sim.temperature[:, :] = 300.0
        
        # Paint a granite blob near the top-center
        granite_center = (3, sim.width // 2)
        granite_radius = 5
        
        # Create granite blob
        yy, xx = np.ogrid[:sim.height, :sim.width]
        g_mask = (xx - granite_center[1]) ** 2 + (yy - granite_center[0]) ** 2 <= granite_radius ** 2
        sim.material_types[g_mask] = MaterialType.GRANITE
        sim.temperature[g_mask] = 300.0
        
        # Update properties
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check that granite remains granite."""
        granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
        magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
        
        success = granite_count > 0 and magma_count == 0
        
        return {
            'success': success,
            'metrics': {
                'granite_count': granite_count,
                'magma_count': magma_count
            },
            'message': f"Granite: {granite_count} cells, Magma: {magma_count} cells"
        }


# ============================================================================
# PYTEST TESTS
# ============================================================================

def test_magma_containment_small():
    """Test small scenario magma containment."""
    scenario = MagmaContainmentScenario('small')
    runner = ScenarioRunner(scenario, sim_width=50, sim_height=50)
    result = runner.run_headless(max_steps=30)
    assert result['success'], f"Small scenario failed: {result['message']}"


def test_magma_containment_large():
    """Test large scenario magma containment."""
    scenario = MagmaContainmentScenario('large')
    runner = ScenarioRunner(scenario, sim_width=80, sim_height=80)
    result = runner.run_headless(max_steps=30)
    assert result['success'], f"Large scenario failed: {result['message']}"


def test_magma_no_physics():
    """Test that magma doesn't move with all physics disabled."""
    scenario = MagmaNoPhysicsScenario('small')
    runner = ScenarioRunner(scenario, sim_width=50, sim_height=50)
    result = runner.run_headless(max_steps=10)
    assert result['success'], f"Baseline test failed: {result['message']}"


def test_granite_vacuum_stability():
    """Test that granite doesn't spontaneously melt in vacuum."""
    scenario = GraniteVacuumScenario()
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=1)
    assert result['success'], f"Granite melted in vacuum: {result['message']}"


@pytest.mark.slow
def test_magma_containment_long_term():
    """Long-term magma containment test."""
    scenario = MagmaContainmentScenario('large')
    runner = ScenarioRunner(scenario, sim_width=80, sim_height=80)
    result = runner.run_headless(max_steps=100)
    assert result['success'], f"Long-term test failed: {result['message']}"


# ============================================================================
# SCENARIO REGISTRY FOR VISUAL RUNNER
# ============================================================================

SCENARIOS = {
    'magma_small': lambda: MagmaContainmentScenario('small'),
    'magma_large': lambda: MagmaContainmentScenario('large'),
    'magma_no_physics': lambda: MagmaNoPhysicsScenario('small'),
    'granite_vacuum': lambda: GraniteVacuumScenario(),
} 