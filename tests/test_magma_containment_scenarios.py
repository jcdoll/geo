"""
Magma containment test scenarios using the test framework.

This module defines various magma containment scenarios that can be run
both as pytest tests and with visualization.
"""

import numpy as np
from typing import Dict, Any, Optional

from tests.test_framework import TestScenario
from materials import MaterialType
from geo_game import GeoGame


class MagmaContainmentScenario(TestScenario):
    """Test scenario for magma containment by surrounding rock."""
    
    def __init__(self, scenario: str = 'small', **kwargs):
        """Initialize with scenario size.
        
        Args:
            scenario: 'small' or 'large' test configuration
        """
        super().__init__(scenario=scenario, **kwargs)
        self.scenario = scenario
        
        # Set dimensions based on scenario
        if scenario == 'small':
            self.sim_width = 50
            self.sim_height = 50
        elif scenario == 'large':
            self.sim_width = 80
            self.sim_height = 80
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
            
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
                        
        # Outer granite shell
        for dy in range(-20, 21):
            for dx in range(-20, 21):
                if dy*dy + dx*dx > 144 and dy*dy + dx*dx <= 400:
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < sim.height and 0 <= x < sim.width:
                        sim.material_types[y, x] = MaterialType.GRANITE
                        sim.temperature[y, x] = 500.0 + 273.15
                        
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
        
        # Count changes
        initial_count = self.initial_state['material_counts'].get(MaterialType.MAGMA, 0)
        count_change = magma_count - initial_count
        
        # Success criteria
        contained = expansion <= 2 and count_change <= 5
        
        return {
            'success': contained,
            'metrics': {
                'magma_count': magma_count,
                'initial_count': initial_count,
                'expansion': expansion,
                'count_change': count_change,
                'bounds': f"({min_y},{max_y})x({min_x},{max_x})"
            },
            'message': f"Magma {'contained' if contained else 'NOT contained'}: "
                      f"expansion={expansion} cells, count_change={count_change}"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        # Focus on the magma region
        if self.scenario == 'small':
            return {
                'focus_region': (18, 32, 18, 32),  # Center around the magma
                'highlight_materials': [MaterialType.MAGMA],
                'show_metrics': ['expansion', 'magma_count', 'bounds']
            }
        else:
            center = 40 if self.scenario == 'large' else 25
            return {
                'focus_region': (center-25, center+25, center-25, center+25),
                'highlight_materials': [MaterialType.MAGMA],
                'show_metrics': ['expansion', 'magma_count', 'bounds']
            }


class MagmaContainmentNoPhysicsScenario(MagmaContainmentScenario):
    """Magma containment with all physics disabled (baseline test)."""
    
    def get_name(self) -> str:
        return f"magma_containment_no_physics_{self.scenario}"
        
    def get_description(self) -> str:
        return f"Baseline test: magma with ALL physics disabled should not move ({self.scenario})"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up scenario and disable all physics."""
        super().setup(sim)
        
        # Disable all physics modules
        sim.enable_heat_diffusion = False
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        
        # Disable fluid dynamics
        sim.fluid_dynamics.calculate_planetary_pressure = lambda: None
        sim.fluid_dynamics.apply_unified_kinematics = lambda dt: None
        
        # Disable material processes
        sim.material_processes.apply_metamorphism = lambda: None
        sim.material_processes.apply_phase_transitions = lambda: None
        sim.material_processes.apply_weathering = lambda: None
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate - with no physics, there should be NO movement."""
        result = super().evaluate(sim)
        
        # Override success criteria - ANY movement is a failure
        expansion = result['metrics']['expansion']
        count_change = result['metrics']['count_change']
        
        result['success'] = expansion == 0 and count_change == 0
        result['message'] = f"No physics test: expansion={expansion}, count_change={count_change}"
        
        return result


class MagmaContainmentHeatOnlyScenario(MagmaContainmentScenario):
    """Magma containment with only heat transfer enabled."""
    
    def get_name(self) -> str:
        return f"magma_containment_heat_only_{self.scenario}"
        
    def get_description(self) -> str:
        return f"Test with ONLY heat transfer enabled ({self.scenario})"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up scenario with only heat transfer."""
        super().setup(sim)
        
        # Keep heat transfer enabled, disable everything else
        sim.fluid_dynamics.calculate_planetary_pressure = lambda: None
        sim.fluid_dynamics.apply_unified_kinematics = lambda dt: None
        sim.material_processes.apply_metamorphism = lambda: None
        sim.material_processes.apply_phase_transitions = lambda: None
        sim.material_processes.apply_weathering = lambda: None
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Heat alone should not cause movement."""
        result = super().evaluate(sim)
        
        # Heat transfer alone should cause minimal expansion
        expansion = result['metrics']['expansion']
        result['success'] = expansion <= 1
        result['message'] = f"Heat only: expansion={expansion} (should be ≤1)"
        
        return result


class MagmaContainmentFluidOnlyScenario(MagmaContainmentScenario):
    """Magma containment with only fluid dynamics enabled."""
    
    def get_name(self) -> str:
        return f"magma_containment_fluid_only_{self.scenario}"
        
    def get_description(self) -> str:
        return f"Test with ONLY fluid dynamics enabled ({self.scenario})"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up scenario with only fluid dynamics."""
        super().setup(sim)
        
        # Disable heat transfer
        sim.enable_heat_diffusion = False
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        
        # Disable material processes
        sim.material_processes.apply_metamorphism = lambda: None
        sim.material_processes.apply_phase_transitions = lambda: None
        sim.material_processes.apply_weathering = lambda: None
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Fluid dynamics should respect binding forces."""
        result = super().evaluate(sim)
        
        # Fluid dynamics with binding forces should allow minimal expansion
        expansion = result['metrics']['expansion']
        result['success'] = expansion <= 2
        result['message'] = f"Fluid dynamics only: expansion={expansion} (should be ≤2)"
        
        return result


class MagmaContainmentMaterialOnlyScenario(MagmaContainmentScenario):
    """Magma containment with only material processes enabled."""
    
    def get_name(self) -> str:
        return f"magma_containment_material_only_{self.scenario}"
        
    def get_description(self) -> str:
        return f"Test with ONLY material processes enabled ({self.scenario})"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up scenario with only material processes."""
        super().setup(sim)
        
        # Disable heat transfer
        sim.enable_heat_diffusion = False
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        
        # Disable fluid dynamics
        sim.fluid_dynamics.calculate_planetary_pressure = lambda: None
        sim.fluid_dynamics.apply_unified_kinematics = lambda dt: None
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Material processes alone should not cause movement."""
        result = super().evaluate(sim)
        
        # Material processes alone should cause minimal expansion
        expansion = result['metrics']['expansion']
        result['success'] = expansion <= 1
        result['message'] = f"Material processes only: expansion={expansion} (should be ≤1)"
        
        return result


class MagmaContainmentGravityOnlyScenario(MagmaContainmentScenario):
    """Magma containment with only self-gravity enabled."""
    
    def get_name(self) -> str:
        return f"magma_containment_gravity_only_{self.scenario}"
        
    def get_description(self) -> str:
        return f"Test with ONLY self-gravity enabled ({self.scenario})"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up scenario with only self-gravity."""
        super().setup(sim)
        
        # Disable heat transfer
        sim.enable_heat_diffusion = False
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        
        # Disable fluid dynamics (which processes gravity)
        sim.fluid_dynamics.calculate_planetary_pressure = lambda: None
        sim.fluid_dynamics.apply_unified_kinematics = lambda dt: None
        
        # Disable material processes
        sim.material_processes.apply_metamorphism = lambda: None
        sim.material_processes.apply_phase_transitions = lambda: None
        sim.material_processes.apply_weathering = lambda: None
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Self-gravity alone should not cause movement without fluid dynamics."""
        result = super().evaluate(sim)
        
        # Gravity alone (without fluid dynamics) should cause no expansion
        expansion = result['metrics']['expansion']
        result['success'] = expansion <= 1
        result['message'] = f"Self-gravity only: expansion={expansion} (should be ≤1)"
        
        return result


class MagmaBindingForceScenario(MagmaContainmentScenario):
    """Test scenario to investigate binding force effectiveness."""
    
    def get_name(self) -> str:
        return f"magma_binding_forces_{self.scenario}"
        
    def get_description(self) -> str:
        return f"Investigates binding force matrix values and effectiveness ({self.scenario})"
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate binding forces and their effect."""
        # Get base evaluation
        result = super().evaluate(sim)
        
        # Add binding force information
        binding_matrix = sim.fluid_dynamics._binding_matrix
        mat_index = sim.fluid_dynamics._mat_index
        
        magma_idx = mat_index[MaterialType.MAGMA]
        basalt_idx = mat_index[MaterialType.BASALT]
        granite_idx = mat_index.get(MaterialType.GRANITE, -1)
        
        magma_magma = binding_matrix[magma_idx, magma_idx]
        magma_basalt = binding_matrix[magma_idx, basalt_idx]
        magma_granite = binding_matrix[magma_idx, granite_idx] if granite_idx >= 0 else 0
        
        # Add binding force metrics
        result['metrics'].update({
            'magma_magma_binding': magma_magma,
            'magma_basalt_binding': magma_basalt,
            'magma_granite_binding': magma_granite
        })
        
        # Check binding force validity
        binding_valid = (magma_magma == 0.0 and magma_basalt > 0.0)
        
        # Override success to include binding force check
        expansion = result['metrics']['expansion']
        result['success'] = binding_valid and expansion <= 2
        result['message'] = (f"Binding forces: magma-magma={magma_magma:.2e}, "
                           f"magma-basalt={magma_basalt:.2e}, expansion={expansion}")
        
        return result
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Show binding force metrics."""
        hints = super().get_visualization_hints()
        hints['show_metrics'] = ['expansion', 'magma_magma_binding', 'magma_basalt_binding']
        return hints 