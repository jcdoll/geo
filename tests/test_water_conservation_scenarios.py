"""
Water conservation test scenarios using the test framework.

These scenarios test that water (in all its phases) is conserved during simulation.
"""

import numpy as np
from typing import Dict, Any, Optional

from tests.test_framework import TestScenario
from materials import MaterialType
from geo_game import GeoGame


class WaterConservationScenario(TestScenario):
    """Base scenario for water conservation tests."""
    
    def __init__(self, cavity_count: int = 50, cavity_radius_range: tuple = (1, 3), 
                 sim_steps: int = 400, **kwargs):
        """Initialize water conservation scenario.
        
        Args:
            cavity_count: Number of surface cavities to create
            cavity_radius_range: (min, max) radius for cavities
            sim_steps: Number of simulation steps to run
        """
        super().__init__(cavity_count=cavity_count, 
                        cavity_radius_range=cavity_radius_range,
                        sim_steps=sim_steps, **kwargs)
        self.cavity_count = cavity_count
        self.cavity_radius_range = cavity_radius_range
        self.sim_steps = sim_steps
        self.tolerance_percent = 1.0  # Allow 1% variation
        
    def get_name(self) -> str:
        return f"water_conservation_{self.cavity_count}_cavities"
        
    def get_description(self) -> str:
        return (f"Tests water conservation with {self.cavity_count} surface cavities "
                f"over {self.sim_steps} steps")
        
    def setup(self, sim: GeoGame) -> None:
        """Set up planet with surface cavities to stress test conservation."""
        # Let default planet generation happen
        # (Don't clear to space like other tests)
        
        # Carve random SPACE craters in the surface to stress gas re-entry logic
        rng = np.random.default_rng(123)
        
        # Find surface region (outer part of planet)
        height, width = sim.height, sim.width
        surface_band = 10  # Outer 10 cells
        
        cavities_created = 0
        for _ in range(self.cavity_count * 2):  # Try extra times to ensure we get enough
            # Random position in outer band
            angle = rng.random() * 2 * np.pi
            radius = min(width, height) // 2 - rng.integers(0, surface_band)
            
            cx = width // 2 + int(radius * np.cos(angle))
            cy = height // 2 + int(radius * np.sin(angle))
            
            # Only create cavity if it's near the surface
            if 0 <= cx < width and 0 <= cy < height:
                if sim.material_types[cy, cx] != MaterialType.SPACE:
                    cavity_radius = rng.integers(*self.cavity_radius_range)
                    sim.delete_material_blob(cx, cy, radius=cavity_radius)
                    cavities_created += 1
                    
            if cavities_created >= self.cavity_count:
                break
                
        # Update properties
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def count_water_cells(self, sim: GeoGame) -> int:
        """Count all water-bearing cells (water, ice, vapor)."""
        mask = (
            (sim.material_types == MaterialType.WATER) |
            (sim.material_types == MaterialType.ICE) |
            (sim.material_types == MaterialType.WATER_VAPOR)
        )
        return int(np.sum(mask))
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate water conservation."""
        current_count = self.count_water_cells(sim)
        initial_count = self.initial_state.get('water_count', 0)
        
        if initial_count == 0:
            return {
                'success': False,
                'metrics': {'water_count': current_count},
                'message': 'No initial water count stored!'
            }
            
        # Calculate change
        change = current_count - initial_count
        percent_change = (change / initial_count * 100) if initial_count > 0 else 0
        
        # Check tolerance
        tolerance = int(initial_count * self.tolerance_percent / 100)
        within_tolerance = abs(change) <= tolerance
        
        # Detailed breakdown by phase
        water_count = np.sum(sim.material_types == MaterialType.WATER)
        ice_count = np.sum(sim.material_types == MaterialType.ICE)
        vapor_count = np.sum(sim.material_types == MaterialType.WATER_VAPOR)
        
        return {
            'success': within_tolerance,
            'metrics': {
                'initial_count': initial_count,
                'current_count': current_count,
                'change': change,
                'percent_change': percent_change,
                'water': water_count,
                'ice': ice_count,
                'vapor': vapor_count,
                'tolerance': tolerance,
                'step': sim.time_step if hasattr(sim, 'time_step') else 0
            },
            'message': f"Water: {initial_count} → {current_count} "
                      f"(Δ={change:+d}, {percent_change:+.1f}%) "
                      f"[W:{water_count} I:{ice_count} V:{vapor_count}]"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial state including water count."""
        super().store_initial_state(sim)
        self.initial_state['water_count'] = self.count_water_cells(sim)
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [MaterialType.WATER, MaterialType.ICE, MaterialType.WATER_VAPOR],
            'show_metrics': ['percent_change', 'water', 'ice', 'vapor', 'step']
        }


class WaterConservationStressTestScenario(WaterConservationScenario):
    """Aggressive stress test with many cavities and long runtime."""
    
    def __init__(self, **kwargs):
        # Override with stress test parameters
        super().__init__(
            cavity_count=100,
            cavity_radius_range=(1, 5),
            sim_steps=1000,
            **kwargs
        )
        
    def get_name(self) -> str:
        return "water_conservation_stress_test"
        
    def get_description(self) -> str:
        return "Aggressive water conservation test with 100 cavities over 1000 steps"


class WaterConservationByPhaseScenario(WaterConservationScenario):
    """Test water conservation with specific physics phases disabled."""
    
    def __init__(self, disabled_phase: Optional[str] = None, **kwargs):
        """Initialize with optional phase to disable.
        
        Args:
            disabled_phase: Name of phase to disable, e.g. 'fluid_dynamics', 'weathering'
        """
        super().__init__(**kwargs)
        self.disabled_phase = disabled_phase
        
    def get_name(self) -> str:
        suffix = f"_no_{self.disabled_phase}" if self.disabled_phase else "_all_phases"
        return f"water_conservation{suffix}"
        
    def get_description(self) -> str:
        phase_desc = f" with {self.disabled_phase} disabled" if self.disabled_phase else ""
        return f"Tests water conservation{phase_desc}"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up scenario and optionally disable a physics phase."""
        super().setup(sim)
        
        # Disable specific phase for diagnostic testing
        if self.disabled_phase == 'fluid_dynamics':
            sim.fluid_dynamics.apply_unified_kinematics = lambda dt: None
        elif self.disabled_phase == 'heat_transfer':
            sim.enable_heat_diffusion = False
            sim.enable_internal_heating = False
            sim.enable_solar_heating = False
            sim.enable_radiative_cooling = False
        elif self.disabled_phase == 'material_processes':
            sim.material_processes.apply_metamorphism = lambda: None
            sim.material_processes.apply_phase_transitions = lambda: None
            sim.material_processes.apply_weathering = lambda: None
        elif self.disabled_phase == 'self_gravity':
            if hasattr(sim, 'calculate_self_gravity'):
                sim.calculate_self_gravity = lambda: None 