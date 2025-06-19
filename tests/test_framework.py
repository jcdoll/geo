"""
Test framework for creating reusable, visualizable test scenarios.

This module provides base classes and utilities for creating test scenarios
that can be run both headless and with visualization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from geo_game import GeoGame
from materials import MaterialType


class TestScenario(ABC):
    """Abstract base class for test scenarios.
    
    Subclasses should implement methods to set up the scenario,
    provide metadata, and evaluate success criteria.
    """
    
    def __init__(self, **kwargs):
        """Initialize scenario with optional parameters."""
        self.params = kwargs
        self.sim = None
        self.initial_state = {}
        
    @abstractmethod
    def get_name(self) -> str:
        """Return a unique name for this scenario."""
        pass
        
    @abstractmethod
    def get_description(self) -> str:
        """Return a human-readable description of what this scenario tests."""
        pass
        
    @abstractmethod
    def setup(self, sim: GeoGame) -> None:
        """Set up the initial conditions for this scenario.
        
        Args:
            sim: The simulation instance to configure
        """
        pass
        
    @abstractmethod
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate the current state and return metrics.
        
        Args:
            sim: The simulation instance to evaluate
            
        Returns:
            Dictionary with evaluation metrics, must include:
            - 'success': bool indicating if test criteria are met
            - 'metrics': dict of numerical metrics
            - 'message': str describing current state
        """
        pass
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide hints for visualization (optional override).
        
        Returns:
            Dictionary with visualization hints:
            - 'focus_region': Optional[Tuple[y_min, y_max, x_min, x_max]]
            - 'highlight_materials': Optional[List[MaterialType]]
            - 'show_metrics': List[str] of metrics to display
        """
        return {}
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial state for comparison (called after setup)."""
        self.sim = sim
        # Store material positions for each material type
        material_positions = {}
        for mat in MaterialType:
            if mat != MaterialType.SPACE:
                positions = self._get_material_positions(sim, mat)
                if positions:
                    material_positions[mat] = positions
                    
        self.initial_state = {
            'material_counts': self._count_materials(sim),
            'material_positions': material_positions,
        }
        
    def _count_materials(self, sim: GeoGame) -> Dict[MaterialType, int]:
        """Count cells of each material type."""
        counts = {}
        for mat in MaterialType:
            count = np.sum(sim.material_types == mat)
            if count > 0:
                counts[mat] = count
        return counts
        
    def _get_material_positions(self, sim: GeoGame, material: Optional[MaterialType] = None) -> set:
        """Get positions of specific material or all non-space materials."""
        if material is not None:
            mask = sim.material_types == material
        else:
            mask = sim.material_types != MaterialType.SPACE
        return set(zip(*np.where(mask)))


class ScenarioRunner:
    """Runs test scenarios and collects results."""
    
    def __init__(self, scenario: TestScenario, sim_width: int = 80, sim_height: int = 80):
        """Initialize runner with a scenario."""
        self.scenario = scenario
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.sim = None
        self.step_count = 0
        self.evaluation_history = []
        
    def setup(self) -> GeoGame:
        """Create simulation and set up the scenario."""
        # Create simulation
        self.sim = GeoGame(
            self.sim_width, 
            self.sim_height, 
            cell_size=100.0,
            quality=1, 
            log_level="WARNING"
        )
        
        # Let scenario configure it
        self.scenario.setup(self.sim)
        
        # Store initial state
        self.scenario.store_initial_state(self.sim)
        
        # Reset step counter
        self.step_count = 0
        self.evaluation_history = []
        
        return self.sim
        
    def step(self) -> Dict[str, Any]:
        """Run one simulation step and evaluate."""
        if self.sim is None:
            raise RuntimeError("Must call setup() before step()")
            
        # Step simulation
        self.sim.step_forward()
        self.step_count += 1
        
        # Evaluate current state
        evaluation = self.scenario.evaluate(self.sim)
        evaluation['step'] = self.step_count
        
        # Store in history
        self.evaluation_history.append(evaluation)
        
        return evaluation
        
    def run_headless(self, max_steps: int, early_stop: bool = True) -> Dict[str, Any]:
        """Run scenario without visualization.
        
        Args:
            max_steps: Maximum number of steps to run
            early_stop: Stop early if success criteria are met
            
        Returns:
            Final evaluation results
        """
        print(f"\nRunning scenario: {self.scenario.get_name()}")
        print(f"Description: {self.scenario.get_description()}")
        print(f"Parameters: {self.scenario.params}")
        print("-" * 60)
        
        # Setup
        self.setup()
        
        # Initial evaluation
        initial_eval = self.scenario.evaluate(self.sim)
        print(f"Initial state: {initial_eval['message']}")
        
        # Run simulation
        for step in range(max_steps):
            eval_result = self.step()
            
            # Print progress every 10 steps or on important events
            if step % 10 == 9 or eval_result.get('success', False):
                print(f"Step {step + 1}: {eval_result['message']}")
                
            # Early stop if success criteria met
            if early_stop and eval_result.get('success', False):
                print(f"\nSuccess criteria met at step {step + 1}!")
                break
                
        # Final evaluation
        final_eval = self.evaluation_history[-1] if self.evaluation_history else initial_eval
        
        print("\nFinal results:")
        print(f"  Steps run: {self.step_count}")
        print(f"  Success: {final_eval.get('success', False)}")
        print(f"  {final_eval['message']}")
        
        if 'metrics' in final_eval:
            print("\nMetrics:")
            for key, value in final_eval['metrics'].items():
                print(f"  {key}: {value}")
                
        return final_eval


class ModuleDisabler:
    """Context manager for temporarily disabling simulation modules."""
    
    def __init__(self, sim: GeoGame, disable_modules: Optional[List[str]] = None):
        """Initialize with simulation and modules to disable."""
        self.sim = sim
        self.disable_modules = disable_modules or []
        self.original_states = {}
        
    def __enter__(self):
        """Disable specified modules."""
        for module in self.disable_modules:
            if module == 'heat_transfer':
                self._disable_heat_transfer()
            elif module == 'fluid_dynamics':
                self._disable_fluid_dynamics()
            elif module == 'material_processes':
                self._disable_material_processes()
            elif module == 'self_gravity':
                self._disable_self_gravity()
                
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original module states."""
        for attr, value in self.original_states.items():
            if '.' in attr:  # Handle nested attributes
                obj_path, attr_name = attr.rsplit('.', 1)
                obj = self.sim
                for part in obj_path.split('.'):
                    obj = getattr(obj, part)
                setattr(obj, attr_name, value)
            else:
                setattr(self.sim, attr, value)
                
    def _disable_heat_transfer(self):
        """Disable heat transfer modules."""
        heat_attrs = [
            'enable_heat_diffusion',
            'enable_internal_heating', 
            'enable_solar_heating',
            'enable_radiative_cooling'
        ]
        for attr in heat_attrs:
            if hasattr(self.sim, attr):
                self.original_states[attr] = getattr(self.sim, attr)
                setattr(self.sim, attr, False)
                
    def _disable_fluid_dynamics(self):
        """Disable fluid dynamics methods."""
        self.original_states['fluid_dynamics.calculate_planetary_pressure'] = \
            self.sim.fluid_dynamics.calculate_planetary_pressure
        self.original_states['fluid_dynamics.apply_unified_kinematics'] = \
            self.sim.fluid_dynamics.apply_unified_kinematics
            
        self.sim.fluid_dynamics.calculate_planetary_pressure = lambda: None
        self.sim.fluid_dynamics.apply_unified_kinematics = lambda dt: None
        
    def _disable_material_processes(self):
        """Disable material process methods."""
        methods = ['apply_metamorphism', 'apply_phase_transitions', 'apply_weathering']
        for method in methods:
            full_attr = f'material_processes.{method}'
            self.original_states[full_attr] = getattr(self.sim.material_processes, method)
            setattr(self.sim.material_processes, method, lambda: None)
            
    def _disable_self_gravity(self):
        """Disable self gravity calculation."""
        if hasattr(self.sim, 'calculate_self_gravity'):
            self.original_states['calculate_self_gravity'] = self.sim.calculate_self_gravity
            self.sim.calculate_self_gravity = lambda: None 