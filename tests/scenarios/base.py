"""Enhanced base class for test scenarios with timeout and progress tracking."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import time
import numpy as np
from geo_game import GeoGame
from materials import MaterialType


class TestScenario(ABC):
    """Enhanced base class for test scenarios.
    
    Provides common functionality for scenarios that can run in both
    headless (pytest) and visual modes.
    """
    
    def __init__(self, **kwargs):
        """Initialize scenario with optional parameters."""
        self.params = kwargs
        self.sim = None
        self.initial_state = {}
        self.start_time = None
        self.timeout = kwargs.get('timeout', None)  # Optional timeout in seconds
        
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
        
        IMPORTANT: Scenarios should create their own deterministic setup,
        not rely on default planet generation.
        
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
            - 'preferred_display_mode': Optional[str] - 'material', 'temperature', etc.
        """
        return {}
        
    def check_timeout(self) -> bool:
        """Check if scenario has exceeded its timeout.
        
        Returns:
            True if timeout exceeded, False otherwise
        """
        if self.timeout is None or self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.timeout
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial state for comparison (called after setup)."""
        self.sim = sim
        self.start_time = time.time()
        
        # Store material counts
        material_counts = {}
        for mat in MaterialType:
            count = np.sum(sim.material_types == mat)
            if count > 0:
                material_counts[mat] = count
                
        # Store material positions for tracking movement
        material_positions = {}
        for mat in MaterialType:
            if mat != MaterialType.SPACE:
                positions = self._get_material_positions(sim, mat)
                if positions:
                    material_positions[mat] = positions
                    
        # Store energy metrics
        total_thermal_energy = np.sum(sim.temperature[sim.material_types != MaterialType.SPACE])
        
        self.initial_state = {
            'material_counts': material_counts,
            'material_positions': material_positions,
            'total_thermal_energy': total_thermal_energy,
            'time': time.time(),
        }
        
    def _count_materials(self, sim: GeoGame) -> Dict[MaterialType, int]:
        """Count cells of each material type."""
        counts = {}
        for mat in MaterialType:
            count = np.sum(sim.material_types == mat)
            if count > 0:
                counts[mat] = count
        return counts
        
    def _get_material_positions(self, sim: GeoGame, material: MaterialType) -> set:
        """Get positions of specific material."""
        positions = np.argwhere(sim.material_types == material)
        return {(int(y), int(x)) for y, x in positions}
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time since scenario started."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
        
    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics dictionary as a readable string."""
        parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)


class ScenarioGroup:
    """Groups related scenarios for organization."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.scenarios = {}
        
    def add_scenario(self, key: str, scenario_class: type, **default_params):
        """Add a scenario to this group."""
        self.scenarios[key] = (scenario_class, default_params)
        
    def get_scenario(self, key: str, **override_params) -> TestScenario:
        """Get an instance of a scenario with optional parameter overrides."""
        if key not in self.scenarios:
            raise KeyError(f"Unknown scenario: {key}")
            
        scenario_class, default_params = self.scenarios[key]
        params = {**default_params, **override_params}
        return scenario_class(**params)
        
    def list_scenarios(self) -> List[str]:
        """List all scenario keys in this group."""
        return list(self.scenarios.keys())