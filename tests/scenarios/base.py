"""
Base class for flux-based test scenarios.

Provides common functionality for scenarios that can run in both
headless (pytest) and visual modes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import time
import numpy as np

from simulation import FluxSimulation
from materials import MaterialType


class FluxTestScenario(ABC):
    """Base class for flux-based test scenarios."""
    
    def __init__(self, **kwargs):
        """Initialize scenario with optional parameters."""
        self.params = kwargs
        self.sim: Optional[FluxSimulation] = None
        self.initial_state = {}
        self.start_time = None
        self.timeout = kwargs.get('timeout', None)
        
    @abstractmethod
    def get_name(self) -> str:
        """Return a unique name for this scenario."""
        pass
        
    @abstractmethod
    def get_description(self) -> str:
        """Return a human-readable description of what this scenario tests."""
        pass
        
    @abstractmethod
    def setup(self, sim: FluxSimulation) -> None:
        """
        Set up the initial conditions for this scenario.
        
        Args:
            sim: The simulation instance to configure
        """
        pass
        
    @abstractmethod
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """
        Evaluate the current state and return metrics.
        
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
        """
        Provide hints for visualization.
        
        Returns:
            Dictionary with visualization hints:
            - 'focus_region': Optional[Tuple[y_min, y_max, x_min, x_max]]
            - 'highlight_materials': Optional[List[MaterialType]]
            - 'show_metrics': List[str] of metrics to display
            - 'preferred_display_mode': Optional[str]
        """
        return {}
        
    def check_timeout(self) -> bool:
        """Check if scenario has exceeded its timeout."""
        if self.timeout is None or self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.timeout
        
    def store_initial_state(self, sim: FluxSimulation) -> None:
        """Store initial state for comparison."""
        self.sim = sim
        self.start_time = time.time()
        
        # Store material inventory
        self.initial_state['material_inventory'] = sim.state.get_material_inventory()
        self.initial_state['total_mass'] = sim.state.get_total_mass()
        self.initial_state['total_energy'] = sim.state.get_total_energy()
        
        # Store material centers of mass for tracking movement
        self.initial_state['material_com'] = {}
        y_grid, x_grid = np.mgrid[:sim.state.ny, :sim.state.nx]
        
        for mat in MaterialType:
            if mat == MaterialType.SPACE:
                continue
                
            phi = sim.state.vol_frac[mat]
            total_volume = np.sum(phi)
            
            if total_volume > 0:
                com_x = np.sum(phi * x_grid) / total_volume
                com_y = np.sum(phi * y_grid) / total_volume
                self.initial_state['material_com'][mat] = (com_x, com_y)
                
    def get_material_movement(self, sim: FluxSimulation) -> Dict[MaterialType, float]:
        """Calculate how far each material has moved from initial position."""
        movements = {}
        y_grid, x_grid = np.mgrid[:sim.state.ny, :sim.state.nx]
        
        for mat in MaterialType:
            if mat == MaterialType.SPACE:
                continue
                
            phi = sim.state.vol_frac[mat]
            total_volume = np.sum(phi)
            
            if total_volume > 0 and mat in self.initial_state['material_com']:
                com_x = np.sum(phi * x_grid) / total_volume
                com_y = np.sum(phi * y_grid) / total_volume
                
                initial_com = self.initial_state['material_com'][mat]
                distance = np.sqrt(
                    (com_x - initial_com[0])**2 + 
                    (com_y - initial_com[1])**2
                )
                movements[mat] = distance * sim.state.dx  # Convert to meters
                
        return movements