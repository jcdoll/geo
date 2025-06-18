"""
Water blob condensation test scenario using the test framework.

This demonstrates how simple tests can be converted to use the framework
for visualization support.
"""

import numpy as np
from typing import Dict, Any

from tests.test_framework import TestScenario
from materials import MaterialType
from geo_game import GeoGame


class WaterBlobCondensationScenario(TestScenario):
    """Test scenario for water condensing from a bar shape to a circular blob."""
    
    def __init__(self, bar_width: int = 30, bar_height: int = 4, **kwargs):
        """Initialize water blob test.
        
        Args:
            bar_width: Width of the initial water bar
            bar_height: Height of the initial water bar
        """
        super().__init__(bar_width=bar_width, bar_height=bar_height, **kwargs)
        self.bar_width = bar_width
        self.bar_height = bar_height
        
    def get_name(self) -> str:
        return "water_blob_condensation"
        
    def get_description(self) -> str:
        return f"Water bar ({self.bar_width}x{self.bar_height}) should condense into a circular blob"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up a horizontal bar of water in space."""
        # Clear to space
        sim.material_types.fill(MaterialType.SPACE)
        sim.temperature.fill(2.7)  # Space temperature
        
        # Create water bar in center
        center_y = sim.height // 2
        center_x = sim.width // 2
        
        y_start = center_y - self.bar_height // 2
        y_end = center_y + self.bar_height // 2
        x_start = center_x - self.bar_width // 2
        x_end = center_x + self.bar_width // 2
        
        # Fill with water
        sim.material_types[y_start:y_end, x_start:x_end] = MaterialType.WATER
        sim.temperature[y_start:y_end, x_start:x_end] = 293.15  # Room temperature
        
        # Update properties
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate how circular the water blob has become."""
        # Find water cells
        water_mask = (sim.material_types == MaterialType.WATER)
        water_count = np.sum(water_mask)
        
        if water_count == 0:
            return {
                'success': False,
                'metrics': {'water_count': 0, 'aspect_ratio': float('inf')},
                'message': 'No water found!'
            }
            
        # Calculate bounding box
        ys, xs = np.where(water_mask)
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Calculate aspect ratio (always >= 1)
        aspect_ratio = max(width / height, height / width)
        
        # Calculate initial aspect ratio for comparison
        initial_aspect_ratio = max(self.bar_width / self.bar_height, 
                                 self.bar_height / self.bar_width)
        
        # Calculate circularity (how close to a circle)
        # Perfect circle has aspect ratio 1.0
        circularity = 1.0 / aspect_ratio
        
        # Success if aspect ratio is close to 1 (circular)
        success = aspect_ratio < 1.6
        
        return {
            'success': success,
            'metrics': {
                'water_count': water_count,
                'aspect_ratio': aspect_ratio,
                'initial_aspect_ratio': initial_aspect_ratio,
                'circularity': circularity,
                'width': width,
                'height': height,
                'center': f"({(min_y + max_y) / 2:.1f}, {(min_x + max_x) / 2:.1f})"
            },
            'message': f"Water blob aspect ratio: {aspect_ratio:.2f} "
                      f"({'circular' if success else 'still elongated'})"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [MaterialType.WATER],
            'show_metrics': ['aspect_ratio', 'circularity', 'water_count', 'center']
        } 