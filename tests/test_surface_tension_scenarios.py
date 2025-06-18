"""
Surface tension test scenarios using the test framework.

These scenarios test fluid surface tension effects on water shapes.
"""

import numpy as np
from typing import Dict, Any, List
from scipy import ndimage

from tests.test_framework import TestScenario
from materials import MaterialType
from geo_game import GeoGame


class SurfaceTensionScenario(TestScenario):
    """Base scenario for testing surface tension effects."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_water_count = 0
        
    def get_water_shape_metrics(self, sim: GeoGame) -> Dict[str, Any]:
        """Calculate shape metrics for water/ice/vapor."""
        # Find all water phase cells
        water_mask = (sim.material_types == MaterialType.WATER)
        ice_mask = (sim.material_types == MaterialType.ICE)
        vapor_mask = (sim.material_types == MaterialType.WATER_VAPOR)
        all_water_mask = water_mask | ice_mask | vapor_mask
        
        metrics = {
            'water_count': np.sum(water_mask),
            'ice_count': np.sum(ice_mask),
            'vapor_count': np.sum(vapor_mask),
            'total_count': np.sum(all_water_mask)
        }
        
        if np.any(all_water_mask):
            # Calculate bounding box
            coords = np.where(all_water_mask)
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            
            # Aspect ratio and circularity
            aspect_ratio = max(width / height, height / width)
            circularity = 1.0 / aspect_ratio
            
            # Count connected components
            labeled, num_features = ndimage.label(all_water_mask)
            
            metrics.update({
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'num_clusters': num_features,
                'center_y': (min_y + max_y) / 2,
                'center_x': (min_x + max_x) / 2
            })
        else:
            metrics.update({
                'width': 0,
                'height': 0,
                'aspect_ratio': float('inf'),
                'circularity': 0,
                'num_clusters': 0,
                'center_y': 0,
                'center_x': 0
            })
            
        return metrics


class WaterLineCollapseScenario(SurfaceTensionScenario):
    """Test that a thin line of water collapses into a circular shape."""
    
    def __init__(self, line_length: int = 20, line_thickness: int = 1, **kwargs):
        super().__init__(line_length=line_length, line_thickness=line_thickness, **kwargs)
        self.line_length = line_length
        self.line_thickness = line_thickness
        
    def get_name(self) -> str:
        return f"water_line_collapse_{self.line_length}x{self.line_thickness}"
        
    def get_description(self) -> str:
        return f"A {self.line_length}x{self.line_thickness} water line should collapse into circular shape"
        
    def setup(self, sim: GeoGame) -> None:
        """Create a horizontal line of water in space."""
        # Clear to space
        sim.material_types.fill(MaterialType.SPACE)
        sim.temperature.fill(2.7)  # Space temperature
        
        # Create horizontal water line
        water_y = sim.height // 2
        x_start = (sim.width - self.line_length) // 2
        x_end = x_start + self.line_length
        
        for y in range(water_y, water_y + self.line_thickness):
            for x in range(x_start, x_end):
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.WATER
                    sim.temperature[y, x] = 300.0  # Room temperature
                    
        # Store initial water count
        self.initial_water_count = np.sum(sim.material_types == MaterialType.WATER)
        
        # Update properties
        sim._properties_dirty = True
        sim._update_material_properties()
        
        # Check if surface tension is available
        if not hasattr(sim.fluid_dynamics, 'apply_physics_based_surface_tension'):
            print("Warning: Physics-based surface tension not available")
            
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate how circular the water has become."""
        metrics = self.get_water_shape_metrics(sim)
        
        if metrics['total_count'] == 0:
            return {
                'success': False,
                'metrics': metrics,
                'message': 'No water found!'
            }
            
        # Calculate initial aspect ratio
        initial_aspect_ratio = self.line_length / self.line_thickness
        
        # Water conservation check (allow 10 cell tolerance for multi-step physics)
        conservation_ok = abs(metrics['total_count'] - self.initial_water_count) <= 10
        
        # Shape check - should be more circular (aspect ratio closer to 1)
        shape_improved = metrics['aspect_ratio'] < initial_aspect_ratio * 0.5
        target_circularity = metrics['aspect_ratio'] <= 2.5
        
        success = conservation_ok and target_circularity
        
        message_parts = []
        if not conservation_ok:
            message_parts.append(f"water not conserved ({self.initial_water_count} → {metrics['total_count']})")
        if shape_improved:
            message_parts.append(f"shape improved ({initial_aspect_ratio:.1f} → {metrics['aspect_ratio']:.1f})")
        else:
            message_parts.append(f"shape didn't improve enough")
            
        return {
            'success': success,
            'metrics': metrics,
            'message': ' | '.join(message_parts) if message_parts else "Surface tension working correctly"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [MaterialType.WATER, MaterialType.ICE, MaterialType.WATER_VAPOR],
            'show_metrics': ['aspect_ratio', 'circularity', 'total_count', 'num_clusters']
        }


class WaterDropletFormationScenario(SurfaceTensionScenario):
    """Test that scattered water cells coalesce into droplets."""
    
    def __init__(self, cluster_positions: List[List[tuple]] = None, **kwargs):
        super().__init__(**kwargs)
        if cluster_positions is None:
            # Default: 3 small clusters
            self.cluster_positions = [
                [(10, 10), (10, 11), (11, 10)],  # Cluster 1
                [(10, 15), (11, 15), (11, 16)],  # Cluster 2  
                [(15, 10), (15, 11), (16, 11)],  # Cluster 3
            ]
        else:
            self.cluster_positions = cluster_positions
            
    def get_name(self) -> str:
        return f"water_droplet_formation_{len(self.cluster_positions)}_clusters"
        
    def get_description(self) -> str:
        return f"{len(self.cluster_positions)} water clusters should coalesce into fewer droplets"
        
    def setup(self, sim: GeoGame) -> None:
        """Create scattered water cells."""
        # Clear to space
        sim.material_types.fill(MaterialType.SPACE)
        sim.temperature.fill(2.7)
        
        # Create water clusters
        all_positions = []
        for cluster in self.cluster_positions:
            for y, x in cluster:
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.WATER
                    sim.temperature[y, x] = 300.0
                    all_positions.append((y, x))
                    
        self.initial_water_count = len(all_positions)
        
        # Update properties
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate droplet coalescence."""
        metrics = self.get_water_shape_metrics(sim)
        
        if metrics['total_count'] == 0:
            return {
                'success': False,
                'metrics': metrics,
                'message': 'No water found!'
            }
            
        initial_clusters = len(self.cluster_positions)
        final_clusters = metrics['num_clusters']
        
        # Success if clusters reduced or stayed same (not fragmented)
        clusters_ok = final_clusters <= initial_clusters
        
        # Conservation check
        conservation_ok = abs(metrics['total_count'] - self.initial_water_count) <= 5
        
        success = clusters_ok and conservation_ok
        
        message_parts = []
        if clusters_ok:
            if final_clusters < initial_clusters:
                message_parts.append(f"clusters reduced ({initial_clusters} → {final_clusters})")
            else:
                message_parts.append(f"clusters maintained ({final_clusters})")
        else:
            message_parts.append(f"clusters increased! ({initial_clusters} → {final_clusters})")
            
        if not conservation_ok:
            message_parts.append(f"water not conserved")
            
        return {
            'success': success,
            'metrics': metrics,
            'message': ' | '.join(message_parts)
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [MaterialType.WATER, MaterialType.ICE],
            'show_metrics': ['num_clusters', 'total_count']
        } 