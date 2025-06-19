"""
Water physics tests with integrated scenarios.

This file combines all water-related test scenarios and pytest tests.
"""

import numpy as np
import pytest
from typing import Dict, Any, Optional, List
from scipy import ndimage

from tests.test_framework import TestScenario, ScenarioRunner
from materials import MaterialType
from geo_game import GeoGame


# ============================================================================
# WATER CONSERVATION SCENARIOS
# ============================================================================

class WaterConservationScenario(TestScenario):
    """Base scenario for water conservation tests."""
    
    def __init__(self, cavity_count: int = 50, cavity_radius_range: tuple = (1, 3), **kwargs):
        """Initialize water conservation scenario."""
        super().__init__(**kwargs)
        self.cavity_count = cavity_count
        self.cavity_radius_range = cavity_radius_range
        self.tolerance_percent = 1.0  # Allow 1% variation
        
    def get_name(self) -> str:
        return f"water_conservation_{self.cavity_count}_cavities"
        
    def get_description(self) -> str:
        return f"Tests water conservation with {self.cavity_count} surface cavities"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up planet with surface cavities to stress test conservation."""
        # Let default planet generation happen
        # Carve random SPACE craters in the surface
        rng = np.random.default_rng(123)
        
        height, width = sim.height, sim.width
        surface_band = 10  # Outer 10 cells
        
        cavities_created = 0
        for _ in range(self.cavity_count * 2):
            angle = rng.random() * 2 * np.pi
            radius = min(width, height) // 2 - rng.integers(0, surface_band)
            
            cx = width // 2 + int(radius * np.cos(angle))
            cy = height // 2 + int(radius * np.sin(angle))
            
            if 0 <= cx < width and 0 <= cy < height:
                if sim.material_types[cy, cx] != MaterialType.SPACE:
                    cavity_radius = rng.integers(*self.cavity_radius_range)
                    sim.delete_material_blob(cx, cy, radius=cavity_radius)
                    cavities_created += 1
                    
            if cavities_created >= self.cavity_count:
                break
                
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
            
        change = current_count - initial_count
        percent_change = (change / initial_count * 100) if initial_count > 0 else 0
        
        tolerance = int(initial_count * self.tolerance_percent / 100)
        within_tolerance = abs(change) <= tolerance
        
        return {
            'success': within_tolerance,
            'metrics': {
                'initial_count': initial_count,
                'current_count': current_count,
                'change': change,
                'percent_change': percent_change,
            },
            'message': f"Water: {initial_count} → {current_count} (Δ={change:+d}, {percent_change:+.1f}%)"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial state including water count."""
        super().store_initial_state(sim)
        self.initial_state['water_count'] = self.count_water_cells(sim)


class WaterConservationStressScenario(WaterConservationScenario):
    """Aggressive stress test with many cavities."""
    
    def __init__(self, **kwargs):
        super().__init__(cavity_count=100, cavity_radius_range=(1, 5), **kwargs)
        
    def get_name(self) -> str:
        return "water_conservation_stress_test"


# ============================================================================
# WATER BLOB SCENARIOS
# ============================================================================

class WaterBlobScenario(TestScenario):
    """Test scenario for water condensing from a bar shape to a circular blob."""
    
    def __init__(self, bar_width: int = 30, bar_height: int = 4, **kwargs):
        """Initialize water blob test."""
        super().__init__(**kwargs)
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
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate how circular the water blob has become."""
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
        
        # Success if aspect ratio is close to 1 (circular)
        success = aspect_ratio < 1.6
        
        return {
            'success': success,
            'metrics': {
                'water_count': water_count,
                'aspect_ratio': aspect_ratio,
                'width': width,
                'height': height,
            },
            'message': f"Water blob aspect ratio: {aspect_ratio:.2f} ({'circular' if success else 'still elongated'})"
        }


# ============================================================================
# SURFACE TENSION SCENARIOS
# ============================================================================

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
            })
        else:
            metrics.update({
                'width': 0,
                'height': 0,
                'aspect_ratio': float('inf'),
                'circularity': 0,
                'num_clusters': 0,
            })
            
        return metrics


class WaterLineCollapseScenario(SurfaceTensionScenario):
    """Test that a thin line of water collapses into a circular shape."""
    
    def __init__(self, line_length: int = 20, line_thickness: int = 1, **kwargs):
        super().__init__(**kwargs)
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
        sim.temperature.fill(2.7)
        
        # Create horizontal water line
        water_y = sim.height // 2
        x_start = (sim.width - self.line_length) // 2
        x_end = x_start + self.line_length
        
        for y in range(water_y, water_y + self.line_thickness):
            for x in range(x_start, x_end):
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.WATER
                    sim.temperature[y, x] = 300.0
                    
        self.initial_water_count = np.sum(sim.material_types == MaterialType.WATER)
        sim._properties_dirty = True
        sim._update_material_properties()
        
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
        
        # Water conservation check
        conservation_ok = abs(metrics['total_count'] - self.initial_water_count) <= 10
        
        # Shape check - should be more circular
        shape_improved = metrics['aspect_ratio'] < initial_aspect_ratio * 0.5
        target_circularity = metrics['aspect_ratio'] <= 2.5
        
        success = conservation_ok and target_circularity
        
        return {
            'success': success,
            'metrics': metrics,
            'message': f"Aspect ratio: {metrics['aspect_ratio']:.2f} ({'circular' if success else 'elongated'})"
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
        for cluster in self.cluster_positions:
            for y, x in cluster:
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.WATER
                    sim.temperature[y, x] = 300.0
                    
        self.initial_water_count = len([1 for cluster in self.cluster_positions for _ in cluster])
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
        
        # Success if clusters reduced or stayed same
        clusters_ok = final_clusters <= initial_clusters
        conservation_ok = abs(metrics['total_count'] - self.initial_water_count) <= 5
        
        success = clusters_ok and conservation_ok
        
        return {
            'success': success,
            'metrics': metrics,
            'message': f"Clusters: {initial_clusters} → {final_clusters}"
        }


# ============================================================================
# WATER LEAKAGE DIAGNOSTIC SCENARIO
# ============================================================================

class WaterLeakageDiagnosticScenario(TestScenario):
    """Diagnostic scenario to identify which physics phases cause water leakage."""
    
    def __init__(self, disabled_phase: str = None):
        """Initialize with optional phase to disable."""
        super().__init__()
        self.disabled_phase = disabled_phase
        self.initial_water_count = 0
        
    def get_name(self) -> str:
        phase_name = self.disabled_phase.replace('.', '_') if self.disabled_phase else "none"
        return f"water_leakage_diagnostic_{phase_name}"
        
    def get_description(self) -> str:
        phase_name = self.disabled_phase or "None (all active)"
        return f"Diagnose water leakage with phase disabled: {phase_name}"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up planet with stress-inducing holes."""
        np.random.seed(0)
        
        # Punch SPACE craters near the surface to stress leakage paths
        for y in range(45, 50):
            for x in range(0, 50, 5):
                sim.delete_material_blob(x, y, radius=1)
                
        # Count initial water
        water_mask = (
            (sim.material_types == MaterialType.WATER) |
            (sim.material_types == MaterialType.ICE) |
            (sim.material_types == MaterialType.WATER_VAPOR)
        )
        self.initial_water_count = int(np.sum(water_mask))
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Count final water and report percentage change."""
        water_mask = (
            (sim.material_types == MaterialType.WATER) |
            (sim.material_types == MaterialType.ICE) |
            (sim.material_types == MaterialType.WATER_VAPOR)
        )
        final_water_count = int(np.sum(water_mask))
        
        if self.initial_water_count == 0:
            pct_change = 0.0
        else:
            pct_change = (final_water_count - self.initial_water_count) / self.initial_water_count * 100.0
            
        phase_name = self.disabled_phase or "None (all active)"
        message = f"Phase disabled: {phase_name:<45} ΔWater = {pct_change:+6.2f}%"
        
        # This is a diagnostic test, always "passes"
        return {
            'success': True,
            'metrics': {
                'initial_water': self.initial_water_count,
                'final_water': final_water_count,
                'percent_change': pct_change,
            },
            'message': message
        }


# ============================================================================
# PYTEST TESTS
# ============================================================================

def test_water_conservation():
    """Test that water is conserved during simulation."""
    scenario = WaterConservationScenario()
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    result = runner.run_headless(max_steps=400)
    assert result['success'], f"Water conservation failed: {result['message']}"


def test_water_conservation_stress():
    """Aggressive water conservation test with many cavities."""
    scenario = WaterConservationStressScenario()
    runner = ScenarioRunner(scenario, sim_width=80, sim_height=80)
    result = runner.run_headless(max_steps=100)
    
    percent_change = result['metrics']['percent_change']
    assert abs(percent_change) < 5.0, f"Water loss too high in stress test: {percent_change:.2f}%"


def test_water_blob_condensation():
    """Test that a water bar condenses into a circular blob."""
    scenario = WaterBlobScenario()
    runner = ScenarioRunner(scenario, sim_width=40, sim_height=40)
    result = runner.run_headless(max_steps=120)
    
    assert result['success'], f"Water blob test failed: {result['message']}"
    aspect_ratio = result['metrics']['aspect_ratio']
    assert aspect_ratio < 1.6, f"Water blob still elongated (ratio {aspect_ratio:.2f})"


@pytest.mark.parametrize("bar_dimensions", [(20, 2), (40, 6), (10, 10)])
def test_water_blob_different_sizes(bar_dimensions):
    """Test water condensation with different initial bar sizes."""
    width, height = bar_dimensions
    scenario = WaterBlobScenario(bar_width=width, bar_height=height)
    runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
    
    result = runner.run_headless(max_steps=150)
    
    # Square should already be circular, others should converge
    if width == height:
        assert result['metrics']['aspect_ratio'] < 1.2, f"Square didn't stay circular"
    else:
        assert result['success'], f"Test failed for {width}x{height}: {result['message']}"


def test_water_line_collapse():
    """Test that a thin water line collapses into a circular shape."""
    scenario = WaterLineCollapseScenario(line_length=20, line_thickness=1)
    runner = ScenarioRunner(scenario, sim_width=40, sim_height=40)
    
    result = runner.run_headless(max_steps=100)
    
    assert result['success'], f"Water line collapse failed: {result['message']}"
    
    aspect_ratio = result['metrics']['aspect_ratio']
    assert aspect_ratio < 2.5, f"Water still too elongated (aspect ratio: {aspect_ratio:.2f})"


def test_water_droplet_formation():
    """Test that scattered water cells coalesce into droplets."""
    scenario = WaterDropletFormationScenario()
    runner = ScenarioRunner(scenario, sim_width=30, sim_height=30)
    
    result = runner.run_headless(max_steps=80)
    
    assert result['success'], f"Droplet formation failed: {result['message']}"
    
    num_clusters = result['metrics']['num_clusters']
    assert num_clusters <= 3, f"Water fragmented into {num_clusters} clusters"


@pytest.mark.parametrize("line_thickness", [1, 2, 3])
def test_surface_tension_line_thickness(line_thickness):
    """Test surface tension on lines of different thicknesses."""
    scenario = WaterLineCollapseScenario(line_length=15, line_thickness=line_thickness)
    runner = ScenarioRunner(scenario, sim_width=30, sim_height=30)
    
    result = runner.run_headless(max_steps=100)
    
    # Check water conservation
    initial_water = scenario.initial_water_count
    final_water = result['metrics']['total_count']
    
    assert abs(final_water - initial_water) <= 10, \
        f"Water not conserved: {initial_water} → {final_water}"


# Diagnostic tests
PHASE_ATTRS = [
    "fluid_dynamics",
    "material_processes",
]

@pytest.mark.parametrize("disabled_phase", [None] + PHASE_ATTRS)
def test_water_leakage_diagnostic(disabled_phase, capsys):
    """Diagnostic test to identify which physics phases cause water leakage."""
    scenario = WaterLeakageDiagnosticScenario(disabled_phase)
    runner = ScenarioRunner(scenario, sim_width=50, sim_height=50)
    
    if disabled_phase:
        runner.module_disabler.add_disabled_phase(disabled_phase)
    
    result = runner.run_headless(max_steps=100)
    
    with capsys.disabled():
        print(f"\n{result['message']}")
    
    assert result['success']  # Diagnostic test always passes


# ============================================================================
# SCENARIO REGISTRY FOR VISUAL RUNNER
# ============================================================================

SCENARIOS = {
    'water_conservation': lambda: WaterConservationScenario(),
    'water_stress': lambda: WaterConservationStressScenario(),
    'water_blob': lambda: WaterBlobScenario(),
    'water_blob_thin': lambda: WaterBlobScenario(bar_width=40, bar_height=2),
    'water_line': lambda: WaterLineCollapseScenario(),
    'water_droplet': lambda: WaterDropletFormationScenario(),
    'water_diagnostic': lambda: WaterLeakageDiagnosticScenario(),
} 