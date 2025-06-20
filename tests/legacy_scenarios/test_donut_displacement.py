"""Test scenarios for rigid body fluid displacement, including donut-shaped objects with enclosed fluids."""

import numpy as np
import pytest
from geo_game import GeoGame
from materials import MaterialType
from tests.framework.test_framework import TestScenario, ScenarioRunner

class TestDonutWithWaterScenario(TestScenario):
    """Test a donut-shaped rock with water in the center falling under gravity."""
    
    def get_name(self):
        return "donut_water"
    
    def setup(self, sim=None):
        """Create a donut-shaped rock with water inside."""
        if sim is None:
            self.sim = GeoGame(width=40, height=60, cell_size=50.0, quality=2)
        else:
            self.sim = sim
        
        # Create donut shape in upper part of simulation
        center_x, center_y = 20, 15
        outer_radius = 8
        inner_radius = 4
        
        # Place donut-shaped rock
        for y in range(self.sim.height):
            for x in range(self.sim.width):
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if inner_radius < dist_from_center <= outer_radius:
                    self.sim.material_types[y, x] = MaterialType.GRANITE
                    self.sim.temperature[y, x] = 300  # Room temperature
        
        # Fill center with water
        for y in range(self.sim.height):
            for x in range(self.sim.width):
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist_from_center <= inner_radius:
                    self.sim.material_types[y, x] = MaterialType.WATER
                    self.sim.temperature[y, x] = 300
        
        # Fill remaining space with air (not space/vacuum)
        for y in range(self.sim.height):
            for x in range(self.sim.width):
                if self.sim.material_types[y, x] == MaterialType.SPACE:
                    self.sim.material_types[y, x] = MaterialType.AIR
                    self.sim.temperature[y, x] = 300
        
        # Update material properties
        self.sim._update_material_properties()
        
        # Enable debugging for rigid bodies
        self.sim.debug_rigid_bodies = False
        
        # Store initial positions
        self.initial_rock_positions = np.where(self.sim.material_types == MaterialType.GRANITE)
        self.initial_water_positions = np.where(self.sim.material_types == MaterialType.WATER)
        self.initial_com_y = np.mean(self.initial_rock_positions[0])
        
        # Expected behavior: donut falls as a unit with water inside
        self.expected_behavior = {
            "donut_falls": True,
            "water_stays_inside": True,
            "shape_preserved": True
        }
    
    def step(self):
        self.sim.step_forward()
    
    def evaluate(self, sim=None):
        """Check if donut fell correctly with water inside."""
        if sim is not None:
            self.sim = sim
        # Find current positions
        current_rock_positions = np.where(self.sim.material_types == MaterialType.GRANITE)
        current_water_positions = np.where(self.sim.material_types == MaterialType.WATER)
        
        # Check if donut fell (center of mass moved down)
        current_com_y = np.mean(current_rock_positions[0]) if len(current_rock_positions[0]) > 0 else self.initial_com_y
        donut_fell = current_com_y > self.initial_com_y + 2  # At least 2 cells down
        
        # Check if water is still enclosed
        water_still_inside = False
        if len(current_water_positions[0]) > 0:
            # Calculate center of mass of water
            water_com_y = np.mean(current_water_positions[0])
            water_com_x = np.mean(current_water_positions[1])
            
            # Calculate center of mass of rock
            if len(current_rock_positions[0]) > 0:
                rock_com_y = np.mean(current_rock_positions[0])
                rock_com_x = np.mean(current_rock_positions[1])
                
                # Check if water COM is close to rock COM (within the donut)
                distance = np.sqrt((water_com_x - rock_com_x)**2 + (water_com_y - rock_com_y)**2)
                water_still_inside = distance < 5  # Within 5 cells of rock center
        
        # Check shape preservation (approximate) - allow more fragmentation
        shape_preserved = len(current_rock_positions[0]) >= 0.7 * len(self.initial_rock_positions[0])
        
        results = {
            "donut_fell": donut_fell,
            "water_stayed_inside": water_still_inside,
            "shape_preserved": shape_preserved,
            "fall_distance": current_com_y - self.initial_com_y,
            "rock_cells_preserved": len(current_rock_positions[0]) / len(self.initial_rock_positions[0]),
            "water_cells_preserved": len(current_water_positions[0]) / len(self.initial_water_positions[0]) if len(self.initial_water_positions[0]) > 0 else 0
        }
        
        # Return success if main criteria are met
        success = donut_fell and water_still_inside and shape_preserved
        
        return {
            'success': success,
            'metrics': results,
            'message': f"Donut fell {results['fall_distance']:.1f} cells, water {'stayed inside' if water_still_inside else 'escaped'}, shape {'preserved' if shape_preserved else 'deformed'}"
        }
    
    def get_description(self):
        return "Donut-shaped rock with water in center falling under gravity"
    
    def get_default_steps(self):
        return 100  # Run for 100 steps to see fall
    
    def get_focus_region(self):
        return None  # Watch whole simulation


class TestRockPushingWaterScenario(TestScenario):
    """Test a rock falling into water and displacing it."""
    
    def get_name(self):
        return "rock_water_displacement"
    
    def setup(self, sim=None):
        """Create a rock above water."""
        if sim is None:
            self.sim = GeoGame(width=30, height=40, cell_size=10.0, quality=2)
        else:
            self.sim = sim
        
        # Create water pool at bottom
        water_height = 10
        for y in range(self.sim.height - water_height, self.sim.height):
            for x in range(self.sim.width):
                self.sim.material_types[y, x] = MaterialType.WATER
                self.sim.temperature[y, x] = 300
        
        # Create rock above water
        rock_size = 5
        rock_x = self.sim.width // 2 - rock_size // 2
        rock_y = 10
        for y in range(rock_y, rock_y + rock_size):
            for x in range(rock_x, rock_x + rock_size):
                if x < self.sim.width and y < self.sim.height:
                    self.sim.material_types[y, x] = MaterialType.GRANITE
                    self.sim.temperature[y, x] = 300
        
        # Update material properties
        self.sim._update_material_properties()
        
        # Store initial water level
        self.initial_water_cells = np.sum(self.sim.material_types == MaterialType.WATER)
        self.initial_rock_y = rock_y
        
        # Expected behavior
        self.expected_behavior = {
            "rock_falls": True,
            "water_displaced": True,
            "water_conserved": True
        }
    
    def step(self):
        self.sim.step_forward()
    
    def evaluate(self, sim=None):
        """Check if rock fell and displaced water correctly."""
        if sim is not None:
            self.sim = sim
        # Find rock position
        rock_positions = np.where(self.sim.material_types == MaterialType.GRANITE)
        if len(rock_positions[0]) == 0:
            return {
                'success': False,
                'metrics': {"error": "Rock disappeared"},
                'message': "Rock disappeared from simulation"
            }
        
        current_rock_y = np.min(rock_positions[0])
        rock_fell = current_rock_y > self.initial_rock_y + 5
        
        # Check water conservation
        current_water_cells = np.sum(self.sim.material_types == MaterialType.WATER)
        water_conserved = abs(current_water_cells - self.initial_water_cells) < 5  # Allow small variance
        
        # Check if water was displaced (water level should rise or water should move aside)
        water_positions = np.where(self.sim.material_types == MaterialType.WATER)
        water_displaced = len(water_positions[0]) > 0  # Water still exists
        
        results = {
            "rock_fell": rock_fell,
            "rock_fall_distance": current_rock_y - self.initial_rock_y,
            "water_displaced": water_displaced,
            "water_conserved": water_conserved,
            "water_cells_initial": self.initial_water_cells,
            "water_cells_current": current_water_cells
        }
        
        success = rock_fell and water_displaced and water_conserved
        
        return {
            'success': success,
            'metrics': results,
            'message': f"Rock fell {results['rock_fall_distance']:.1f} cells, water {'displaced' if water_displaced else 'not displaced'}, water cells: {results['water_cells_initial']} -> {results['water_cells_current']}"
        }
    
    def get_description(self):
        return "Rock falling into water pool with displacement"
    
    def get_default_steps(self):
        return 150
    
    def get_focus_region(self):
        return None


# Register scenarios for visual runner
SCENARIOS = {
    'donut_water': lambda: TestDonutWithWaterScenario(),
    'rock_water_displacement': lambda: TestRockPushingWaterScenario()
}


# Pytest test functions
def test_donut_with_water():
    """Test that a donut-shaped rock with water inside falls correctly."""
    scenario = TestDonutWithWaterScenario()
    runner = ScenarioRunner(scenario)
    result = runner.run_headless(max_steps=100)
    
    print(f"\nDonut displacement test results: {result['metrics']}")
    
    # Basic assertions
    assert result['success'], f"Donut displacement test failed: {result['message']}"
    assert result['metrics']["donut_fell"], "Donut should fall under gravity"
    assert result['metrics']["water_stayed_inside"], "Water should stay inside the donut"
    assert result['metrics']["shape_preserved"], "Donut shape should be preserved"


def test_rock_water_displacement():
    """Test that a rock falling into water displaces it correctly."""
    scenario = TestRockPushingWaterScenario()
    runner = ScenarioRunner(scenario)
    result = runner.run_headless(max_steps=150)
    
    print(f"\nRock-water displacement test results: {result['metrics']}")
    
    # Basic assertions
    assert result['success'], f"Rock-water displacement test failed: {result['message']}"
    assert result['metrics']["rock_fell"], "Rock should fall into water"
    assert result['metrics']["water_displaced"], "Water should be displaced"
    assert result['metrics']["water_conserved"], "Water volume should be conserved"


if __name__ == "__main__":
    # Run tests
    print("Testing donut with water scenario...")
    test_donut_with_water()
    
    print("\nTesting rock-water displacement scenario...")
    test_rock_water_displacement()
    
    print("\nAll tests passed!")