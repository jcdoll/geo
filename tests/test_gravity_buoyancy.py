"""
Gravity and buoyancy tests with integrated scenarios.

This file tests gravitational attraction and buoyancy effects in fluids.
"""

import numpy as np
import pytest
from typing import Dict, Any, Optional

from tests.test_framework import TestScenario, ScenarioRunner
from materials import MaterialType
from geo_game import GeoGame





# ============================================================================
# GRAVITY SCENARIOS
# ============================================================================

class FluidGravityScenario(TestScenario):
    """Test that fluid falls toward a gravitational body."""
    
    def __init__(self, rock_radius: int = 5, fluid_size: int = 3, 
                 fluid_material: MaterialType = MaterialType.WATER, **kwargs):
        """Initialize fluid gravity scenario."""
        super().__init__(**kwargs)
        self.rock_radius = rock_radius
        self.fluid_size = fluid_size
        self.fluid_material = fluid_material
        self.initial_fluid_distance = None
        
    def get_name(self) -> str:
        return f"fluid_gravity_{self.fluid_material.name.lower()}"
        
    def get_description(self) -> str:
        return f"{self.fluid_material.name} falls toward rock planet due to gravity"
        
    def setup(self, sim: GeoGame) -> None:
        """Create a rock planet with fluid blob in space nearby."""
        # Configure simulation for gravity demonstration
        # Use large cell_depth for stronger gravity (10000x multiplier)
        sim.cell_depth = 1000000.0
        sim.fluid_dynamics.velocity_threshold = False  # No velocity requirement
        sim.enable_pressure = False  # Disable pressure which interferes
        
        # Clear to space
        sim.material_types.fill(MaterialType.SPACE)
        sim.temperature.fill(2.7)
        
        # Create circular rock planet in center
        center_y, center_x = sim.height // 2, sim.width // 2
        
        for dy in range(-self.rock_radius, self.rock_radius + 1):
            for dx in range(-self.rock_radius, self.rock_radius + 1):
                if dy*dy + dx*dx <= self.rock_radius * self.rock_radius:
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < sim.height and 0 <= x < sim.width:
                        sim.material_types[y, x] = MaterialType.BASALT
                        sim.temperature[y, x] = 300.0
        
        # Place fluid blob above and to the right
        fluid_center_y = center_y - self.rock_radius - 5
        fluid_center_x = center_x + self.rock_radius + 5
        
        for dy in range(-self.fluid_size//2, self.fluid_size//2 + 1):
            for dx in range(-self.fluid_size//2, self.fluid_size//2 + 1):
                y, x = fluid_center_y + dy, fluid_center_x + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = self.fluid_material
                    if self.fluid_material == MaterialType.WATER:
                        sim.temperature[y, x] = 293.15
                    elif self.fluid_material == MaterialType.MAGMA:
                        sim.temperature[y, x] = 1500.0 + 273.15
                    else:
                        sim.temperature[y, x] = 300.0
        
        # Calculate initial distance
        self.initial_fluid_distance = np.sqrt(
            (fluid_center_y - center_y)**2 + (fluid_center_x - center_x)**2
        )
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check if fluid has moved toward the planet."""
        # Find fluid cells
        fluid_mask = (sim.material_types == self.fluid_material)
        if not np.any(fluid_mask):
            return {
                'success': False,
                'metrics': {'fluid_count': 0},
                'message': f'No {self.fluid_material.name} found!'
            }
        
        # Calculate center of mass of fluid
        ys, xs = np.where(fluid_mask)
        fluid_center_y = np.mean(ys)
        fluid_center_x = np.mean(xs)
        
        # Find planet center (rock cells)
        rock_mask = (sim.material_types == MaterialType.BASALT)
        if np.any(rock_mask):
            rock_ys, rock_xs = np.where(rock_mask)
            planet_center_y = np.mean(rock_ys)
            planet_center_x = np.mean(rock_xs)
        else:
            planet_center_y = sim.height // 2
            planet_center_x = sim.width // 2
        
        # Calculate current distance
        current_distance = np.sqrt(
            (fluid_center_y - planet_center_y)**2 + 
            (fluid_center_x - planet_center_x)**2
        )
        
        # Check if fluid moved closer
        distance_change = self.initial_fluid_distance - current_distance
        success = distance_change > 2.0  # Should move at least 2 cells closer
        
        return {
            'success': success,
            'metrics': {
                'fluid_count': np.sum(fluid_mask),
                'initial_distance': self.initial_fluid_distance,
                'current_distance': current_distance,
                'distance_change': distance_change
            },
            'message': f"Fluid moved {distance_change:.1f} cells closer to planet"
        }


class RockOnIceScenario(TestScenario):
    """Test that rock falls through melted ice due to gravity."""
    
    def __init__(self, planet_radius: int = 10, ice_thickness: int = 5, **kwargs):
        """Initialize rock on ice scenario."""
        super().__init__(**kwargs)
        self.planet_radius = planet_radius
        self.ice_thickness = ice_thickness
        self.rock_initial_y = None
        
    def get_name(self) -> str:
        return "rock_on_melting_ice"
        
    def get_description(self) -> str:
        return "Rock falls through ice when it melts into water"
        
    def setup(self, sim: GeoGame) -> None:
        """Create rock planet covered in ice with a rock on top."""
        # Configure simulation for gravity demonstration
        sim.cell_depth = 1000000.0  # Strong gravity
        sim.fluid_dynamics.velocity_threshold = False
        sim.enable_pressure = False
        
        # Clear to space
        sim.material_types.fill(MaterialType.SPACE)
        sim.temperature.fill(2.7)
        
        center_y, center_x = sim.height // 2, sim.width // 2
        
        # Create rock core
        core_radius = self.planet_radius - self.ice_thickness
        for dy in range(-self.planet_radius, self.planet_radius + 1):
            for dx in range(-self.planet_radius, self.planet_radius + 1):
                dist_sq = dy*dy + dx*dx
                if dist_sq <= core_radius * core_radius:
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < sim.height and 0 <= x < sim.width:
                        sim.material_types[y, x] = MaterialType.BASALT
                        sim.temperature[y, x] = 250.0  # Cold rock
        
        # Add ice shell
        for dy in range(-self.planet_radius, self.planet_radius + 1):
            for dx in range(-self.planet_radius, self.planet_radius + 1):
                dist_sq = dy*dy + dx*dx
                if core_radius * core_radius < dist_sq <= self.planet_radius * self.planet_radius:
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < sim.height and 0 <= x < sim.width:
                        sim.material_types[y, x] = MaterialType.ICE
                        sim.temperature[y, x] = 250.0  # Below melting point
        
        # Place a rock on top of the ice
        self.rock_initial_y = center_y - self.planet_radius - 2
        rock_x = center_x
        
        sim.material_types[self.rock_initial_y, rock_x] = MaterialType.GRANITE
        sim.temperature[self.rock_initial_y, rock_x] = 250.0
        
        # Heat the ice to start melting
        ice_mask = (sim.material_types == MaterialType.ICE)
        sim.temperature[ice_mask] = 280.0  # Above melting point
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check if the rock has fallen."""
        # Find the granite rock
        granite_mask = (sim.material_types == MaterialType.GRANITE)
        if not np.any(granite_mask):
            return {
                'success': False,
                'metrics': {'rock_found': False},
                'message': 'Granite rock not found!'
            }
        
        # Get rock position
        ys, xs = np.where(granite_mask)
        rock_y = np.mean(ys)
        
        # Calculate how far it fell
        fall_distance = rock_y - self.rock_initial_y
        
        # Count water cells (melted ice)
        water_count = np.sum(sim.material_types == MaterialType.WATER)
        ice_count = np.sum(sim.material_types == MaterialType.ICE)
        
        # Success if rock fell significantly and ice melted
        success = fall_distance > 3.0 and water_count > 0
        
        return {
            'success': success,
            'metrics': {
                'fall_distance': fall_distance,
                'rock_y': rock_y,
                'water_count': water_count,
                'ice_count': ice_count
            },
            'message': f"Rock fell {fall_distance:.1f} cells, {water_count} water cells formed"
        }


# ============================================================================
# BUOYANCY SCENARIOS
# ============================================================================

class BuoyancyScenario(TestScenario):
    """Test scenario for buoyancy effects with different material pairs."""
    
    def __init__(self, fluid_material: MaterialType = MaterialType.WATER,
                 bubble_material: MaterialType = MaterialType.AIR,
                 planet_radius: int = 15,
                 bubble_depth: int = 5,
                 bubble_radius: int = 3):
        super().__init__()
        self.fluid_material = fluid_material
        self.bubble_material = bubble_material
        self.planet_radius = planet_radius
        self.bubble_depth = bubble_depth
        self.bubble_radius = bubble_radius
        self.bubble_initial_y = None
        
    def get_name(self) -> str:
        return f"buoyancy_{self.bubble_material.value}_in_{self.fluid_material.value}"
        
    def get_description(self) -> str:
        return f"{self.bubble_material.value.upper()} bubble rises in {self.fluid_material.value.upper()} due to buoyancy"
        
    def add_circle(self, sim: GeoGame, center_x: int, center_y: int, radius: int, material: MaterialType):
        """Helper to add a circle of material."""
        for y in range(max(0, center_y - radius), min(sim.height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(sim.width, center_x + radius + 1)):
                if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                    sim.material_types[y, x] = material
                    sim.temperature[y, x] = 293.15  # Room temperature
        
    def setup(self, sim: GeoGame) -> None:
        """Create a fluid planet with a bubble inside."""
        # Configure simulation for buoyancy demonstration
        sim.cell_depth = 1000000.0  # Strong gravity for buoyancy
        sim.fluid_dynamics.velocity_threshold = False
        sim.enable_pressure = True  # Keep pressure for buoyancy
        
        # Disable heat transfer to prevent magma cooling
        sim.enable_heat_diffusion = False
        sim.enable_radiative_cooling = False
        sim.enable_solar_heating = False
        sim.enable_internal_heating = False
        
        # HACK: Completely disable heat transfer by monkey-patching
        sim.heat_transfer.solve_heat_diffusion = lambda: (sim.temperature, 1.0)
        
        # HACK: Also disable metamorphism to prevent material transitions
        sim.material_processes.apply_metamorphism = lambda: None
        sim.material_processes.apply_phase_transitions = lambda: None
        
        # Disable surface tension which can remove isolated fluid cells
        sim.enable_surface_tension = False
        
        # Disable weathering to prevent material transitions
        sim.enable_weathering = False
        
        # Clear to space
        sim.material_types.fill(MaterialType.SPACE)
        sim.temperature.fill(2.7)
        
        center_y, center_x = sim.height // 2, sim.width // 2
        
        # Create fluid planet using helper
        self.add_circle(sim, center_x, center_y, self.planet_radius, self.fluid_material)
        
        # Set appropriate temperature for fluid
        fluid_mask = (sim.material_types == self.fluid_material)
        if self.fluid_material == MaterialType.WATER:
            sim.temperature[fluid_mask] = 293.15
        elif self.fluid_material == MaterialType.MAGMA:
            sim.temperature[fluid_mask] = 1500.0 + 273.15
        
        # Place bubble at specified depth below surface
        bubble_y = center_y - self.planet_radius + self.bubble_depth
        bubble_x = center_x
        self.bubble_initial_y = bubble_y
        
        # Create bubble using helper
        self.add_circle(sim, bubble_x, bubble_y, self.bubble_radius, self.bubble_material)
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check if the bubble has risen."""
        # Find bubble cells
        bubble_mask = (sim.material_types == self.bubble_material)
        if not np.any(bubble_mask):
            # Debug: check what materials are present
            unique_materials = np.unique(sim.material_types)
            material_names = [mat.name for mat in unique_materials]
            return {
                'success': False,
                'metrics': {'bubble_count': 0, 'materials_present': material_names},
                'message': f'No {self.bubble_material.name} bubble found! Materials: {material_names}'
            }
        
        # Get bubble position
        ys, xs = np.where(bubble_mask)
        bubble_y = np.mean(ys)
        bubble_count = len(ys)
        
        # Calculate rise distance (negative because rising means decreasing y)
        rise_distance = self.bubble_initial_y - bubble_y
        
        # Check if bubble is near surface (find fluid surface)
        fluid_mask = (sim.material_types == self.fluid_material)
        if np.any(fluid_mask):
            fluid_ys, _ = np.where(fluid_mask)
            surface_y = np.min(fluid_ys)  # Top of fluid
            distance_to_surface = bubble_y - surface_y
        else:
            distance_to_surface = float('inf')
        
        # Success if bubble rose significantly
        success = rise_distance > 3.0
        
        return {
            'success': success,
            'metrics': {
                'bubble_count': bubble_count,
                'rise_distance': rise_distance,
                'bubble_y': bubble_y,
                'distance_to_surface': distance_to_surface
            },
            'message': f"{self.bubble_material.name} rose {rise_distance:.1f} cells in {self.fluid_material.name}"
        }


# ============================================================================
# PYTEST TESTS
# ============================================================================

# NOTE: Buoyancy tests are currently failing due to several issues:
# 1. Gravity is realistically weak for small planets (need ~10000x multiplier)
# 2. Pressure solver creates large negative pressures that dominate forces
# 3. Force-based swapping requires velocity differences even with dv_thresh=0
# 4. Surface tension code removes isolated fluid cells
# 5. The physics is too realistic for simple demonstrations
#
# To fix buoyancy:
# - Add gravity scaling factor to GeoGame for stronger gravity
# - Add option to disable pressure calculation or use simpler model
# - Allow pure force-based swapping without velocity requirement for fluids
# - Make surface tension optional or less aggressive
# - Consider adding a simplified "demo mode" for basic physics tests

def test_water_falls_to_planet():
    """Test that water falls toward a rock planet."""
    scenario = FluidGravityScenario(rock_radius=5, fluid_size=3, fluid_material=MaterialType.WATER)
    runner = ScenarioRunner(scenario, sim_width=30, sim_height=30)
    result = runner.run_headless(max_steps=50)
    assert result['success'], f"Water didn't fall toward planet: {result['message']}"


def test_magma_falls_to_planet():
    """Test that magma falls toward a rock planet."""
    scenario = FluidGravityScenario(rock_radius=5, fluid_size=3, fluid_material=MaterialType.MAGMA)
    runner = ScenarioRunner(scenario, sim_width=30, sim_height=30)
    result = runner.run_headless(max_steps=50)
    assert result['success'], f"Magma didn't fall toward planet: {result['message']}"


def test_rock_falls_through_melted_ice():
    """Test that rock falls when ice melts to water."""
    scenario = RockOnIceScenario(planet_radius=10, ice_thickness=5)
    runner = ScenarioRunner(scenario, sim_width=40, sim_height=40)
    result = runner.run_headless(max_steps=100)
    assert result['success'], f"Rock didn't fall through melted ice: {result['message']}"


def test_air_rises_in_water():
    """Test buoyancy: air bubble rises in water."""
    scenario = BuoyancyScenario(
        fluid_material=MaterialType.WATER,
        bubble_material=MaterialType.AIR,
        planet_radius=20,  # Larger planet
        bubble_depth=8      # Deeper initial position
    )
    runner = ScenarioRunner(scenario, sim_width=50, sim_height=50)
    result = runner.run_headless(max_steps=50)  # Less steps needed
    assert result['success'], f"Air didn't rise in water: {result['message']}"

def test_space_rises_in_water():
    """Test buoyancy: space (vacuum) bubble rises in water."""
    scenario = BuoyancyScenario(
        fluid_material=MaterialType.WATER,
        bubble_material=MaterialType.SPACE,
        planet_radius=15,
        bubble_depth=5
    )
    runner = ScenarioRunner(scenario, sim_width=40, sim_height=40)
    result = runner.run_headless(max_steps=80)
    assert result['success'], f"Space bubble didn't rise in water: {result['message']}"

def test_air_rises_in_magma():
    """Test buoyancy: air bubble rises in magma."""
    scenario = BuoyancyScenario(
        fluid_material=MaterialType.MAGMA,
        bubble_material=MaterialType.AIR,
        planet_radius=15,
        bubble_depth=5
    )
    runner = ScenarioRunner(scenario, sim_width=40, sim_height=40)
    result = runner.run_headless(max_steps=80)
    assert result['success'], f"Air didn't rise in magma: {result['message']}"


@pytest.mark.parametrize("fluid,bubble", [
    (MaterialType.WATER, MaterialType.AIR),
    (MaterialType.WATER, MaterialType.SPACE),
    (MaterialType.MAGMA, MaterialType.AIR),
    (MaterialType.MAGMA, MaterialType.SPACE),
])
def test_buoyancy_combinations(fluid, bubble):
    """Test various fluid/bubble combinations for buoyancy."""
    scenario = BuoyancyScenario(
        fluid_material=fluid,
        bubble_material=bubble,
        planet_radius=12,
        bubble_depth=4
    )
    runner = ScenarioRunner(scenario, sim_width=35, sim_height=35)
    result = runner.run_headless(max_steps=60)
    
    assert result['success'], f"{bubble.name} didn't rise in {fluid.name}: {result['message']}"





# ============================================================================
# SCENARIO REGISTRY FOR VISUAL RUNNER
# ============================================================================

SCENARIOS = {
    'gravity_water': lambda: FluidGravityScenario(fluid_material=MaterialType.WATER),
    'gravity_magma': lambda: FluidGravityScenario(fluid_material=MaterialType.MAGMA),
    'rock_on_ice': lambda: RockOnIceScenario(),
    'buoyancy_air_water': lambda: BuoyancyScenario(MaterialType.WATER, MaterialType.AIR),
    'buoyancy_space_water': lambda: BuoyancyScenario(MaterialType.WATER, MaterialType.SPACE),
    'buoyancy_air_magma': lambda: BuoyancyScenario(MaterialType.MAGMA, MaterialType.AIR),
} 