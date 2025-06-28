"""Test scenarios for flux-based simulation.

This package contains all test scenarios organized by category:
- fluids.py: Hydrostatic equilibrium, water drops, buoyancy
- materials.py: Phase transitions (freezing, melting, evaporation)
- thermal.py: Heat diffusion, uranium heating, solar/radiative effects

Each scenario can be:
1. Run via pytest using test framework
2. Run visually using run_visual_tests.py
3. Imported and used directly
"""

# Import base classes
from .base import FluxTestScenario

# Import all scenarios for easy access
from .fluids import (
    HydrostaticEquilibriumScenario,
    WaterDropFallScenario,
    BuoyancyScenario,
    RockSinkingScenario,
)

from .materials import (
    WaterFreezingScenario,
    IceMeltingScenario,
    WaterEvaporationScenario,
    RockMeltingScenario,
)

from .thermal import (
    HeatDiffusionScenario,
    UraniumHeatingScenario,
    SolarHeatingScenario,
    RadiativeCoolingScenario,
)

# Import ScenarioGroup to organize scenarios
from .groups import ScenarioGroup

# Define scenario groups for organization
SCENARIO_GROUPS = {
    'fluids': ScenarioGroup('Fluid Dynamics', 'Tests for fluid behavior and gravity'),
    'materials': ScenarioGroup('Material Transitions', 'Tests for phase changes'),
    'thermal': ScenarioGroup('Thermal Physics', 'Tests for heat transfer and radiation'),
}

# Register fluid scenarios
SCENARIO_GROUPS['fluids'].add_scenario('hydrostatic_equilibrium', HydrostaticEquilibriumScenario)
SCENARIO_GROUPS['fluids'].add_scenario('water_drop_fall', WaterDropFallScenario)
SCENARIO_GROUPS['fluids'].add_scenario('buoyancy', BuoyancyScenario)
SCENARIO_GROUPS['fluids'].add_scenario('rock_sinking', RockSinkingScenario)

# Register material scenarios
SCENARIO_GROUPS['materials'].add_scenario('water_freezing', WaterFreezingScenario)
SCENARIO_GROUPS['materials'].add_scenario('ice_melting', IceMeltingScenario)
SCENARIO_GROUPS['materials'].add_scenario('water_evaporation', WaterEvaporationScenario)
SCENARIO_GROUPS['materials'].add_scenario('rock_melting', RockMeltingScenario)

# Register thermal scenarios
SCENARIO_GROUPS['thermal'].add_scenario('heat_diffusion', HeatDiffusionScenario)
SCENARIO_GROUPS['thermal'].add_scenario('uranium_heating', UraniumHeatingScenario)
SCENARIO_GROUPS['thermal'].add_scenario('solar_heating', SolarHeatingScenario)
SCENARIO_GROUPS['thermal'].add_scenario('radiative_cooling', RadiativeCoolingScenario)

# Flat dictionary for easy access
ALL_SCENARIOS = {}
for group in SCENARIO_GROUPS.values():
    ALL_SCENARIOS.update(group.scenarios)