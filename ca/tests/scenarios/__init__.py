"""Test scenarios that can run in both headless and visual modes.

This package contains all test scenarios organized by category:
- fluids.py: Water conservation, magma flow, fluid gravity
- rigid_body.py: Rigid body with enclosed fluid, rigid body fluid displacement, rigid body rotation
- mechanics.py: Gravity fall, buoyancy, hydrostatic pressure
- materials.py: Material stability, metamorphism, phase transitions

Each scenario can be:
1. Run via pytest using test_scenarios.py
2. Run visually using run_visual_tests.py
3. Imported and used directly
"""

# Import base classes
from .base import TestScenario, ScenarioGroup

# Import all scenarios for easy access
from .fluids import (
    WaterConservationScenario,
    WaterConservationStressScenario,
    WaterBlobScenario,
    WaterLineCollapseScenario,
    MagmaFlowScenario,
    FluidGravityScenario,
)

from .mechanics import (
    GravityFallScenario,
    BuoyancyScenario,
    HydrostaticPressureScenario,
)

from .materials import (
    MaterialStabilityScenario,
    MetamorphismScenario,
    PhaseTransitionScenario,
)

from .rigid_body import (
    RigidBodyWithEnclosedFluidScenario,
    RigidBodyFluidDisplacementScenario,
    RigidBodyRotationScenario,
)

# Define scenario groups for organization
SCENARIO_GROUPS = {
    'fluids': ScenarioGroup('Fluid Dynamics', 'Tests for fluid behavior'),
    'mechanics': ScenarioGroup('Mechanical Physics', 'Tests for gravity, buoyancy, pressure'),
    'materials': ScenarioGroup('Material Physics', 'Tests for phase transitions and stability'),
    'rigid_body': ScenarioGroup('Rigid Body Physics', 'Tests for rigid body dynamics'),
}

# Register fluid scenarios
SCENARIO_GROUPS['fluids'].add_scenario('water_conservation', WaterConservationScenario)
SCENARIO_GROUPS['fluids'].add_scenario('water_conservation_stress', WaterConservationStressScenario)
SCENARIO_GROUPS['fluids'].add_scenario('water_blob', WaterBlobScenario, blob_width=20, blob_height=10)
SCENARIO_GROUPS['fluids'].add_scenario('water_line_collapse', WaterLineCollapseScenario)
SCENARIO_GROUPS['fluids'].add_scenario('magma_flow', MagmaFlowScenario, volcano_size=10)
SCENARIO_GROUPS['fluids'].add_scenario('fluid_gravity', FluidGravityScenario)

# Register mechanics scenarios
SCENARIO_GROUPS['mechanics'].add_scenario('gravity_fall', GravityFallScenario)
SCENARIO_GROUPS['mechanics'].add_scenario('buoyancy_ice', BuoyancyScenario)
SCENARIO_GROUPS['mechanics'].add_scenario('hydrostatic', HydrostaticPressureScenario)

# Register material scenarios
SCENARIO_GROUPS['materials'].add_scenario('material_stability', MaterialStabilityScenario)
SCENARIO_GROUPS['materials'].add_scenario('metamorphism', MetamorphismScenario)
SCENARIO_GROUPS['materials'].add_scenario('phase_transition', PhaseTransitionScenario)

# Register rigid body scenarios
SCENARIO_GROUPS['rigid_body'].add_scenario('rigid_container', RigidBodyWithEnclosedFluidScenario)
SCENARIO_GROUPS['rigid_body'].add_scenario('rigid_displacement', RigidBodyFluidDisplacementScenario)
SCENARIO_GROUPS['rigid_body'].add_scenario('rigid_rotation', RigidBodyRotationScenario)

# Flat dictionary for backward compatibility
ALL_SCENARIOS = {}
for group in SCENARIO_GROUPS.values():
    ALL_SCENARIOS.update(group.scenarios)