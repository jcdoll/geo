# Test Suite

This directory contains a comprehensive test suite for the geological physics simulation, organized by test type and purpose.

## Directory Structure

```
tests/
├── diagnostics/         # Debugging and analysis tools
├── framework/          # Test infrastructure
├── integration/        # Full system integration tests
├── legacy_scenarios/   # Original scenario files for backward compatibility
├── scenarios/          # Organized visual test scenarios
└── unit/              # Pure unit tests
```

### Unit Tests (`unit/`)
Pure functionality tests for individual components:
- **Material Properties**: `test_materials.py`, `test_material_cache_cleanup.py`
- **Physics Solvers**: `test_pressure_solver.py`, `test_poisson_solver.py`, `test_self_gravity.py`
- **Fluid Dynamics**: `test_buoyancy_physics.py`, `test_surface_tension.py`, `test_velocity_projection.py`
- **Rigid Body Mechanics**: `test_rigid_body_fall.py`, `test_rigid_body_fluid_container.py`, `test_simple_container_movement.py`
- **Motion Physics**: `test_motion_physics.py`, `test_unified_kinematics.py`, `test_buoyancy_mechanics.py`
- **Visualization**: `test_visualizer_functionality.py`

### Integration Tests (`integration/`)
Tests for complete system behavior:
- `test_simulation_lifecycle.py` - Full simulation initialization, stepping, reset
- `test_integration.py` - Complete workflow tests
- `test_simulation_engine.py` - Engine integration
- `test_reset_simulation.py` - Reset functionality
- `test_space_integrity.py` - Conservation laws

### Visual Scenarios (`scenarios/`)
Interactive test scenarios that can be run with visualization:
- **`test_buoyancy.py`** - Ice floating, density comparison, rigid body buoyancy
- **`test_fluids.py`** - Water conservation, magma flow, surface tension
- **`test_rigid_body.py`** - Falling rocks, container dynamics, donut displacement
- **`test_materials.py`** - Magma containment, phase transitions, metamorphic gradients

### Diagnostic Tools (`diagnostics/`)
Specialized debugging utilities:
- `test_acceleration_analysis.py` - Detailed motion and force tracking
- `test_pressure_diagnosis.py` - Pressure field analysis and debugging

### Framework (`framework/`)
Test infrastructure:
- `test_framework.py` - Base classes for scenario-based testing
- `test_visualizer.py` - Visualization extension for test scenarios
- `run_visual_tests.py` - Command-line runner for visual scenarios

### Legacy Scenarios (`legacy_scenarios/`)
Original test files preserved for backward compatibility:
- `test_magma.py`, `test_water.py`, `test_gravity_buoyancy.py`, `test_donut_displacement.py`

## Running Tests

### Run All Tests
```bash
# Run complete test suite
python tests/run_all_tests.py all

# Run specific test categories
python tests/run_all_tests.py unit
python tests/run_all_tests.py integration
python tests/run_all_tests.py scenarios --list
```

### Run Unit Tests with Pytest
```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_pressure_solver.py

# Run with verbose output
pytest -v tests/unit/
```

### Run Visual Scenarios
```bash
# From the project root directory:

# List all available scenarios
python tests/framework/run_visual_tests.py --list

# Run specific scenario
python tests/framework/run_visual_tests.py ice_floating

# Run with custom parameters
python tests/framework/run_visual_tests.py water_conservation --steps 500 --size 80

# Alternative: use python -m from anywhere
python -m tests.framework.run_visual_tests ice_floating
```

## Available Visual Scenarios

### Buoyancy Scenarios
- `ice_floating` - Ice vs granite falling into water
- `rock_donut_container` - Rock donut with magma preserving temperature
- `density_comparison` - Multiple materials with different densities
- `rigid_body_buoyancy` - Complete rigid body ice/granite test

### Fluid Scenarios
- `water_conservation` - Water droplet volume conservation
- `magma_flow` - Hot magma flowing and cooling
- `surface_tension` - Water droplet coalescence

### Rigid Body Scenarios
- `falling_rock` - Granite blocks falling as coherent units
- `container_fall` - Container with water falling together
- `donut_displacement` - Donut-shaped object displacing fluid

### Material Scenarios
- `magma_containment` - Hot magma with metamorphic rock walls
- `phase_transitions` - Ice melting, water freezing, magma solidifying
- `metamorphic_gradient` - Rock metamorphism from heat source

## Creating New Tests

### Unit Test Template
```python
# tests/unit/test_my_component.py
import numpy as np
import pytest

def test_my_feature():
    """Test specific functionality"""
    # Setup
    data = create_test_data()
    
    # Execute
    result = my_function(data)
    
    # Assert
    assert result == expected_value
```

### Visual Scenario Template
```python
# tests/scenarios/test_my_category.py
from tests.framework.test_framework import TestScenario
from materials import MaterialType

class MyScenario(TestScenario):
    def get_name(self) -> str:
        return "my_scenario"
    
    def get_description(self) -> str:
        return "Tests my feature visually"
    
    def setup(self, sim):
        """Set up initial conditions"""
        sim.material_types[10, 10] = MaterialType.WATER
        sim.temperature[10, 10] = 300.0
    
    def evaluate(self, sim, step):
        """Evaluate success criteria"""
        success = check_conditions(sim)
        if step % 10 == 0:
            print(f"Step {step}: Status update")
        return success

SCENARIOS = {
    'my_scenario': lambda: MyScenario(),
}
```

## Visual Test Controls

When running visual tests:
- **SPACE** - Play/Pause simulation
- **LEFT/RIGHT** - Step backward/forward
- **R** - Reset simulation
- **H** - Show help/controls
- **S** - Save screenshot
- **ESC** - Exit

## Test Organization Benefits

1. **Clear Separation**: Unit tests vs integration tests vs visual scenarios
2. **No Redundancy**: 9 buoyancy test files consolidated into 2 focused files
3. **Easy Discovery**: Tests organized by type and purpose
4. **Diagnostic Tools**: Specialized debugging utilities preserved
5. **Backward Compatible**: Legacy scenarios still available

## Physics Test Coverage

- **Buoyancy**: Comprehensive tests for ice floating, density stratification
- **Rigid Bodies**: Falling blocks, container dynamics, fluid displacement
- **Fluids**: Conservation, flow, surface tension
- **Materials**: Phase transitions, metamorphism, thermal properties
- **Gravity**: Self-gravity fields, external gravity, pressure calculations
- **Thermal**: Heat diffusion, radiation, material transitions

## Known Issues

- Buoyancy magnitude needs tuning for perfect floating equilibrium
- Some legacy gravity tests are failing due to realistic weak gravity
- Force-based swapping requires velocity differences that can prevent simple buoyancy

## Best Practices

1. **Unit Tests**: Test individual functions and calculations
2. **Integration Tests**: Test complete workflows and system behavior
3. **Visual Scenarios**: Test complex emergent behavior and physics interactions
4. **Diagnostics**: Use for debugging specific physics issues
5. **Clear Names**: Use descriptive test and scenario names
6. **Documentation**: Comment complex test setups and assertions