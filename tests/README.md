# Test Suite

This directory contains physics simulation tests organized by category. Each test file contains both scenario definitions and pytest tests, allowing tests to be run headlessly (for CI/CD) or with real-time visualization for debugging.

## Structure

### Scenario-Based Tests
These files contain both test scenarios and pytest tests in a single file:

- **`test_magma.py`** - Magma containment and stability tests
- **`test_water.py`** - Water conservation, blob formation, and surface tension tests  
- **`test_gravity_buoyancy.py`** - Gravitational attraction and buoyancy tests

### Core Physics Tests
Traditional unit tests for core simulation components:

- **`test_poisson_solver.py`** - Gravity field solver
- **`test_unified_kinematics.py`** - Fluid dynamics and material swapping
- **`test_motion_physics.py`** - Force field and velocity calculations
- **`test_materials.py`** - Material property calculations
- **`test_self_gravity.py`** - Self-gravity calculations
- **`test_velocity_projection.py`** - Velocity field projections

### Integration Tests
- **`test_integration.py`** - Complete simulation workflows
- **`test_simulation_engine.py`** - Basic simulation operations
- **`test_reset_simulation.py`** - Simulation state reset
- **`test_space_integrity.py`** - Space cell conservation
- **`test_material_cache_cleanup.py`** - Material cache management

### Framework & Tools
- **`test_framework.py`** - Base classes for scenario-based testing
- **`test_visualizer.py`** - Test visualizer extension
- **`test_visualizer_functionality.py`** - Visualizer unit tests
- **`run_visual_tests.py`** - Command-line tool for visual test execution

## Creating a New Test

### For Scenario-Based Tests

Add your scenario and test to one of the existing category files (`test_magma.py`, `test_water.py`, `test_chunk.py`) or create a new category file:

```python
"""
My category tests with integrated scenarios.
"""

import numpy as np
import pytest
from typing import Dict, Any

from tests.test_framework import TestScenario, ScenarioRunner
from materials import MaterialType
from geo_game import GeoGame


# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

class MyTestScenario(TestScenario):
    """Description of what this scenario tests."""
    
    def get_name(self) -> str:
        return "my_test_scenario"
    
    def get_description(self) -> str:
        return "Tests that my feature works correctly"
    
    def setup(self, sim: GeoGame) -> None:
        """Set up initial conditions."""
        sim.material_types[10, 10] = MaterialType.WATER
        sim.temperature[10, 10] = 300.0
        sim._update_material_properties()
    
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate current state."""
        success = check_my_conditions(sim)
        
        return {
            'success': success,
            'metrics': {'my_metric': calculate_metric(sim)},
            'message': f"Test {'passed' if success else 'failed'}"
        }


# ============================================================================
# PYTEST TESTS
# ============================================================================

def test_my_feature():
    """Test my feature using the scenario."""
    scenario = MyTestScenario()
    runner = ScenarioRunner(scenario)
    result = runner.run_headless(max_steps=100)
    assert result['success'], f"Test failed: {result['message']}"


# ============================================================================
# SCENARIO REGISTRY FOR VISUAL RUNNER
# ============================================================================

SCENARIOS = {
    'my_test': lambda: MyTestScenario(),
}
```

Then import your scenarios in `run_visual_tests.py`:

```python
from tests.test_my_category import SCENARIOS as MY_SCENARIOS
SCENARIOS.update(MY_SCENARIOS)
```

## Running Tests

### Headless (Pytest)

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_my_feature.py

# Run with verbose output
pytest -v tests/test_my_feature.py
```

### Visual Mode

```bash
# List all available scenarios
python tests/run_visual_tests.py --list

# Run a specific scenario
python tests/run_visual_tests.py my_test

# Run with custom parameters
python tests/run_visual_tests.py my_test --steps 500 --size 80 --scale 8

# Run scenario variant
python tests/run_visual_tests.py magma_containment --variant large

# Run with physics disabled
python tests/run_visual_tests.py water_phase --disabled-phase fluid_dynamics
```

### Visual Controls

When running visually:
- **SPACE**: Play/Pause simulation
- **R**: Step forward one frame
- **S**: Save screenshot
- **ESC**: Exit

## Available Scenarios

Run `python tests/run_visual_tests.py --list` to see all available scenarios.

### Magma Tests (`test_magma.py`)
- `magma_small`: Small magma containment test
- `magma_large`: Large magma containment test  
- `magma_no_physics`: Baseline with all physics disabled
- `granite_vacuum`: Granite stability in vacuum

### Water Tests (`test_water.py`)
- `water_conservation`: Water conservation with surface cavities
- `water_stress`: Aggressive conservation test
- `water_blob`: Water bar condensing into blob
- `water_blob_thin`: Thin water bar variant
- `water_line`: Water line surface tension collapse
- `water_droplet`: Water droplet formation
- `water_diagnostic`: Water leakage diagnostics

### Gravity & Buoyancy Tests (`test_gravity_buoyancy.py`)
- `gravity_water`: Water falls toward rock planet (currently failing)
- `gravity_magma`: Magma falls toward rock planet (currently failing)
- `rock_on_ice`: Rock falls through melting ice (currently failing)
- `buoyancy_air_water`: Air bubble rises in water (skipped - not working)
- `buoyancy_space_water`: Space bubble rises in water (skipped - not working)
- `buoyancy_air_magma`: Air bubble rises in magma (skipped - not working)

**Note**: Gravity and buoyancy tests are currently failing due to physics implementation issues. The gravity is realistically weak for small planets, and the force-based swapping requires velocity differences that prevent simple buoyancy from working.

## Best Practices

1. **Clear Success Criteria**: Define unambiguous pass/fail conditions
2. **Meaningful Metrics**: Track quantitative values for debugging
3. **Descriptive Messages**: Provide context about what went wrong
4. **Visualization Hints**: Help users focus on relevant areas
5. **Parameterization**: Make scenarios configurable for different cases

## Debugging Tips

1. **Start Paused**: Visual tests start paused - step through with 'R'
2. **Use Screenshots**: Press 'S' to capture interesting states
3. **Watch Metrics**: The right panel shows real-time metrics
4. **Focus Regions**: Yellow boxes highlight areas of interest
5. **Phase Testing**: Disable physics modules to isolate issues

## Architecture Benefits

- **Reusability**: Write once, run headless or visual
- **Maintainability**: Consistent structure across all tests
- **Debuggability**: See exactly what's happening in failing tests
- **Documentation**: Scenarios self-document their purpose
- **Flexibility**: Easy to add new scenarios or modify existing ones 