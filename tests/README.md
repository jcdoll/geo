# Test Scenario Framework

This directory contains a unified test framework that allows tests to be run both headlessly (for CI/CD) and with real-time visualization for debugging and understanding physics behavior.

## Overview

The framework consists of:

1. **Base Classes** (`test_framework.py`):
   - `TestScenario`: Abstract base for all test scenarios
   - `ScenarioRunner`: Runs scenarios and collects results
   - `ModuleDisabler`: Selectively disable physics modules

2. **Scenario Implementations**:
   - Test-specific logic inheriting from `TestScenario`
   - Define setup, evaluation criteria, and visualization hints

3. **Pytest Integration**:
   - Standard pytest files that use scenarios
   - Can run in CI/CD without display

4. **Visual Runner** (`run_visual_tests.py`):
   - Interactive visualization of any scenario
   - Real-time metrics and status display

## Creating a New Test Scenario

### 1. Create Scenario Class

```python
from tests.test_framework import TestScenario
from materials import MaterialType
from geo_game import GeoGame

class MyTestScenario(TestScenario):
    def get_name(self) -> str:
        return "my_test_scenario"
    
    def get_description(self) -> str:
        return "Tests that my feature works correctly"
    
    def setup(self, sim: GeoGame) -> None:
        """Set up initial conditions"""
        # Configure simulation state
        sim.material_types[10, 10] = MaterialType.WATER
        sim.temperature[10, 10] = 300.0
        sim._update_material_properties()
    
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate current state"""
        success = check_my_conditions(sim)
        
        return {
            'success': success,
            'metrics': {
                'my_metric': calculate_metric(sim),
                'step': sim.time_step
            },
            'message': f"Test {'passed' if success else 'failed'}"
        }
    
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Optional visualization configuration"""
        return {
            'highlight_materials': [MaterialType.WATER],
            'focus_region': (5, 15, 5, 15),  # y_min, y_max, x_min, x_max
            'show_metrics': ['my_metric', 'step']
        }
```

### 2. Create Pytest File

```python
import pytest
from tests.test_framework import ScenarioRunner
from tests.my_test_scenarios import MyTestScenario

def test_my_feature():
    """Test my feature using the scenario."""
    scenario = MyTestScenario()
    runner = ScenarioRunner(scenario)
    result = runner.run_headless(max_steps=100)
    assert result['success'], f"Test failed: {result['message']}"
```

### 3. Add to Visual Runner

Edit `run_visual_tests.py` to add your scenario:

```python
from tests.my_test_scenarios import MyTestScenario

SCENARIOS = {
    # ... existing scenarios ...
    'my_test': MyTestScenario,
}
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

### Magma Tests
- `magma_containment`: Tests magma stays contained by solid rock
- `magma_no_physics`: Baseline with all physics disabled
- `magma_heat_only`: Only heat transfer enabled
- `magma_fluid_only`: Only fluid dynamics enabled
- `magma_binding`: Tests binding force configuration

### Water Tests
- `water_blob`: Water bar condenses into circular blob
- `water_conservation`: Tests water is conserved with surface cavities
- `water_stress`: Aggressive conservation test
- `water_phase`: Conservation with specific physics disabled

### Mechanics Tests
- `chunk_settle`: Single chunk falling with terminal velocity
- `multi_chunk`: Multiple materials falling simultaneously

### Surface Effects
- `water_line`: Thin water line collapses via surface tension
- `water_droplet`: Scattered water coalesces into droplets

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