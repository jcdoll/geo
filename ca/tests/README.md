# Test Suite

This directory contains a comprehensive test suite for the geological physics simulation, organized by test type and purpose.

## 🚀 Quick Start - Running Tests

**IMPORTANT: A Makefile has been created to make running tests easy. From the project root:**

```bash
# Run all tests with verbose output
make test

# Run only previously failed tests
make test-failed

# Run tests stopping at first failure (quick debugging)
make test-quick

# Get a summary count of passed/failed tests
make test-summary

# Run specific test suites
make test-unit        # Only unit tests
make test-scenarios   # Only scenario tests
make test-integration # Only integration tests
```

**Alternative: Use the bash script directly:**
```bash
./tests/run_all_tests.sh
```

**Note:** Both methods automatically activate the virtual environment and set PYTHONPATH.

## Directory Structure

```
tests/
├── scenarios/          # Dual-mode test scenarios (headless & visual)
│   ├── base.py        # Enhanced TestScenario base class
│   ├── fluids.py      # Fluid dynamics scenarios
│   ├── mechanics.py   # Gravity, buoyancy, pressure scenarios
│   ├── materials.py   # Phase transitions, metamorphism scenarios
│   └── test_scenarios.py  # Pytest wrapper functions
├── framework/          # Test infrastructure
├── unit/              # Pure unit tests
├── integration/       # Full system integration tests
├── diagnostics/       # Debugging and analysis tools
└── legacy_scenarios/  # Original scenarios (being migrated)
```

## Test Organization

### Scenario Tests (`scenarios/`)
**Dual-mode scenarios that can run headless (pytest) or visual (interactive):**

#### Fluid Dynamics (`fluids.py`)
- `WaterConservationScenario` - Water volume conservation during flow
- `WaterConservationStressScenario` - Aggressive conservation test
- `WaterBlobScenario` - Blob cohesion and shape evolution
- `WaterLineCollapseScenario` - Tall water column collapse
- `MagmaFlowScenario` - Magma flow and cooling to basalt
- `FluidGravityScenario` - Fluid behavior near gravitational body

#### Mechanical Physics (`mechanics.py`)
- `GravityFallScenario` - Objects falling under gravity
- `BuoyancyScenario` - Buoyancy forces in fluids
- `HydrostaticPressureScenario` - Pressure gradients in static fluids

#### Material Physics (`materials.py`)
- `MaterialStabilityScenario` - Material stability in various environments
- `MetamorphismScenario` - Rock metamorphism under pressure/temperature
- `PhaseTransitionScenario` - Melting, freezing, vaporization


### Unit Tests (`unit/`)
Pure functionality tests for individual components:
- **Material Properties**: `test_materials.py`, `test_material_cache_cleanup.py`
- **Physics Solvers**: `test_pressure_solver.py`, `test_poisson_solver.py`
- **Fluid Dynamics**: `test_buoyancy_physics.py`
- **Visualization**: `test_visualizer_functionality.py`

### Integration Tests (`integration/`)
Tests for complete system behavior:
- `test_simulation_lifecycle.py` - Full simulation initialization, stepping, reset
- `test_integration.py` - Complete workflow tests
- `test_reset_simulation.py` - Reset functionality
- `test_space_integrity.py` - Conservation laws

### Diagnostic Tools (`diagnostics/`)
Specialized debugging utilities:
- `test_acceleration_analysis.py` - Detailed motion and force tracking
- `test_pressure_diagnosis.py` - Pressure field analysis

## Running Tests

### Headless Testing (pytest)
```bash
# Run all scenario tests
pytest tests/scenarios/test_scenarios.py -v

# Run specific test category
pytest tests/scenarios/test_scenarios.py -k "water" -v
pytest tests/scenarios/test_scenarios.py -k "gravity" -v
pytest tests/scenarios/test_scenarios.py -k "material" -v

# Run with specific parameters
pytest tests/scenarios/test_scenarios.py::test_water_conservation -v

# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest --cov=. tests/
```

### Visual Testing (Interactive)
```bash
# List all available scenarios
python tests/run_visual_tests.py --list

# Run specific scenario visually
python tests/run_visual_tests.py water_conservation
python tests/run_visual_tests.py gravity_fall
python tests/run_visual_tests.py magma_flow

# Run with custom parameters
python tests/run_visual_tests.py water_blob --size 80 blob_width=30 blob_height=15
```

### Visual Test Controls
When running visual tests:
- **SPACE** - Play/Pause simulation
- **RIGHT** - Step forward
- **M** - Cycle display modes
- **R** - Reset scenario
- **ESC** - Exit

## Test Categories

### Fluids (6 scenarios)
- Water conservation (basic & stress test)
- Fluid collapse and flow
- Magma dynamics
- Gravitational fluid behavior

### Mechanics (3 scenarios)
- Gravity fall for different materials
- Buoyancy in various fluid/object combinations
- Hydrostatic pressure profiles

### Materials (3 scenarios)
- Material stability in different environments
- Rock metamorphism under pressure/heat
- Phase transitions (ice→water, water→vapor, etc.)


## Creating New Tests

### Adding a New Scenario
1. Choose the appropriate category file (`fluids.py`, `mechanics.py`, etc.)
2. Create a class inheriting from `TestScenario`
3. Implement required methods:
   ```python
   class MyNewScenario(TestScenario):
       def get_name(self) -> str:
           return "my_scenario_name"
       
       def get_description(self) -> str:
           return "What this scenario tests"
       
       def setup(self, sim: GeoGame) -> None:
           """Create initial conditions - don't use default planet!"""
           sim.material_types[:] = MaterialType.AIR
           # Set up your specific test conditions
       
       def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
           """Check success criteria"""
           success = check_your_conditions(sim)
           return {
               'success': success,
               'metrics': {'key': value},
               'message': "Current state description"
           }
   ```

4. Add pytest wrapper in `test_scenarios.py`:
   ```python
   def test_my_new_scenario():
       """Test description."""
       scenario = MyNewScenario(param1=value1)
       runner = ScenarioRunner(scenario, sim_width=60, sim_height=60)
       result = runner.run_headless(max_steps=100)
       assert result['success'], result['message']
   ```

5. Register in `__init__.py` for visual runner:
   ```python
   SCENARIO_GROUPS['category'].add_scenario('my_scenario', MyNewScenario)
   ```

## Key Design Principles

1. **Self-Contained Tests**: Each scenario creates its own environment, no default planet dependency
2. **Dual Mode**: Same scenario logic for both headless (CI) and visual (debugging)
3. **Deterministic**: Use fixed random seeds for reproducibility
4. **Timeout Support**: Prevent hanging tests with timeout parameter
5. **Clear Success Criteria**: Each scenario defines what constitutes success
6. **Meaningful Metrics**: Track relevant physical quantities

## Environment Configuration

- **Headless Mode**: Automatically configured via `conftest.py`:
  - `SDL_VIDEODRIVER=dummy` for pygame
  - `MPLBACKEND=Agg` for matplotlib
  
- **Python Path**: Tests automatically add project root to sys.path

## Performance Testing

Tests marked with `@pytest.mark.slow` are performance tests:
```bash
# Run including slow tests
pytest tests/scenarios/ -v --slow

# Skip slow tests (default)
pytest tests/scenarios/ -v
```

## Known Physics Issues

With the simplified viscosity-based approach:
1. **Everything flows** - All materials move based on viscosity (rocks slowly, water quickly)
2. **No rigid bodies** - Removed for simplicity and performance
3. **Material transitions** - Still occur based on temperature/pressure conditions

## Migration Status

- ✅ All scenarios migrated from legacy_scenarios
- ✅ Organized into logical categories
- ✅ Pytest wrappers created for all scenarios
- ✅ Visual runner updated with all scenarios
- ✅ Documentation updated
- ⏳ Legacy_scenarios directory can be removed after verification