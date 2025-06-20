# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Always Test Your Changes

**Before considering any task complete, ALWAYS test your code changes:**
1. For Python code: Run the affected files or tests to ensure no syntax/import errors
2. For test code: Run at least one test to verify it works
3. Check for basic errors like AttributeError, ImportError, etc.
4. If you get an error, iterate and fix it - don't leave broken code

## Important: Read AGENTS.md First

Before making any changes, **read `AGENTS.md`** which contains:
- AI assistant behavior rules and restrictions
- Required workflow (including running pytest before commits)
- Code style guidelines (PEP8, 120-char lines, double quotes)
- Anchor comment conventions (`AIDEV-NOTE`, `AIDEV-TODO`, `AIDEV-QUESTION`)
- References to other key documents (`PHYSICS.md`, `README.md`)

## Commands

### CRITICAL: Python Virtual Environment
**ALWAYS activate the virtual environment before running ANY Python commands:**
```bash
source .venv/bin/activate
```
**⚠️ IMPORTANT: All Python tools (pytest, python, pip) REQUIRE the venv to be activated first!**

### Running the Simulation
```bash
python main.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run tests with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_materials.py

# Run with coverage
pytest --cov=. tests/
```

### Visual Testing
```bash
# List available test scenarios
python tests/run_visual_tests.py --list

# Run specific scenario with visualization
python tests/run_visual_tests.py water_blob

# Run scenario with custom parameters
python tests/run_visual_tests.py water_conservation --steps 500 --size 80

# Note: Visual tests require a display. For headless testing use pytest instead:
# SDL_VIDEODRIVER=dummy may hang in pygame event loop
```

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run linting (if available)
# Note: Check if flake8, black, or other linters are configured
```

## Architecture

### Modular Physics Engine
The codebase has been refactored from a monolithic engine to a modular architecture:

- `geo_game.py` - Main simulation facade, inherits from CoreState + CoreToolsMixin
- `core_state.py` - Shared state and grid allocation for physics modules
- `core_tools.py` - Interactive tools mixin (heat sources, pressure application)

### Physics Modules
Each physics domain is isolated in its own module:
- `heat_transfer.py` - Heat diffusion calculations
- `fluid_dynamics.py` - Fluid flow and material swapping
- `gravity_solver.py` - Gravitational field calculations using Poisson solver
- `pressure_solver.py` - Pressure field calculations
- `atmospheric_processes.py` - Atmospheric physics
- `material_processes.py` - Rock metamorphism and phase transitions

### Materials System
- `materials.py` - Material types, properties, and metamorphic transitions
- MaterialType enum covers igneous, sedimentary, metamorphic rocks plus fluids (water, magma, air, space)
- MaterialDatabase provides physical properties (density, thermal conductivity, melting points)

### Visualization
- `visualizer.py` - Interactive pygame-based visualization with multiple display modes
- Real-time rendering of materials, temperature, pressure, and power
- Interactive tools for heat source placement and pressure application

### Legacy Code
- `deprecated/` - Contains original monolithic simulation engine for reference
- New code should use the modular architecture, not the deprecated engine

## Testing Framework

### Scenario-Based Testing
The test suite uses a unique dual-mode approach:
- Tests can run headlessly (pytest) or with real-time visualization 
- Scenarios are defined in test files and can be executed visually for debugging
- Test categories: `test_magma.py`, `test_water.py`, `test_gravity_buoyancy.py`

### Test Structure
- Inherit from `TestScenario` class for scenario-based tests
- Implement `setup()`, `evaluate()`, and description methods
- Use `ScenarioRunner` for execution in both modes
- Register scenarios in `SCENARIOS` dict for visual runner

### Visual Test Controls
When running visual tests (uses standard visualizer controls):
- SPACE: Play/Pause simulation
- LEFT arrow: Step backward
- RIGHT arrow: Step forward  
- R: Reset simulation
- H: Show complete controls help
- S: Save screenshot
- ESC: Exit

## Key Implementation Notes

### Import Handling
The codebase handles both package and direct execution imports using try/except blocks. When adding new modules, follow this pattern:

```python
try:
    from .module_name import SomeClass
except ImportError:
    from module_name import SomeClass
```

### Material Processing
- Material transitions are based on pressure-temperature conditions
- Use `_update_material_properties()` after modifying material_types grid
- Material cache cleanup is automatically handled but tested in `test_material_cache_cleanup.py`

### Grid Conventions
- Default grid: 100x60 cells at 50m cell size (5km x 3km domain)
- Coordinate system: (0,0) at top-left, +x right, +y down
- Arrays use [y, x] indexing following numpy convention

### Physics Integration
- Physics modules are called in sequence during `step_forward()`
- Each module operates on shared state arrays (temperature, pressure, velocity, etc.)
- Gravity solver uses finite difference Poisson equation for realistic gravitational fields

## Common Workflows

### Adding New Material Types
1. Add enum value to MaterialType in `materials.py`
2. Add properties to MaterialDatabase
3. Update visualization colors in `visualizer.py`
4. Add tests for new material behavior

### Adding New Physics
1. Create new module following existing patterns
2. Add to CoreState initialization 
3. Integrate into step_forward() sequence in geo_game.py
4. Add unit tests and scenario-based tests

### Debugging Physics Issues
1. Use visual test runner to see behavior in real-time
2. Add focus regions and metrics to scenario evaluation
3. Disable specific physics phases to isolate issues
4. Use screenshot capture for documenting bugs

## Known Issues

### Surface Tension Implementation Issues
- Fixed sign error (forces now point inward for convex droplets)
- However, the continuum CSF model creates instabilities on discrete grids:
  - Interface cells "dance" with fractal-like patterns
  - Forces push water outward in unstable oscillations
  - Curvature calculations are too noisy for 50m cells
- Root cause: The continuum surface tension formula (f = σκn|∇c|) doesn't work well on coarse discrete grids
- Possible solutions:
  1. Use the simpler pressure-based approach in `_compute_surface_tension_pressure`
  2. Implement nearest-neighbor cohesion forces
  3. Use energy minimization approach
  4. Add smoothing/damping to reduce noise
- Current workaround: Surface tension disabled in tests, relaxed criteria allow some fragmentation