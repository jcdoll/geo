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
- References to other key documents (`PHYSICS_CA.md`, `PHYSICS_FLUX.MD`, `README.md`)

## Commands

### CRITICAL: Python Virtual Environment
**ALWAYS activate the virtual environment before running ANY Python commands:**
```bash
source .venv/bin/activate
```
**⚠️ IMPORTANT: All Python tools (pytest, python, pip) REQUIRE the venv to be activated first!**

### Running the Simulation

#### Flux-Based Simulation (New)
```bash
# Run with default planet scenario
python main.py

# Run with specific scenario
python main.py --scenario volcanic

# Run with custom size
python main.py --scenario layered --size 150
```

#### CA-Based Simulation (Legacy)
```bash
python geo_visualizer.py
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

### Performance Optimizations
Recent optimizations have significantly improved simulation speed:
- **Vectorized movement**: 7.7x speedup (19ms → 2.5ms) through batch processing of cells
- **Time series optimization**: Removed unused planet_radius calculation, added material count caching
- **Pre-allocated buffers**: Reuse memory for movement operations
- **Overall improvement**: 22 FPS on 100x60 grid (was 15 FPS)

### Simplified Physics Engine
The codebase uses a modular architecture focused on **fast, viscosity-based flow**:

- `geo_game.py` - Main simulation facade, inherits from CoreState + CoreToolsMixin
- `core_state.py` - Shared state and grid allocation for physics modules (includes optimized time series)
- `core_tools.py` - Interactive tools mixin (heat sources, pressure application)

### Physics Modules
Each physics domain is isolated in its own module:
- `heat_transfer.py` - Heat diffusion calculations (material-based heat generation)
- `fluid_dynamics.py` - **Vectorized viscosity-based flow** (7.7x faster, no rigid bodies!)
- `gravity_solver.py` - Gravitational field calculations using Poisson solver
- `pressure_solver.py` - Pressure field calculations
- `atmospheric_processes.py` - Atmospheric physics
- `material_processes.py` - Rock metamorphism and phase transitions

**Key Simplification**: All materials flow based on viscosity - rocks flow slowly (0.9), water flows easily (0.05)

### Materials System
- `materials.py` - Material types, properties, and metamorphic transitions
- MaterialType enum covers igneous, sedimentary, metamorphic rocks plus fluids (water, magma, air, space) and uranium
- MaterialDatabase provides physical properties (density, thermal conductivity, melting points, heat generation)
- Uranium material provides radioactive heat generation (no phase transitions)

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

## Performance Notes

### Current Performance (2025-01-21)
- **Target**: 30+ FPS for 128x128 grids
- **Achieved**: ~13 FPS for 100x60 grid (77ms/step)
- **Previous**: ~4 FPS with rigid body mechanics

### Performance Breakdown
- Movement calculation: 28ms (main bottleneck)
- Time series recording: 19ms
- Heat diffusion: 9ms
- Other physics: ~20ms

### Optimization Opportunities
- Vectorize remaining movement loops
- Reduce time series recording frequency
- Consider numba JIT for hot paths
- GPU acceleration for field solvers

## Known Issues

### Surface Tension
- **REMOVED FROM CODEBASE**: Surface tension is fundamentally incompatible with geological-scale grids (50m cells)
- The scale mismatch is extreme - molecular forces (nanometers) vs geological cells (50 meters) 
- See PHYSICS.md for detailed explanation of why surface tension cannot work at these scales
- Water behaves correctly as a bulk fluid under gravity without surface tension

### Rigid Bodies
- **REMOVED FROM CODEBASE**: All rigid body mechanics have been removed for simplicity and performance
- Everything flows based on material viscosity - rocks flow very slowly, fluids flow quickly
- This simplification improved performance by 3x and makes the simulation more predictable

## Flux-Based Simulation (flux/)

The flux-based simulation provides a more physically accurate model using continuous fields and volume fractions:

### Architecture
- `main.py` - Entry point with command-line interface
- `scenarios.py` - **Centralized scenario definitions** (no duplication between files)
  - Available scenarios: empty, planet, layered, volcanic, ice
  - Each scenario is a function that configures the initial state
- `simulation.py` - Main simulation loop with operator splitting
- `state.py` - Volume fraction state management (multiple materials per cell)
- `transport.py` - Material advection (vectorized with 30x speedup) and heat diffusion
- `physics.py` - Gravity, pressure, and momentum calculations
- `materials.py` - Material properties database (9 materials)
- `visualizer.py` - Interactive visualization with toolbar UI

### Key Differences from CA Simulation
- **Volume fractions**: Multiple materials can exist in each cell
- **Continuous fields**: Temperature, pressure, velocity are continuous
- **Flux-based transport**: Exact conservation using face-centered fluxes
- **Better physics**: Proper momentum equations, phase transitions with latent heat

### Performance (Flux-Based)
- **Target**: 100+ FPS for 100x100 grids
- **Achieved**: 640 FPS with optimized transport
- **Bottlenecks**: Heat diffusion (54%), gravity solve (15%)

### Heat Transfer Methods
- **ADI (Alternating Direction Implicit)**: Current default in `heat_transfer.py`
  - ~30ms for 100x100 grid
  - Unconditionally stable for diffusion
  - Excellent performance for parabolic PDEs
- **Multigrid**: Alternative in `heat_transfer_multigrid.py`
  - ~300ms for 100x100 grid (10x slower than ADI)
  - Better suited for elliptic problems (pressure/gravity)
  - Not recommended for heat diffusion