# Geology Simulator

A modern 2D geological simulation using flux-based physics for realistic material flow and transitions.

## Quick Start

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the simulation
python main.py                      # Default Earth-like planet
python main.py --scenario volcanic  # Volcanic island
python main.py --scenario ice       # Ice world with subsurface ocean
python main.py --size 200          # Larger grid
```

## Features

### Flux-Based Physics (NEW!)
- **Volume Fractions**: Each cell can contain multiple materials (e.g., 30% water, 70% air)
- **Continuous Transport**: Materials flow based on physical flux calculations
- **Conservation Laws**: Exact mass, momentum, and energy conservation
- **Unified Framework**: All materials use the same transport equations

### Core Simulation
- **Multi-Material Cells**: Realistic mixing and interfaces
- **Self-Gravity**: Dynamic gravitational field using multigrid solver
- **Phase Transitions**: Temperature/pressure-based changes (ice↔water↔vapor, rock→magma)
- **Heat Transfer**: Conduction, radioactive heating, solar radiation
- **Fluid Dynamics**: Navier-Stokes with variable viscosity
- **Performance**: Target 30+ FPS at 128×128 grid

### Interactive Features
- **Real-Time Visualization**: Multiple display modes
- **Interactive Tools**: Add heat, pressure, materials
- **Time Control**: Pause, step, adjust speed
- **Scenarios**: Pre-built worlds to explore

## Usage

### Main Application

```bash
python main.py [options]

Options:
  --scenario {planet,volcanic,ice,empty}  Initial world type
  --size N                                Grid size (default: 128)
  --scale M                               Cell size in meters (default: 50)
  --width W --height H                    Custom dimensions
```

### Scenarios

1. **Planet** (default)
   - Atmosphere, ocean, and rocky crust
   - Uranium deposits for heat generation
   - Active weather and erosion

2. **Volcanic Island**
   - Island with magma chamber
   - Surrounding ocean
   - Active volcanism

3. **Ice World**
   - Frozen surface
   - Subsurface liquid ocean
   - Tidal heating effects

4. **Empty**
   - Blank canvas
   - Build your own world

### Controls

**Simulation**
- `SPACE` - Pause/Resume
- `←/→` - Step backward/forward
- `R` - Reset
- `ESC` - Exit
- `H` - Show help

**Display Modes**
- `1` - Material (dominant)
- `2` - Material (composite)
- `3` - Temperature
- `4` - Pressure
- `5` - Velocity
- `6` - Density

**Tools** (number keys select tools, then click/drag to use)
- Heat sources
- Pressure application  
- Material placement
- And more...

## Materials

| Material | Properties | Transitions |
|----------|------------|-------------|
| Water | Flows easily, moderate density | →Ice (<0°C), →Vapor (>100°C) |
| Ice | Solid, less dense than water | →Water (>0°C) |
| Rock | Very viscous flow, high density | →Magma (>1200°C) |
| Magma | Viscous flow, radiates heat | →Rock (<1200°C) |
| Sand | Granular flow, medium density | Forms from rock weathering |
| Air | Low density, low viscosity | Greenhouse effects |
| Uranium | Radioactive, generates heat | No transitions |
| Space | Vacuum, no properties | No transitions |

## Architecture

### Flux-Based Approach (NEW!)
The new flux-based approach replaces the cellular automata (CA) system:

**Core Modules**
- `state.py` - Simulation state with volume fractions
- `transport.py` - Flux-based mass/momentum transport
- `physics.py` - Gravity, pressure, thermal physics
- `materials.py` - Material properties and transitions
- `simulation.py` - Main loop with operator splitting
- `visualizer.py` - Interactive pygame visualization
- `main.py` - Entry point with scenarios

**Key Differences from CA**
1. **Volume Fractions**: Multiple materials per cell vs single material
2. **Continuous Physics**: Flux equations vs discrete rules
3. **Better Conservation**: Exact conservation vs approximate
4. **No Rigid Bodies**: Everything flows based on viscosity
5. **Unified Transport**: Same equations for all materials

### Key Concepts

1. **Volume Fractions** (φᵢ)
   - Each material has fraction 0 ≤ φᵢ ≤ 1
   - Sum equals 1: Σφᵢ = 1
   - Mixture properties: ρ = Σ(φᵢρᵢ)

2. **Flux Transport**
   - Face-centered fluxes for conservation
   - Upwind scheme for stability
   - No artificial diffusion

3. **Operator Splitting**
   - Separate physics solved sequentially
   - Maintains accuracy and stability
   - Allows different time scales

## Testing

### Run Test Suite
```bash
pytest tests/                    # All tests
pytest tests/ -v                # Verbose
pytest tests/ -k "water"        # Specific tests
```

### Visual Testing
```bash
# List available test scenarios
python tests/run_visual_tests.py --list

# Run specific scenario with visualization
python tests/run_visual_tests.py water_drop_fall
python tests/run_visual_tests.py heat_diffusion --size 80
```

Test categories:
- **Fluid Dynamics**: Hydrostatic equilibrium, buoyancy, flow
- **Phase Transitions**: Freezing, melting, evaporation
- **Thermal Physics**: Diffusion, radiation, nuclear heating

## Development

### Performance
- Vectorized numpy operations
- Optional Numba JIT compilation
- Pre-allocated arrays for efficiency

### Adding Features
1. New materials → Edit `materials.py`
2. New physics → Add to `physics.py`
3. New tools → Extend `visualizer.py`
4. New scenarios → Add to `main.py`

### Debugging
```bash
# Profile performance
python -m cProfile -o profile.dat main.py --size 64

# Test specific physics
python tests/run_visual_tests.py <scenario_name>
```

## Physics Details

See [PHYSICS_FLUX.md](PHYSICS_FLUX.md) for detailed physics documentation including:
- Conservation equations
- Numerical methods
- Material properties
- Phase transition rules

## References

### Simulation Methods
- **Eulerian**: Fixed grid, flux-based (this project)
- **Lagrangian**: Particle-based, moving mesh
- **SPH**: Smoothed Particle Hydrodynamics
- **CA**: Cellular Automata (old version)

### Inspirations
- Games:
  - [Noita](https://store.steampowered.com/app/881100/Noita/)
  - [Powder Game](https://dan-ball.jp/en/javagame/dust/)
  - [The Powder Toy](https://powdertoy.co.uk/)
- Simulators:
  - [ASPECT](https://aspect.geodynamics.org/)
  - [ConMan](https://geodynamics.org/resources/conman)
- Tutorials:
  - [Mantle Convection](https://www.youtube.com/watch?v=x6mcua0HOJs)
  - [Plate Tectonics](https://www.youtube.com/watch?v=iKAVRgIrUOU)
  - [Earth Structure](https://www.youtube.com/watch?v=rSKMYc1CQHE)
  - [Geology Basics](https://www.youtube.com/watch?v=8nIB7e_eds4)
  - [SPH Tutorial](https://tommccracken.net/sph-fluid-simulation/)
- Papers:
  - [Planetary Formation](https://arxiv.org/abs/astro-ph/0610051)

## Legacy CA Version

The original cellular automata version is preserved in the `ca/` directory. It uses discrete cell states and local update rules rather than continuous flux calculations. This approach has numerous problems based on the quantization of cell state, for example incorrect pressures in uniform bodies of fluid. The continuous flux based approach solves these problems.

## License

This project is open source and available under the MIT License.