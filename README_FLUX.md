# Flux-Based Geological Simulation

A modern, flux-based approach to geological simulation using volume fractions and continuous transport.

## Quick Start

```bash
# Run default planet scenario
python main.py

# Run specific scenarios
python main.py --scenario volcanic --size 150
python main.py --scenario ice
python main.py --scenario empty --size 200 --scale 100

# Run test scenarios visually
python tests/run_visual_tests.py --list
python tests/run_visual_tests.py water_drop_fall
```

## Architecture

The flux-based approach differs fundamentally from cellular automata:

- **Volume Fractions**: Each cell can contain multiple materials (e.g., 30% water, 70% air)
- **Continuous Transport**: Materials flow based on flux calculations, not discrete swaps
- **Conservation Laws**: Exact mass, momentum, and energy conservation
- **Unified Flow**: All materials flow based on viscosity (rocks flow very slowly)

### Core Modules

- `state.py` - Simulation state with volume fractions and mixture properties
- `transport.py` - Flux-based mass transport with upwind scheme
- `physics.py` - Gravity, pressure, momentum equations
- `materials.py` - Material properties and phase transitions
- `simulation.py` - Main simulation loop with operator splitting
- `visualizer.py` - Interactive visualization
- `optimizations.py` - Numba JIT compilation for performance

### Key Concepts

1. **Volume Fractions** (φᵢ):
   - Each material has a volume fraction 0 ≤ φᵢ ≤ 1
   - Sum of all fractions equals 1: Σφᵢ = 1
   - Mixture density: ρ = Σ(φᵢρᵢ)

2. **Face-Centered Fluxes**:
   - Fluxes computed at cell faces for exact conservation
   - Upwind scheme for numerical stability
   - No artificial diffusion

3. **Material Transitions**:
   - Temperature/pressure-based phase changes
   - Latent heat tracking
   - Smooth transitions using rates

## Scenarios

### Interactive Scenarios (main.py)

1. **Planet** (default):
   - Atmosphere, ocean, and rocky crust
   - Uranium deposits for heat generation
   - Solar heating enabled

2. **Volcanic Island**:
   - Island with magma chamber
   - Ocean surrounding
   - Hot magma core

3. **Ice World**:
   - Ice shell over subsurface ocean
   - Rocky core
   - Temperature gradients

4. **Empty**:
   - Blank canvas for experimentation
   - Add materials with interactive tools

### Test Scenarios

Run visually: `python tests/run_visual_tests.py <scenario>`

**Fluid Dynamics**:
- `hydrostatic_equilibrium` - Water pressure gradient
- `water_drop_fall` - Gravity and spreading
- `buoyancy` - Ice floating on water

**Material Transitions**:
- `water_freezing` - Water → Ice
- `ice_melting` - Ice → Water
- `water_evaporation` - Water → Vapor
- `rock_melting` - Rock → Magma

**Thermal Physics**:
- `heat_diffusion` - Conduction through materials
- `uranium_heating` - Radioactive heat
- `solar_heating` - Surface albedo effects
- `radiative_cooling` - Stefan-Boltzmann cooling

## Controls

### Keyboard
- **SPACE**: Pause/Resume
- **R**: Reset simulation
- **S**: Save screenshot
- **H**: Show help
- **ESC**: Exit

### Display Modes
- **1**: Material view (dominant material)
- **2**: Material composite (blended)
- **3**: Temperature
- **4**: Pressure
- **5**: Velocity
- **6**: Density

### Tools (number keys)
- Various interactive tools for heat, pressure, materials

## Physics

The simulation uses operator splitting to solve:

1. **Self-gravity** (Poisson equation)
2. **Pressure** (hydrostatic + dynamic)
3. **Momentum** (Navier-Stokes)
4. **Advection** (material transport)
5. **Diffusion** (heat, momentum)
6. **Phase transitions** (T-P based)
7. **Solar heating** (ray marching)
8. **Radiative cooling** (Stefan-Boltzmann)

## Performance

- Vectorized numpy operations
- Optional Numba JIT compilation
- Typical: 30+ FPS for 128x128 grid
- Scales well with grid size

## Development

```bash
# Run tests
pytest tests/

# Run specific test visually
python tests/run_visual_tests.py water_freezing --size 80

# Profile performance
python -m cProfile -o profile.dat main.py --size 64
```

## Material Properties

| Material | Density | Viscosity | Melting | Boiling | Albedo |
|----------|---------|-----------|---------|---------|---------|
| Water    | 1000    | 0.001     | 273K    | 373K    | 0.5     |
| Ice      | 917     | 1e15      | 273K    | -       | 0.9     |
| Rock     | 2700    | 1e20      | 1473K   | -       | 0.3     |
| Magma    | 2400    | 100       | -       | -       | 0.7     |
| Air      | 1.2     | 1.8e-5    | -       | -       | 0.0     |

## Differences from CA Version

1. **No Rigid Bodies**: Everything flows based on viscosity
2. **Volume Fractions**: Multiple materials per cell
3. **Continuous Physics**: Not discrete cellular rules
4. **Better Conservation**: Exact mass/energy conservation
5. **Unified Framework**: All materials use same transport