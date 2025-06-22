# 2D Geology Simulator

A fast, fun planetary simulation system that models geological processes using simplified physics. Everything flows based on material viscosity - no rigid bodies!

## Features

### Core Simulation
- **Viscosity-Based Flow**: All materials flow at different rates (rocks slowly, water quickly)
- **Self-Gravity**: Dynamic gravitational field calculation
- **Heat Transfer**: Realistic heat diffusion with material-based heating (uranium!)
- **Pressure Dynamics**: Proper hydrostatic/lithostatic pressure using multigrid solver
- **Material Transitions**: Temperature/pressure-based phase changes (ice↔water↔vapor, rock→magma)
- **Performance**: ~22 FPS for 100x60 grid (1.5x faster with recent optimizations)

### Interactive Visualization
- **Real-time Rendering**: Pygame-based visualization
- **Multiple Display Modes**:
  - Material types (color-coded by geological classification)
  - Temperature (thermal gradient visualization)
  - Pressure (pressure gradient visualization)
  - Power/energy flow
- **Interactive Tools**:
  - Heat source placement (simulate magma intrusions)
  - Pressure application (simulate tectonic stress)
  - Material placement and editing
  - Real-time inspection of cell properties

### User Controls
- **Time Control**: Play/pause, step forward/backward, adjustable time steps
- **Interactive Editing**: Click and drag to modify simulation conditions
- **Multiple Visualization Modes**: Switch between rock types, temperature, and pressure views

## Installation

Abbreviated instructions are as follows:

   ```bash
   wsl --install -d Ubuntu
   cd geo
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python main.py
   ```

## Usage

### Controls
Controls are printed to the terminal when you run the simulator. Press H in the simulation for complete controls help.

## Geological Processes Modeled

### Rock Types
- **Igneous**: Granite, Basalt, Obsidian, Pumice, Andesite
- **Sedimentary**: Sandstone, Limestone, Shale, Conglomerate
- **Metamorphic**: Gneiss, Schist, Slate, Marble, Quartzite

### Metamorphic Transitions
- Shale → Slate → Schist → Gneiss (increasing P-T)
- Sandstone → Quartzite
- Limestone → Marble
- Granite → Gneiss

### Physical Properties
Each rock type has realistic:
- Density (kg/m³)
- Thermal conductivity (W/m·K)
- Specific heat (J/kg·K)
- Melting point (°C)
- Compressive strength (MPa)
- Porosity

## Architecture

### Modular Physics Engine
The codebase uses a modular architecture with physics domains isolated in separate modules:

- **`geo_game.py`**: Main simulation facade, inherits from CoreState + CoreToolsMixin
- **`core_state.py`**: Shared state and grid allocation for physics modules
- **`core_tools.py`**: Interactive tools mixin (heat sources, pressure application)

### Physics Modules
Each physics domain is isolated in its own module:
- **`heat_transfer.py`**: Heat diffusion calculations using operator splitting
- **`fluid_dynamics.py`**: **Simplified viscosity-based flow** (no rigid bodies!)
- **`gravity_solver.py`**: Self-gravity field calculations using multigrid Poisson solver
- **`pressure_solver.py`**: Hydrostatic pressure field calculations
- **`atmospheric_processes.py`**: Atmospheric physics and greenhouse effects
- **`material_processes.py`**: Rock metamorphism and phase transitions

### Materials System
- **`materials.py`**: Material types, properties, and metamorphic transitions
- **`visualizer.py`**: Interactive pygame-based visualization with multiple display modes

### Key Physical Models

- **Simplified Movement**: All materials flow based on viscosity (0.0 = no resistance, 1.0 = no flow)
- **Self-Gravity**: Dynamic gravity field calculation using multigrid Poisson solver
- **Heat Transfer**: Operator splitting with unconditional stability
- **Material-Based Heating**: Uranium and other materials generate heat
- **No Rigid Bodies**: Everything flows - rocks slowly (viscosity 0.9), water quickly (0.05)
- **Atmospheric Effects**: Greenhouse warming and enhanced turbulent mixing

## Examples

### Scenario 1: Magma Intrusion
1. Start simulation
2. Select "Heat Source" tool
3. Click and drag in lower crust area
4. Watch heat propagate upward
5. Observe metamorphic aureole formation

### Scenario 2: Tectonic Compression
1. Select "Pressure" tool
2. Apply pressure to sedimentary layers
3. Observe metamorphic transitions
4. Use backward stepping to see process in reverse

### Scenario 3: Deep Crustal Processes
1. Set simulation to auto-step
2. Add multiple heat sources at depth
3. Watch long-term thermal evolution
4. Switch between visualization modes

## Performance Optimizations

Recent optimizations have significantly improved simulation speed:
- **Vectorized movement**: 7.7x speedup (19ms → 2.5ms) through batch processing
- **Optimized time series**: Removed unused planet_radius calculation
- **Pre-allocated buffers**: Reduced memory allocation overhead
- **Material count caching**: Only recalculate when materials change

## Technical Details

See [PHYSICS.md](PHYSICS.md) for details.

## Testing

See the [testing readme](./tests/README.md).

## License

This project is open source and available under the MIT License.
