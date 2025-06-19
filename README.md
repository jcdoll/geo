# 2D Geology Simulator

A real-time interactive geological simulation system that models heat transfer, pressure dynamics, and rock metamorphism in a 2D cross-section of the Earth's crust.

## Features

### Core Simulation
- **Heat Transfer**: Realistic heat diffusion using finite difference methods
- **Pressure Dynamics**: Lithostatic pressure calculation based on rock density
- **Rock Metamorphism**: P-T based metamorphic transitions (e.g., shale → slate → schist → gneiss)
- **Magma Processes**: Rock melting and cooling with appropriate igneous rock formation
- **Time Reversibility**: Full backward time stepping capability

### Interactive Visualization
- **Real-time Rendering**: 60 FPS pygame-based visualization
- **Multiple Display Modes**:
  - Rock types (color-coded by geological classification)
  - Temperature (thermal gradient visualization)
  - Pressure (pressure gradient visualization)
- **Interactive Tools**:
  - Heat source placement (simulate magma intrusions)
  - Pressure application (simulate tectonic stress)
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
- **`fluid_dynamics.py`**: Fluid flow and material swapping
- **`gravity_solver.py`**: Gravitational field calculations using Poisson solver
- **`pressure_solver.py`**: Pressure field calculations
- **`atmospheric_processes.py`**: Atmospheric physics and greenhouse effects
- **`material_processes.py`**: Rock metamorphism and phase transitions

### Materials System
- **`materials.py`**: Material types, properties, and metamorphic transitions
- **`visualizer.py`**: Interactive pygame-based visualization with multiple display modes

### Key Physical Models

- **Gravitational Physics**: Dynamic gravity field calculation using Poisson equation solver
- **Heat Transfer**: Operator splitting approach with configurable radiative cooling methods
- **Cell-Swapping Mechanics**: Physics-based material movement using force thresholds
- **Surface Tension**: Curvature-based cohesive forces for fluid interface minimization
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

## Technical Details

See [PHYSICS.md](PHYSICS.md) for details.

## Testing

See the [testing readme](./tests/README.md).

## License

This project is open source and available under the MIT License.
