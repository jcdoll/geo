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

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Simulator**:
   ```bash
   python main.py
   ```

## Usage

### Basic Controls
- **Space**: Play/Pause simulation
- **R**: Step forward in time
- **T**: Step backward in time
- **1/2/3**: Switch between display modes (rocks/temperature/pressure)

### Interactive Tools
- **Left Click + Drag**: Apply selected tool to simulation
- **Mouse Wheel**: Adjust tool radius
- **Shift + Mouse Wheel**: Adjust tool intensity

### Available Tools
1. **Heat Source**: Add thermal energy (simulate magma intrusions)
2. **Pressure**: Apply tectonic stress
3. **Inspect**: View detailed cell properties

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

### Core Components

1. **`rock_types.py`**: Rock classification system with physical properties
2. **`simulation_engine.py`**: Main simulation engine with physics calculations
3. **`visualizer.py`**: Interactive pygame-based visualization
4. **`main.py`**: Entry point and user interface

### Key Algorithms

- **Heat Diffusion**: Finite difference solution to the heat equation
- **Pressure Calculation**: Lithostatic pressure from overlying rock weight
- **Metamorphic Logic**: P-T phase diagrams for rock transitions
- **Time Stepping**: Explicit forward Euler with history tracking

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

### Performance Optimizations
- NumPy vectorized operations for grid computations
- Numba JIT compilation for heat diffusion kernel
- Efficient pygame rendering with surface blitting
- Sparse history storage for time reversal

### Simulation Parameters
- **Grid Size**: 100x60 cells (configurable)
- **Cell Size**: 1000m (1 km per cell)
- **Time Step**: 1000 years (adjustable)
- **Temperature Range**: 15°C surface to 1500°C+ at depth
- **Pressure Range**: 0.1 MPa surface to 1000+ MPa at depth

## Limitations and Future Enhancements

### Current Limitations
- 2D simulation only (no 3D effects)
- Simplified fluid dynamics
- No plate tectonics simulation
- Limited weathering and erosion

### Potential Enhancements
- Fluid flow modeling
- Plate boundary interactions
- Surface processes (weathering, erosion, sedimentation)
- More complex geochemistry
- Structural geology (faulting, folding)

## License

This project is open source and available under the MIT License.

## Next steps

- Additional rock types and transitions
- More sophisticated physics models
- Better visualization and UI
- Performance optimizations
- Educational materials and tutorials 