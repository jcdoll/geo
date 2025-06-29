# SPH Geological Simulation

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with default planet scenario
python main.py

# Run with different scenarios
python main.py --scenario water
python main.py --scenario volcanic

# Run with specific backend
python main.py --backend numba  # Fast CPU
python main.py --backend cpu    # Pure NumPy

# Adjust size and particles
python main.py --size 200 --particles 5000
```

## Controls

The visualizer includes a full toolbar UI adapted from the flux implementation:

### Keyboard Controls
- **SPACE** - Pause/Resume simulation
- **RIGHT** - Step forward one frame (when paused)
- **TAB/M** - Cycle display modes
- **T** - Cycle tools
- **1-9** - Select material
- **H** - Toggle help
- **I** - Toggle info display
- **L** - Toggle performance metrics
- **S** - Save screenshot
- **ESC** - Exit

### Mouse Controls
- **Left Click** - Apply current tool
- **Right Click** - Select particle for inspection
- **Shift+Click** - Reverse tool action
- **Scroll** - Adjust tool radius
- **Shift+Scroll** - Adjust tool intensity

### Tools
1. **Material (M)** - Add/change particle materials
2. **Heat (H)** - Add/remove heat
3. **Velocity (V)** - Set particle velocities
4. **Delete (D)** - Remove particles

### Display Modes
- **Material** - Color by material type
- **Temperature** - Heat map
- **Pressure** - Pressure field
- **Velocity** - Speed visualization
- **Density** - Density field
- **Phase** - Solid/liquid/gas states

### Physics Modules
Toggle on/off via toolbar checkboxes:
- Gravity
- Pressure forces
- Viscosity
- Heat transfer
- Phase transitions

## Scenarios

### Planet
Basic rocky planet with gravity. Great for testing material interactions.

### Water
Water drop scenario for fluid dynamics testing.

### Volcanic
Planet with hot core showing magma dynamics (uses earth-like layered planet).

### Layered
Multi-layer planet with different materials.

### Thermal
Thermal test scenario for heat transfer validation.

## Performance

The simulation automatically selects the best backend:
- **< 1,000 particles**: CPU (NumPy)
- **1,000 - 50,000 particles**: Numba JIT
- **> 50,000 particles**: GPU (if available)

Typical performance:
- CPU: ~10 FPS for 1,000 particles
- Numba: ~100 FPS for 1,000 particles
- Numba: ~30 FPS for 5,000 particles

## Implementation

The SPH implementation features:
- Fully vectorized for CPU/GPU efficiency
- Three-tier backend system (CPU/Numba/GPU)
- Material system with 9 material types
- Phase transitions with latent heat
- Self-gravity and external gravity
- Heat conduction and radiation
- Interactive visualization with toolbar UI

## Testing

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run all SPH tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test files
pytest tests/test_sph_physics.py
pytest tests/test_sph_stability.py

# Run SPH module tests
pytest sph/test_*.py

# Run all SPH-related tests
pytest tests/ sph/test_*.py -v
```

## Files

- `main.py` - Main entry point
- `sph/` - SPH implementation package
  - `core/` - Particle data structures, kernels, spatial hashing
  - `physics/` - Forces, materials, thermal, gravity
  - `scenarios/` - Initial conditions
  - `visualizer.py` - Interactive visualization
  - `api.py` - Unified API with backend dispatch
- `tests/` - Test suite
  - `test_sph_physics.py` - Physics validation tests
  - `test_sph_stability.py` - Numerical stability tests