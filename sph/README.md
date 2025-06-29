# SPH-Based Geological Simulation

This directory contains the Smoothed Particle Hydrodynamics (SPH) implementation for the geological simulation project.

## Why SPH?

After encountering fundamental numerical instabilities with extreme density ratios (space: 0.001 kg/m³ vs uranium: 19,000 kg/m³) in the flux-based Eulerian approach, we've switched to SPH which naturally handles:

- Extreme density ratios without numerical instabilities
- Free surfaces and material interfaces
- Pseudo-rigid body physics through cohesive forces
- Mass conservation (exact by construction)

## Project Structure

```
sph/
├── PHYSICS.md           # Comprehensive physics reference
├── README.md           # This file
├── core/               # Core SPH implementation
│   ├── particle.py     # Particle data structures
│   ├── kernel.py       # SPH kernel functions
│   ├── spatial.py      # Spatial hashing for neighbor search
│   └── integrator.py   # Time integration schemes
├── physics/            # Physics modules
│   ├── density.py      # Density computation
│   ├── forces.py       # Force calculations
│   ├── thermal.py      # Heat transfer
│   ├── gravity.py      # Self-gravity (direct & tree-code)
│   └── materials.py    # Material properties & phase transitions
├── simulation.py       # Main simulation loop
├── visualization.py    # Particle rendering
└── tests/             # Test suite
    ├── test_kernel.py
    ├── test_conservation.py
    └── scenarios/      # Physical test scenarios
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation
cd ..  # Go to project root
python main.py
```

## Performance & GPU Acceleration

SPH supports three backends with automatic selection:

1. **CPU** - Pure NumPy (baseline)
2. **Numba** - JIT-compiled (~20-50x faster) ✅ **Recommended**  
3. **GPU** - CuPy/CUDA (50-200x faster for large simulations)

**Note for RTX 5080 users**: The pre-built CuPy doesn't support this GPU yet. The Numba backend provides excellent performance (50,000+ FPS for typical simulations).

See [GPU_SETUP.md](GPU_SETUP.md) for GPU setup details.

## Quick Start

```python
# The simulation automatically selects the fastest backend
import sph

# Check available backends
sph.print_backend_info()

# Manually select backend if needed
sph.set_backend('numba')  # Fast CPU backend (recommended)
```

## Key Features

1. **Natural Material Interactions**
   - No special cases for fluid vs solid
   - Material properties determine behavior
   - Smooth transitions during phase changes

2. **Pseudo-Rigid Bodies**
   - Cohesive forces between particles
   - Stress-based fracture
   - Realistic rock mechanics

3. **Extreme Density Ratios**
   - Stable with 10^6:1 density variations
   - No numerical tricks needed
   - Physical pressure calculation

4. **Conservation Laws**
   - Mass: Exact (particles have fixed mass)
   - Momentum: Preserved in interactions
   - Energy: Tracked including phase transitions

## Performance Targets

- 30+ FPS for 10,000 particles on CPU
- O(N) complexity with spatial hashing
- Optimized kernel evaluations
- Adaptive timestepping

## Implementation Phases

### Phase 1: Core SPH (Current)
- [x] Basic particle system design
- [ ] Kernel functions (cubic spline, quintic)
- [ ] Spatial hashing
- [ ] Density and pressure calculation
- [ ] Basic force integration

### Phase 2: Fluid Dynamics
- [ ] Pressure forces
- [ ] Viscosity (artificial and physical)
- [ ] Free surface detection
- [ ] Boundary conditions

### Phase 3: Solid Mechanics
- [ ] Stress tensor evolution
- [ ] Elastic forces
- [ ] Cohesive bonds
- [ ] Fracture mechanics

### Phase 4: Thermal Physics
- [ ] Heat conduction
- [ ] Radiation
- [ ] Phase transitions
- [ ] Latent heat

### Phase 5: Advanced Features
- [ ] Self-gravity (Barnes-Hut tree)
- [ ] Multiple materials
- [ ] Chemical reactions
- [ ] Performance optimization

## Design Principles

1. **Physics First**: All behaviors emerge from physical laws
2. **No Hacks**: No artificial limits or special cases
3. **Clean Code**: Modular design with clear interfaces
4. **Testable**: Comprehensive test coverage
5. **Fast**: Optimize after correctness

## Material Examples

```python
# Define custom material
class Granite(Material):
    def __init__(self):
        super().__init__(
            density=2700,          # kg/m³
            bulk_modulus=50e9,     # Pa
            shear_modulus=30e9,    # Pa
            yield_strength=200e6,  # Pa
            cohesion_strength=10e6,# Pa
            melting_point=1500,    # K
            specific_heat=790,     # J/(kg·K)
        )
```

## Visualization

The visualization system will show:
- Particles colored by material type
- Velocity vectors
- Temperature as hue
- Stress as brightness
- Cohesive bonds as lines

## Testing

Run tests with:
```bash
pytest sph/tests/
```

Key test scenarios:
- Dam break (standard SPH validation)
- Elastic bounce
- Heat diffusion
- Phase transitions
- Conservation verification

## References

1. Monaghan, J.J. (2005). "Smoothed particle hydrodynamics"
2. Gray, J.P. et al. (2001). "SPH elastic dynamics"
3. Bui, H.H. et al. (2008). "Lagrangian meshfree methods for solid mechanics"