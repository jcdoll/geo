# Geo - Geological Process Simulation

This repository contains three different approaches to simulating geological processes at planetary scales. Each implementation has different strengths and limitations.

This was an experiment in using Claude Code to generate a significant project.

Project conclusions:
* Simulating a planet is hard
* All options have significant upsides and downsides given the arbitrary constraints that I imposed (>3 fps)
* CA only really works with arbitrary cell swapping rules, because matter is discretized and so you can't solve for velocity and pressure in any meaningful way
    * In general solving for pressure in an accurate manner was a huge challenge, requiring velocity projection methods in the end
* The flux solver came close, except for handling the large density variation between rock and space (stiff system)
* SPH requires loose coupling between particles to go fast (for large timesteps) but it needs strong coupling to mimic solid-solid interactions

Claude Code conclusions:
* Poor IDE feedback compared with Cursor and Windsurf is a huge negative
* Even Opus 4 does incorrect things - fails to read CLAUDE.md, repeats the same error multiple times within a single session, fails to listen
* This was very useful to just pump out a huge amount of code to test ideas, but the quality was relatively poor

## Implementations

### 1. CA (Cellular Automata) - Simple and Stable
- **Directory**: `ca/`
- **Run**: `python ca/main.py`
- **Description**: Grid-based simulation with simple physics including self-gravity, thermal diffusion, solar heating, and radiative cooling. Cell swaps are density-based.
- **Strengths**: Fast, stable, handles large density variations well
- **Limitations**: Limited fluid dynamics, no true pressure solving

### 2. Flux - Advanced Grid Physics  
- **Directory**: `flux/`
- **Run**: `python flux/main.py`
- **Description**: Attempts to solve proper fluid dynamics using velocity projection for pressure. More physically accurate than CA.
- **Strengths**: Better fluid behavior, proper pressure gradients
- **Limitations**: Unstable with large density variations (space/rock), cannot simulate full planet

### 3. SPH (Smoothed Particle Hydrodynamics) - Meshless Approach
- **Directory**: `sph/`
- **Run**: `python sph/main.py`
- **Description**: Particle-based method that avoids grid limitations. Good for fluid dynamics.
- **Strengths**: No grid artifacts, natural fluid behavior, handles free surfaces well
- **Limitations**: Difficult to model high-strength materials (rock), requires small timesteps for stability

## Quick Start

Each implementation can be run independently:

```bash
# Cellular Automata
python ca/main.py

# Flux-based 
python flux/main.py

# SPH
python sph/main.py
```

## Requirements

All implementations require:
- Python 3.8+
- NumPy
- Pygame (for visualization)
- Numba (optional, for acceleration)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Documentation

Each implementation has its own documentation:
- `ca/PHYSICS.md` - CA physics implementation details
- `flux/PHYSICS.md` - Flux solver physics
- `sph/PHYSICS.md` - SPH method and equations

## Testing

Each implementation has its own test suite:
```bash
# Run CA tests
python -m pytest ca/tests/

# Run Flux tests  
python -m pytest flux/tests/

# Run SPH tests
python -m pytest sph/tests/
```

## Choosing an Implementation

- **For stability and simplicity**: Use CA
- **For accurate fluid dynamics**: Use Flux (within its stability limits)
- **For free-surface flows**: Use SPH

## Known Limitations

### CA
- Simplified physics model
- No true pressure solving
- Limited fluid spreading

### Flux
- Cannot handle extreme density ratios (space/rock)
- Numerical instabilities in low-density regions
- Limited to smaller scale simulations

### SPH
- High computational cost for solids
- Requires very small timesteps
- Particle disorder can affect accuracy

## Future Work

See [CONCLUSION_PLAN.md](CONCLUSION_PLAN.md) for planned improvements and research directions.

## License

This is a research project for exploring geological simulation methods.