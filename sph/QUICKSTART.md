# SPH Geological Simulation - Quick Start

## Installation Status

✅ **All components are installed and working!**
- NumPy-based CPU backend 
- Numba JIT acceleration (26.9 FPS for 1467 particles)
- Interactive visualizer with full toolbar UI

## Running the Simulation

### Interactive Mode (requires display)
```bash
# Activate environment
source .venv/bin/activate

# Run with default settings
python main.py

# Run with Numba acceleration (recommended)
python main.py --backend numba

# Try different scenarios
python main.py --scenario water --size 50
python main.py --scenario volcanic --size 100 --backend numba
```

### Headless Mode (no display required)
```bash
# Test performance
python main_headless.py --steps 100 --backend numba

# Run non-interactive demo
python demo_sph.py
```

## Controls

When running `python main.py`:

- **SPACE** - Pause/Resume
- **TAB/M** - Cycle display modes (Material, Temperature, Pressure, etc.)
- **T** - Cycle tools
- **1-9** - Select materials
- **H** - Show help
- **Click toolbar** - Toggle physics modules, select materials, change display

## Performance

With Numba backend:
- ~27 FPS for 1,467 particles
- ~100 FPS for 369 particles  

## Known Issues

- GPU backend requires proper CUDA installation (not available in WSL)
- Visualizer requires X11 display (use X server for WSL)

## Next Steps

1. For headless testing: `python main_headless.py`
2. For interactive use: Set up X11 forwarding or use native Linux/Windows
3. Try different scenarios and physics settings via the toolbar