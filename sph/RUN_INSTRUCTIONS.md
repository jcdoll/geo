# How to Run the SPH Simulation

## ✅ Installation Verified

All components are installed and working correctly:
- SPH physics engine ✓
- Numba acceleration ✓ 
- Interactive visualizer ✓

## Running the Simulation

### Option 1: With Display (Recommended)

```bash
source .venv/bin/activate
python main.py
```

**Requirements:**
- For WSL: Install an X server (e.g., VcXsrv, Xming) and set `export DISPLAY=:0`
- For Linux: Works out of the box
- For SSH: Use X11 forwarding (`ssh -X`)

### Option 2: Headless Testing

```bash
source .venv/bin/activate

# Performance test
python main_headless.py --steps 100 --backend numba

# Simple demo
python demo_sph.py
```

### Option 3: Test Everything Works

```bash
source .venv/bin/activate

# Test core systems
python test_viz_minimal.py

# Non-interactive test  
python test_main.py
```

## Common Issues

### "No display" Error
The visualizer requires a graphical display. Solutions:
1. **WSL**: Install VcXsrv and run `export DISPLAY=:0`
2. **SSH**: Connect with `ssh -X username@host`
3. **Headless**: Use `main_headless.py` instead

### Performance
- Use `--backend numba` for best performance
- Reduce particles with `--size 50` for faster testing
- Target FPS can be adjusted with `--fps 30`

## Example Commands

```bash
# Small fast simulation
python main.py --size 50 --backend numba

# Water scenario
python main.py --scenario water --backend numba

# Large volcanic planet
python main.py --scenario volcanic --size 200 --backend numba

# Benchmark performance
python main_headless.py --steps 100 --particles 5000 --backend numba
```

## Verification

Run this to verify everything works:
```bash
python test_viz_minimal.py
```

Output should show:
```
✓ Visualizer created successfully
✓ Physics step completed
✓ All core systems working!
```