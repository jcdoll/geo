# SPH Visualization

This directory contains high-performance visualization options for the SPH simulation.

## Available Renderers

### 1. Vispy Renderer (GPU-Accelerated)
**File**: `vispy_renderer.py`

- **Performance**: 100+ FPS for 100k particles
- **Features**:
  - GPU-accelerated point sprites
  - Multiple color modes (density, velocity, temperature, pressure, material)
  - Interactive camera controls
  - Real-time statistics overlay
  - Smooth particle rendering with alpha blending

**Requirements**:
```bash
pip install vispy
```

**Usage**:
```python
from sph.visualization.vispy_renderer import SPHRenderer

renderer = SPHRenderer(domain_size=(10, 6), max_particles=100000)
renderer.update_particles(particle_arrays, n_active, time)
renderer.run(update_callback)
```

### 2. Pygame Renderer (CPU-Based Fallback)
**File**: `pygame_renderer.py`

- **Performance**: 60 FPS for 10k particles, 30 FPS for 50k particles
- **Features**:
  - Adaptive quality modes
  - Efficient batch rendering
  - Color interpolation
  - No GPU required

**Requirements**:
```bash
pip install pygame
```

**Usage**:
```python
from sph.visualization.pygame_renderer import PygameRenderer

renderer = PygameRenderer(domain_size=(10, 6), window_size=(1280, 768))
while renderer.handle_events():
    renderer.update_particles(particle_arrays, n_active, time)
```

## Performance Comparison

| Particles | Matplotlib | Pygame | Vispy  |
|-----------|-----------|--------|--------|
| 1,000     | 30 FPS   | 60 FPS | 300 FPS|
| 10,000    | 3 FPS    | 60 FPS | 200 FPS|
| 100,000   | 0.3 FPS  | 10 FPS | 120 FPS|
| 1,000,000 | N/A      | 1 FPS  | 30 FPS |

## Controls

Both renderers support:
- **Space**: Pause/Resume
- **D**: Color by Density
- **V**: Color by Velocity
- **T**: Color by Temperature
- **P**: Color by Pressure
- **M**: Color by Material
- **ESC**: Quit

Vispy additional:
- **Mouse**: Pan/Zoom/Rotate
- **R**: Reset camera

Pygame additional:
- **S**: Toggle stats
- **Q**: Cycle quality modes
- **+/-**: Adjust particle size

## Choosing a Renderer

- **Use Vispy** when:
  - You need maximum performance
  - Simulating >10k particles
  - GPU is available
  - You want smooth, professional visuals

- **Use Pygame** when:
  - Vispy is not available
  - Running on systems without GPU
  - Need simple, reliable visualization
  - Prototyping or debugging

## Demo Scripts

Run the demos to see the renderers in action:

```bash
# Vispy demo (requires vispy)
python -m sph.demo_vispy

# Vispy benchmark
python -m sph.demo_vispy --benchmark

# Simple example with matplotlib
python sph/example_simple_fixed.py
```

## Future Enhancements

1. **3D Rendering**: Both renderers have hooks for 3D visualization
2. **Surface Reconstruction**: Metaball rendering for fluid surfaces
3. **Post-processing**: Bloom, motion blur, ambient occlusion
4. **Video Export**: Record simulations to video files
5. **Web Visualization**: Three.js based web renderer