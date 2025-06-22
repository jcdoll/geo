# PHYSICS REFERENCE - FLUX BASED EULERIAN

## Overview

This document describes a flux-based approach for simulating planetary geology that solves the fundamental issues with cellular automata (CA) approaches while maintaining performance targets of 30+ FPS for a 128×128 grid.

## GOLDEN RULES

- Review AGENTS.md before starting work.

- Do not add artificial limits, e.g. minimum or maximum temperatures, or minimum or maximum temperature changes per step. These are artificial limits that can obscure bugs in the code or true physical phenomena. The only exception is that temperature can not go below absolute zero (0 K). If it does that indicates a problem with the model.

- Do not keep legacy code or legacy interfaces to maintain compatibility. The code for this project is self-contained. There are no external callers of this code.

- Use a dispatcher design pattern to select between multiple physics implementation options.

- When in doubt, add traceback logging so that any error sources are correctly identified.

## Theory

### State Variables

The system state consists of continuous fields defined on a fixed Eulerian grid:

```
ρ(x,y,t)    - Total density [kg/m³]
vx(x,y,t)   - x-velocity [m/s]
vy(x,y,t)   - y-velocity [m/s]
T(x,y,t)    - Temperature [K]
P(x,y,t)    - Pressure [Pa]
φᵢ(x,y,t)   - Volume fraction of material i (Σφᵢ = 1)
```

### Governing Equations

#### 1. Mass Conservation (Continuity Equation)
```
∂ρ/∂t + ∇·(ρv) = 0
```

In flux form:
```
∂ρ/∂t = -[∂(ρvx)/∂x + ∂(ρvy)/∂y]
```

#### 2. Momentum Conservation
```
∂(ρv)/∂t + ∇·(ρvv) = -∇P + ρg
```

Where g includes both self-gravity and any external gravity fields.

#### 3. Self-Gravity
The gravitational potential Φ satisfies Poisson's equation:
```
∇²Φ = 4πGρ
```

With gravitational acceleration:
```
g = -∇Φ
```

#### 4. Pressure
For hydrostatic equilibrium, pressure satisfies:
```
∇P = ρg
```

Taking the divergence:
```
∇²P = ∇·(ρg) = ρ(∇·g) + g·(∇ρ)
```

For self-gravity: ∇·g = -4πGρ, giving:
```
∇²P = -4πGρ² + g·(∇ρ)
```

#### 5. Heat Transfer
Energy conservation with thermal diffusion:
```
∂(ρcₚT)/∂t + ∇·(ρcₚTv) = ∇·(k∇T) + Q
```

Where:
- cₚ = specific heat capacity
- k = thermal conductivity
- Q = heat sources (radioactive decay, etc.)

#### 6. Multi-Material Transport
Each material's volume fraction evolves by:
```
∂φᵢ/∂t + ∇·(φᵢv) = 0
```

With constraint: Σφᵢ = 1

#### 7. Solar Heating
Solar radiation provides the primary external energy input:
```
Q_solar = I₀ × (1 - albedo) × absorption_coefficient
```

Where:
- I₀ = incident solar flux (W/m²)
- albedo = material reflectance (0-1)
- absorption_coefficient = material-dependent absorption

#### 8. Radiative Cooling
All materials emit thermal radiation according to Stefan-Boltzmann law:
```
Q_radiative = -ε σ (T⁴ - T_space⁴)
```

Where:
- ε = thermal emissivity (material property)
- σ = 5.67×10⁻⁸ W/(m²·K⁴) (Stefan-Boltzmann constant)
- T = material temperature (K)
- T_space = 2.7 K (cosmic background)

#### 9. Material Phase Transitions
Materials transform based on temperature and pressure conditions:
```
Rate = f(T, P) × φ_source
```

Transitions include:
- Water system: ice ↔ water ↔ vapor
- Crystallization: magma → rock (cooling)
- Weathering: rock → sand (chemical and physical processes)

Weathering combines chemical and physical processes:
```
Chemical rate = exp((T - 288)/14.4) × water_factor
Physical rate = freeze_thaw_cycles × thermal_stress
```
Where water_factor = 3.0 if water is present

### Mixture Properties
For cells containing multiple materials:
```
ρ = Σ(φᵢρᵢ)                    # Volume-weighted density
k = 1/Σ(φᵢ/kᵢ)                 # Harmonic mean conductivity
cₚ = Σ(φᵢcₚᵢ)                  # Volume-weighted heat capacity
```

## Implementation

### Discretization

We use a finite volume approach on a regular grid with cell-centered variables and face-centered fluxes.

```
Cell (i,j):
- Center: stores ρ, v, T, P, φ
- Faces: compute fluxes F

Grid spacing: Δx = Δy = dx
Time step: Δt (CFL limited)
```

### Simulation

We use operator splitting for clarity and modularity:

```python
def flux_based_simulation():
    # Initialize
    state = create_initial_state()
    solar_angle = 0.0
    
    while running:
        # 1. Compute timestep (CFL condition)
        dt = timestep(state)

        # 2. Simulate timestep
        timestep(state, dt)
        
        # 3. Render
        render(state)
        
        # 4. Update other state variables
        solar_angle += dt * SOLAR_ROTATION_RATE # day/night cycle
```

```python
def timestep(state, dt):
    # 1. Self-gravity (existing multigrid solver)
    gx, gy = solve_gravity_multigrid(state.density)
    
    # 2. Pressure (existing multigrid solver)
    pressure = solve_pressure_multigrid(state.density, gx, gy)
    
    # 3. Momentum update (pressure gradients + gravity)
    update_momentum(state, pressure, gx, gy, dt)
    
    # 4. Advection (flux-based transport)
    advect_materials_flux(state, dt)
    
    # 5. Thermal diffusion (flux-based)
    diffuse_heat_flux(state, dt)
    
    # 6. Solar heating
    apply_solar_heating(state, solar_angle)
    
    # 7. Radiative cooling
    apply_radiative_cooling(state)
    
    # 8. Phase transitions
    apply_phase_changes(state)
```

### Flux Computation

The key innovation is computing fluxes at cell faces:

```python
def compute_mass_flux(density, velocity, dt, dx):
    """Compute mass flux through cell faces"""
    
    # X-direction flux at face (i+1/2, j)
    flux_x = zeros(nx+1, ny)
    
    for i in range(1, nx):
        for j in range(ny):
            # Velocity at face
            v_face = 0.5 * (velocity[i,j] + velocity[i-1,j])
            
            # Upwind scheme for stability
            if v_face > 0:
                # Flow from left cell
                flux_x[i,j] = density[i-1,j] * v_face * dt/dx
            else:
                # Flow from right cell
                flux_x[i,j] = density[i,j] * v_face * dt/dx
    
    return flux_x
```

### Material Transport

Each material is advected with the flow:

```python
def advect_materials_flux(state, dt):
    """Transport all materials using flux form"""
    
    # For each material
    for mat in range(n_materials):
        # Compute fluxes
        flux_x = compute_material_flux_x(
            state.vol_frac[mat], 
            state.velocity_x, 
            dt, dx
        )
        flux_y = compute_material_flux_y(
            state.vol_frac[mat], 
            state.velocity_y, 
            dt, dx
        )
        
        # Update using flux divergence
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                state.vol_frac[mat][i,j] += (
                    flux_x[i,j] - flux_x[i+1,j] +
                    flux_y[i,j] - flux_y[i,j+1]
                )
    
    # Ensure constraints
    normalize_volume_fractions(state.vol_frac)
    update_mixture_density(state)
```

### Pressure Solver Integration

The existing multigrid solver with red-black smoothing remains unchanged:

```python
def solve_pressure_multigrid(density, gx, gy):
    """Existing multigrid Poisson solver"""
    
    # Build RHS: ∇²P = -4πGρ² + g·∇ρ
    rhs = build_pressure_rhs(density, gx, gy)
    
    # V-cycle with red-black smoothing
    pressure = multigrid_vcycle(rhs, n_levels=4)
    
    # Boundary conditions
    pressure[density < vacuum_threshold] = 0
    
    return pressure
```

### Momentum Update

Forces are applied consistently:

```python
def update_momentum(state, pressure, gx, gy, dt):
    """Update velocities from forces"""
    
    # Pressure gradient (using same stencil as Poisson solver!)
    dpdx = gradient_x(pressure)
    dpdy = gradient_y(pressure)
    
    # Update velocities
    state.vx += dt * (-dpdx/state.density + gx)
    state.vy += dt * (-dpdy/state.density + gy)
    
    # Apply viscosity for stability
    apply_viscous_damping(state, dt)
```

### Solar Heating Implementation

The DDA (Digital Differential Analyzer) ray-marching algorithm efficiently traces solar rays:

```python
def apply_solar_heating(state, solar_angle):
    """Apply solar heating using ray marching"""
    
    # Solar direction vector
    ux = cos(solar_angle)
    uy = sin(solar_angle)
    
    # Spawn rays from boundary
    for ray in spawn_rays_from_boundary(ux, uy):
        intensity = SOLAR_CONSTANT
        
        # March ray through grid
        for cell in dda_march(ray, ux, uy):
            # Compute effective absorption from mixture
            absorption = compute_mixture_absorption(state.vol_frac[cell])
            
            # Deposit energy
            absorbed = intensity * absorption
            state.heat_source[cell] += absorbed / cell_volume
            
            # Attenuate ray
            intensity *= (1 - absorption)
            
            # Stop if opaque
            if absorption >= 0.99:
                break
```

Material absorption coefficients:
- AIR: 0.001 (nearly transparent)
- WATER_VAPOR: 0.005 (slight absorption)
- WATER: 0.02 (moderate absorption)
- ICE: 0.01 (semi-transparent)
- Rocks/solids: 1.0 (opaque)

### Radiative Cooling Implementation

Radiative cooling with greenhouse effect:

```python
def apply_radiative_cooling(state):
    """Apply Stefan-Boltzmann cooling with greenhouse effect"""
    
    for i in range(nx):
        for j in range(ny):
            # Skip space cells
            if state.density[i,j] < vacuum_threshold:
                continue
                
            # Compute effective emissivity from mixture
            emissivity = compute_mixture_emissivity(state.vol_frac[:,i,j])
            
            # Stefan-Boltzmann law
            cooling = emissivity * STEFAN_BOLTZMANN * (
                state.temperature[i,j]**4 - T_SPACE**4
            )
            
            # Greenhouse effect from water vapor column
            vapor_column = compute_vapor_column_above(state, i, j)
            greenhouse_factor = 0.1 + 0.5 * tanh(log(1 + vapor_column/1.0) / 10)
            
            # Apply cooling
            state.heat_source[i,j] -= cooling * (1 - greenhouse_factor)
```

### Atmospheric Processes

The atmosphere requires special handling for realistic behavior:

```python
def enhance_atmospheric_diffusion(state):
    """Apply enhanced mixing to atmospheric cells"""
    
    # Identify atmospheric cells (air and water vapor)
    atmos_mask = (state.vol_frac[AIR] > 0.5) | (state.vol_frac[WATER_VAPOR] > 0.5)
    
    # Enhanced diffusion coefficient (3x for turbulence)
    state.thermal_diffusivity[atmos_mask] *= 3.0
    state.momentum_diffusivity[atmos_mask] *= 3.0
```

Greenhouse effect calculation:
```python
def compute_greenhouse_factor(vapor_column):
    """Compute greenhouse trapping from water vapor"""
    base = 0.1      # Minimum greenhouse effect
    max_factor = 0.6 # Maximum greenhouse effect
    scale = 1.0      # kg/m² scaling factor
    
    # Smooth transition using tanh
    return base + (max_factor - base) * np.tanh(np.log(1 + vapor_column/scale) / 10)
```

## Key Advantages Over CA Approach

### 1. Continuous Mass Transport

**CA Problem**: Cell swapping is binary - either swap or don't. This creates:
- Quantization noise in flow
- Unclear criteria for when to swap
- Mass not conserved (different materials have different densities)
- Difficulty achieving hydrostatic equilibrium

**Flux Solution**: Transport arbitrary amounts of mass:
```python
# CA approach (problematic)
if should_swap(cell1, cell2):
    swap(cell1, cell2)  # All or nothing!

# Flux approach (correct)
mass_to_move = flux * area * dt  # Continuous amount
cell1.mass -= mass_to_move
cell2.mass += mass_to_move
```

### 2. Consistent Pressure-Velocity Coupling

**CA Problem**: Pressure solver and movement rules use different discretizations:
- Pressure solver assumes continuous density field
- Movement rules operate on discrete cells
- Result: Forces don't balance at equilibrium

**Flux Solution**: Same discretization throughout:
- Pressure solver uses cell-centered values
- Flux computation uses same grid
- Forces exactly balance at equilibrium

### 3. Exact Conservation

The flux formulation guarantees:
```
Mass leaving cell (i,j) = Mass entering neighbor cells
```

This is exact to machine precision, unlike CA swapping.

### 4. Natural Multi-Material Handling

**CA**: Each cell has one material - interfaces are stepped

**Flux**: Each cell can contain multiple materials with smooth transitions:
```python
# Cell can be 70% rock, 30% water
# Leads to realistic mixing and interfaces
```

## Performance Optimizations

### Numba Acceleration
```python
@numba.njit(parallel=True, fastmath=True)
def fast_flux_kernel(density, velocity, dt, dx):
    # 5-10x speedup from JIT compilation
    # Parallel loops for multi-core
    # fastmath for SIMD vectorization
```

### Sparse Updates
```python
# Only compute fluxes where velocity > threshold
active_mask = (abs(velocity) > 1e-6) | (density > vacuum_threshold)
```

### Memory Layout
```python
# Structure for cache efficiency
state = {
    'density': np.zeros((nx, ny), dtype=np.float32),
    'vol_frac': np.zeros((n_materials, nx, ny), dtype=np.float32)
}
```

## Expected Performance

For 128×128 grid with 9 materials:
- Gravity solve: 15ms (existing)
- Pressure solve: 15ms (existing)
- Material flux: 5ms (new)
- Momentum update: 2ms
- Thermal diffusion: 3ms
- Solar heating: 2ms
- Radiative cooling: 2ms
- Phase transitions: 3ms
- **Total: ~47ms = 21 FPS**

With Numba optimization:
- **Total: ~30ms = 33 FPS** ✓

## Materials System

### Materials Overview

The materials in the simulation are:

- space
- air
- water
- water vapor
- ice
- rock
- sand
- uranium
- magma

Phase transitions are:

- ice <-> water <-> water vapor
- rock + water -> sand
- sand + heat + pressure -> rock
- magma + cooling -> rock
- uranium (no transitions)
- air (no transitions)

Some noteable behavior:
- water vapor has lower density than air, so water evaporates, rises, cooles, and then falls back to the surface as water or ice
- ice has lower density than water and so floats
- rock and sand both have higher density than water and so sink
- uranium has the highest density and 
- we do not currently transition from rock back to magma

### Material Properties

| Material    | Density (kg/m³) | Viscosity | Albedo | Thermal Conductivity (W/m·K) | Heat Capacity (J/kg·K) | Emissivity | Heat Generation (W/m³) |
|-------------|-----------------|-----------|--------|------------------------------|------------------------|------------|------------------------|
| SPACE       | 0               | 0.0       | 0.0    | 0.0                          | 0                      | 0.0        | 0                      |
| AIR         | 1.2             | 0.005     | 0.0    | 0.025                        | 1005                   | 0.8        | 0                      |
| WATER       | 1000            | 0.01      | 0.06   | 0.6                          | 4186                   | 0.96       | 0                      |
| WATER_VAPOR | 0.6             | 0.005     | 0.0    | 0.02                         | 2080                   | 0.8        | 0                      |
| ICE         | 917             | 0.15      | 0.8    | 2.2                          | 2100                   | 0.97       | 0                      |
| ROCK        | 2700            | 0.35      | 0.3    | 3.0                          | 1000                   | 0.95       | 0                      |
| SAND        | 1600            | 0.1       | 0.4    | 0.3                          | 830                    | 0.95       | 0                      |
| URANIUM     | 19000           | 0.4       | 0.15   | 27.0                         | 116                    | 0.9        | 5×10⁻⁴                 |
| MAGMA       | 2700            | 0.05      | 0.9    | 1.5                          | 1200                   | 0.95       | 0                      |

Notes:
- Viscosity: 0 = no resistance (space), 1 = no flow (theoretical solid)
- Albedo: 0 = perfect absorber, 1 = perfect reflector
- Heat generation: Only uranium has radioactive decay heating

### Phase Transitions

| From ↓ To →   | SPACE | AIR | WATER    | WATER_VAPOR | ICE      | ROCK                 | SAND        | URANIUM | MAGMA     |
|---------------|-------|-----|----------|-------------|----------|----------------------|-------------|---------|-----------|
| SPACE         | -     | -   | -        | -           | -        | -                    | -           | -       | -         |
| AIR           | -     | -   | -        | -           | -        | -                    | -           | -       | -         |
| WATER         | -     | -   | -        | T > 373K    | T < 273K | -                    | -           | -       | -         |
| WATER_VAPOR   | -     | -   | T < 373K | -           | -        | -                    | -           | -       | -         |
| ICE           | -     | -   | T > 273K | -           | -        | -                    | -           | -       | -         |
| ROCK          | -     | -   | -        | -           | -        | -                    | Weathering* | -       | Future    |
| SAND          | -     | -   | -        | -           | -        | P > 10 MPa, T > 673K | -           | -       | -         |
| URANIUM       | -     | -   | -        | -           | -        | -                    | -           | -       | -         |
| MAGMA         | -     | -   | -        | -           | -        | T < 1273K            | -           | -       | -         |

*Weathering conditions:
- Chemical: exp((T-288)/14.4) × water_factor, where water_factor = 3.0 if adjacent to water
- Physical: Freeze-thaw cycles near 273K

Transition notes:
- Water evaporation/condensation depends on both temperature and pressure
- Rock melting temperature varies by composition (granite ~1473K, basalt ~1673K)
- Magma crystallization produces different rocks based on cooling rate
- Sand lithification requires both pressure and temperature over geological time
- Uranium has no transitions (stable for simulation purposes)

### Phase Transitions Implementation

Phase transitions modify volume fractions in place based on temperature and pressure:

```python
def apply_phase_transitions(state):
    """Vectorized material phase transitions based on T-P conditions"""
    
    T = state.temperature
    P = state.pressure
    dt = state.dt
    
    # Water -> Ice (freezing at T < 273K)
    freeze_mask = (T < 273) & (state.vol_frac[WATER] > 0)
    freeze_rate = np.where(freeze_mask, 0.1 * dt, 0.0)  # 10% per second
    freeze_amount = state.vol_frac[WATER] * np.minimum(freeze_rate, 1.0)
    
    state.vol_frac[WATER] -= freeze_amount
    state.vol_frac[ICE] += freeze_amount
    state.heat_source += freeze_amount * 3.34e5  # Latent heat of fusion
    
    # Water -> Water Vapor (evaporation at T > 373K)
    evap_mask = (T > 373) & (state.vol_frac[WATER] > 0)
    evap_rate = np.where(evap_mask, 0.05 * dt, 0.0)
    evap_amount = state.vol_frac[WATER] * np.minimum(evap_rate, 1.0)
    
    state.vol_frac[WATER] -= evap_amount
    state.vol_frac[WATER_VAPOR] += evap_amount
    state.heat_source -= evap_amount * 2.26e6  # Latent heat of vaporization
    
    # ... similar patterns for all other transitions ...
    # Key concept: Use boolean masks and vectorized operations
    # to process entire grid at once, avoiding nested loops
```

## Visualization and Interaction

### Display Modes

The visualizer supports multiple rendering modes to inspect different aspects of the simulation:

#### Material Visualization
- Dominant Material: Shows the material with highest volume fraction per cell
- Individual Layers: View volume fraction of specific materials (0-1 grayscale)
- Composite Blend: RGB mixing based on material fractions for smooth interfaces

#### Physical Field Visualization
- Temperature: Color gradient from blue (cold) to red (hot)
- Pressure: Contour lines or gradient display
- Velocity: Vector arrows showing flow direction and magnitude
- Gravity: Vector field display showing gravitational acceleration
- Power Density: Heat generation/absorption visualization

### Interactive Tools

#### Material Manipulation
```python
# Add material at cursor position
def add_material(state, x, y, material_type, amount, radius):
    mask = distance_from(x, y) < radius
    available_space = 1.0 - state.vol_frac.sum(axis=0)
    add_amount = np.minimum(amount, available_space) * mask
    state.vol_frac[material_type] += add_amount

# Remove material (replace with space)
def remove_material(state, x, y, radius):
    mask = distance_from(x, y) < radius
    state.vol_frac[:, mask] = 0.0  # All materials to zero
```

#### Physics Tools
- Heat Source: Add/remove thermal energy at cursor
- Pressure Tool: Apply positive/negative pressure
- Velocity Injection: Set velocity vectors in regions

### Controls

| Key | Action |
|-----|--------|
| SPACE | Play/Pause simulation |
| ← → | Step backward/forward |
| R | Reset simulation |
| 1-9 | Select material type |
| M | Cycle display mode |
| S | Save screenshot |
| H | Show help |
| ESC | Exit |

| Mouse | Action |
|-------|--------|
| Left Click | Apply selected tool (add) |
| Shift + Left Click | Apply selected tool (remove) |
| Scroll | Adjust tool radius |
| Shift+Scroll | Adjust tool intensity |
| Right Click | Select cell to show material info |

### Information Display

Real-time overlay showing:
- Performance: FPS, ms/frame, optimization status
- Simulation Time: Current time, timestep size
- Material Inventory: Total volume of each material type
- Conservation Metrics: 
  - Total mass: Σ(ρ × φ × cell_volume)
  - Total energy: Σ(ρ × cₚ × T × φ × cell_volume)
- Cursor Info: Material fractions, T, P, v at mouse position

### Implementation Notes

The visualizer uses efficient rendering:
```python
# Composite material colors
def render_materials(vol_frac, material_colors):
    # vol_frac shape: [n_materials, ny, nx]
    # material_colors shape: [n_materials, 3] (RGB)
    rgb_image = np.einsum('myx,mc->yxc', vol_frac, material_colors)
    return np.clip(rgb_image, 0, 1)
```

## Summary

The flux-based approach solves the fundamental issues with CA simulation:
1. **Continuous transport** instead of discrete swapping
2. **Exact conservation** of mass and energy
3. **Correct equilibrium** states (pressure balances gravity)
4. **Natural multi-material** support
5. **Fast enough** for real-time simulation (30+ FPS)

The key insight: By allowing arbitrary amounts of mass to flow between cells (rather than swapping whole cells), we get smooth, physically correct behavior while maintaining the simplicity and performance of a grid-based method.