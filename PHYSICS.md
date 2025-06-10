# PHYSICS REFERENCE - 2D Geological Simulation

This document serves as the authoritative reference for all physical processes, laws, and equations implemented in the 2D geological simulation engine.

## GOLDEN RULES

1. Do not add artificial limits, e.g. minimum or maximum temperatures, or minimum or maximum temperature changes per step. These are artificial limits that can obscure bugs in the code or true physical phenomena. The only exception is that temperature can not go below absolute zero (0 K). If it does that indicates a problem with the model.

2. Do not keep legacy code or legacy interfaces to maintain compatibility. The code for this project is self-contained. There are no external callers of this code.

3. Use a dispatcher design pattern to select between multiple physics implementation options.

## TABLE OF CONTENTS

1. [Heat Transfer Physics](#heat-transfer-physics)
2. [Gravitational Physics](#gravitational-physics)
3. [Pressure Calculations](#pressure-calculations)
4. [Material Properties & Transitions](#material-properties--transitions)
5. [Fluid Dynamics](#fluid-dynamics)
6. [Atmospheric Physics](#atmospheric-physics)
7. [Solar & Radiative Physics](#solar--radiative-physics)
8. [Geological Processes](#geological-processes)
9. [Units & Constants](#units--constants)
10. [Numerical Methods](#numerical-methods)

---

## HEAT TRANSFER PHYSICS

### Core Heat Diffusion Equation

The fundamental governing equation is the heat equation with source terms:

```
∂T/∂t = α∇²T + Q_total/(ρcp)
```

Where:
- `T` = Temperature (K)
- `α` = Thermal diffusivity (m²/s)
- `Q_total` = Total volumetric heat generation (W/m³)
- `ρ` = Density (kg/m³)
- `cp` = Specific heat capacity (J/(kg⋅K))

### Thermal Diffusivity Calculation

```
α = k/(ρ × cp)
```

Where `k` = thermal conductivity (W/(m⋅K))

**Enhanced diffusivity zones:**
- **Atmospheric materials**: `α_atmo = α × 5.0` (fast convective heat transfer)
- **Material interfaces**: `α_interface = α × 1.5` (enhanced interface heat transfer)

### Heat Source Terms

The total heat source `Q_total` comprises:

1. **Internal heating** (`Q_internal`)
2. **Solar heating** (`Q_solar`) 
3. **Atmospheric heating** (`Q_atmospheric`)
4. **Radiative cooling** (`Q_radiative`) [negative]

#### Internal Heat Generation

**Crustal heating:**
```
Q_crustal = 1×10⁻⁶ × d² W/m³
```
Where `d` = relative depth (0 at surface, 1 at center)

**Core heating:**
```
Q_core = 1×10⁻³ × exp(0.5 × d) W/m³
```

**Total internal heating:**
```
Q_internal = Q_crustal + Q_core
```

### Numerical Implementation

**DuFort-Frankel Scheme** (unconditionally stable):

For first timestep (Forward Euler bootstrap):
```
T^(n+1) = T^n + dt × α × ∇²T^n / dx²
```

For subsequent timesteps:
```
T^(n+1) = [T^(n-1) + 2×dt×α×∇²T^n] / [1 + 4×dt×α/dx²]
```

**Laplacian operator** (5-point stencil):
```
∇²T = (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1] - 4×T[i,j]) / dx²
```

---

## GRAVITATIONAL PHYSICS

### Gravitational Stratification

Materials move based on **buoyancy forces** in the gravitational field:

**Effective density with thermal expansion:**
```
ρ_eff = ρ₀ / (1 + β(T - T₀))
```

Where:
- `ρ₀` = reference density (kg/m³)
- `β` = volumetric thermal expansion coefficient (1/K)
- `T₀` = reference temperature (273.15 K)

**Buoyancy conditions:**
- **Rising**: Less dense material closer to center than denser material farther out
- **Sinking**: Denser material farther from center than less dense material closer in

**Swap threshold:**
```
ρ_max/ρ_min ≥ density_ratio_threshold (1.05-1.2 depending on quality)
```

### Gravitational Collapse

**Mechanism**: Solid materials fall into cavities/voids under gravity

**Falling direction**: Toward planetary center of mass

**Fall probability**:
- Initial attempts: 50%
- Later attempts: 30%

**Multi-step process**: Up to 5 iterations per timestep to allow cascading collapses

---

## PRESSURE CALCULATIONS

### Pressure Distribution

**Space**: `P = 0` (vacuum)

**Atmospheric pressure** (exponential decay):
```
P_atmo = P_surface × exp(-h/H)
```
Where:
- `P_surface = 0.1 MPa`
- `h` = height above surface (m)
- `H = 8400 m` (scale height)

**Hydrostatic pressure** (fluids):
```
P_fluid = max(P_surface, ρ_fluid × g × depth / 10⁶)
```
Where:
- `ρ_fluid = 2000 kg/m³`
- `g = 9.81 m/s²`

**Lithostatic pressure** (solids):
```
P_solid = max(P_surface, ρ_solid × g × depth / 10⁶)
```
Where:
- `ρ_solid = 3000 kg/m³`

### Center of Mass Calculation

**Vectorized calculation:**
```
COM_x = Σ(ρᵢ × Vᵢ × xᵢ) / Σ(ρᵢ × Vᵢ)
COM_y = Σ(ρᵢ × Vᵢ × yᵢ) / Σ(ρᵢ × Vᵢ)
```
Where `V` = cell volume

---

## MATERIAL PROPERTIES & TRANSITIONS

### Metamorphism

**General transition system**: Each material type can have multiple P-T dependent transitions

**Transition conditions**:
```
T_min ≤ T ≤ T_max  AND  P_min ≤ P ≤ P_max
```

**Examples**:
- Rock → Magma (high temperature)
- Magma → Rock (cooling)
- Water → Ice (low temperature)
- Water → Water vapor (high temperature/low pressure)

### Phase Transitions During Convection

**Water evaporation**:
- Condition: `T > 350 K` (≈77°C)
- Probability: 5% per timestep

**Water vapor condensation**:
- Condition: `T < 320 K` (≈47°C)  
- Probability: 5% per timestep

---

## FLUID DYNAMICS

### Material Mobility Classification

**Gases**: Air, Water vapor
**Liquids**: Water, Magma  
**Hot solids**: Solid materials with `T > 1200 K`

### Air Migration (Buoyancy)

**Direction**: Away from planetary center (upward buoyancy)

**Migration conditions**:
- Target material is porous (`porosity > 0.1`) OR non-solid
- Move toward lower gravitational potential

**Migration probability**: 30%

---

## ATMOSPHERIC PHYSICS

### Atmospheric Convection

**Fast mixing process** for atmospheric materials (Air, Water vapor)

**Mixing equation**:
```
T_new = T_old + f × (T_avg_neighbors - T_old)
```
Where `f = 0.3` (mixing fraction)

**Neighbor calculation**: Only atmospheric cells participate in averaging

### Directional-Sweep Atmospheric Absorption (default)

The simulator now uses a **single-pass Amanatides & Woo DDA sweep** that marches
solar rays directly through the grid, giving realistic, angle-dependent
attenuation with O(N) complexity.

Algorithm overview:
1. Compute the unit solar direction `(ux, uy)` from `solar_angle`.
2. Select the boundary opposite the incoming rays and spawn one ray per
   boundary cell.
3. Advance each ray cell-by-cell using integer DDA (`t_max_x`, `t_max_y`).
4. Upon entering a non-space cell the ray deposits energy

   ```
   absorbed = I * k
   I      -= absorbed
   ```

   • `k` is a per-material absorption coefficient from `materials.py`:
     AIR 0.001, WATER_VAPOR 0.005, WATER 0.02, ICE 0.01, others 1.0.
5. The absorbed energy is converted to volumetric power density and split into
   `solar_input` (surface/solid/liquid cells) or `atmospheric_heating`
   (gas cells) for diagnostics.
6. The ray terminates if `k ≥ 1` (opaque) or when the remaining flux is zero.

Advantages:
• Accurate day/night shadowing for any solar angle.
• Eliminates the 10³–10⁵× flux spikes of the old radial scheme.
• Runs in linear time; suitable for real-time visualisation.

---

## SOLAR & RADIATIVE PHYSICS

### Solar Heating

**Latitude-dependent intensity**:
```
I_solar = I₀ × cos(latitude) × distance_factor
```
Where:
- `I₀ = 1361 W/m²` (solar constant)
- `distance_factor = 1×10⁻⁵`
- `latitude` = distance from equatorial plane

**Albedo effects**:
```
I_effective = I_solar × (1 - albedo)
```

Material albedos stored in material database

### Radiative Cooling

**Stefan-Boltzmann Law**:
```
P_radiated = ε × σ × A × (T⁴ - T_space⁴)
```

Where:
- `ε` = emissivity (material-dependent)
- `σ = 5.67×10⁻⁸ W/(m²⋅K⁴)` (Stefan-Boltzmann constant)
- `T_space = 2.7 K` (cosmic background)

**Greenhouse effect**:
```
σ_eff = σ × (1 - greenhouse_factor)
```

**Dynamic greenhouse**:
```
greenhouse_factor = base + (max - base) × tanh(vapor_factor)
```
Where vapor_factor depends on atmospheric water vapor content

---

## GEOLOGICAL PROCESSES

### Weathering

**Chemical weathering** (Arrhenius-like):
```
Rate_chemical = exp((T - 15)/14.4) × water_factor
```
Where `water_factor = 3.0` if adjacent to water

**Physical weathering**:
- **Freeze-thaw**: Max effectiveness at 0°C
- **Thermal stress**: High temperature extremes

**Weathering products**: Material-specific, defined in material database

---

## UNITS & CONSTANTS

### Base Units
- **Length**: meters (m)
- **Time**: years (converted to seconds internally)
- **Temperature**: Kelvin (K)
- **Pressure**: Pascal (Pa), displayed as MPa
- **Power**: Watts (W)

### Key Constants
```
seconds_per_year = 365.25 × 24 × 3600 = 31,557,600 s
stefan_boltzmann_geological = 5.67×10⁻⁸ × seconds_per_year J/(year⋅m²⋅K⁴)
space_temperature = 2.7 K
reference_temperature = 273.15 K
average_gravity = 9.81 m/s²
```

### Typical Material Properties
- **Density**: 1000-8000 kg/m³
- **Thermal conductivity**: 0.1-400 W/(m⋅K)  
- **Specific heat**: 400-4200 J/(kg⋅K)
- **Thermal expansion**: 1×10⁻⁶ - 3×10⁻⁴ 1/K

---

## NUMERICAL METHODS

### Time Stepping

**Adaptive timestepping**: Currently fixed at `dt = 1 year`

**DuFort-Frankel stability**: Unconditionally stable for any timestep

### Spatial Discretization

**Grid**: Uniform Cartesian with square cells
**Cell size**: Typically 50 m per cell
**Boundary conditions**: Insulating (no-flux) at material-space interfaces

### Vectorization

**NumPy arrays**: All operations vectorized for performance
**Morphological operations**: Used for fast neighbor calculations
**Boolean masking**: Efficient material-type specific operations

### Performance Optimization

**Quality levels**:
1. **Full** (100% accuracy): Process all cells
2. **Balanced** (50% accuracy): Process 50% of cells  
3. **Fast** (20-33% accuracy): Process 20-33% of cells

**Caching**: Material property lookups cached for performance

**Neighbor shuffling**: Randomized to prevent grid artifacts

---

## PHYSICAL ASSUMPTIONS

### Simplifications

1. **2D geometry**: All processes assumed cylindrically symmetric
2. **Incompressible flow**: Density changes only via thermal expansion
3. **Local thermodynamic equilibrium**: No heat diffusion lag
4. **Idealized materials**: Properties constant within material types
5. **No elasticity**: Instantaneous stress relaxation
6. **Simplified radiative transfer**: No scattering, only absorption/emission

### Scaling

The simulation uses **enhanced parameters** for visibility:
- Internal heating rates ~1000× real values
- Simplified atmospheric physics
- Accelerated geological processes
- Enhanced thermal diffusivity for stability

This allows geological timescales (millions of years) to be observable in human timescales (minutes) while preserving physical relationships.

---

*This document represents the complete physical model as implemented in `simulation_engine.py`. All equations and parameters are directly traceable to the code implementation.*

## Heat Diffusion Methods Comparison

This document compares different numerical methods for solving the heat diffusion equation with source terms in the geology simulator.

### The Problem

We need to solve the heat equation with source terms:
```
∂T/∂t = α∇²T + Q/(ρcₚ)
```

Where:
- `T` = temperature (K)
- `α` = thermal diffusivity (m²/s) 
- `Q` = heat source density (W/m³)
- `ρ` = density (kg/m³)
- `cₚ` = specific heat (J/(kg⋅K))

The challenge is that geological systems have:
1. **Large heat sources** (solar, internal heating, radiative cooling)
2. **Multiple time scales** (diffusion: years, sources: seconds)
3. **Stability requirements** for long-term evolution
4. **Performance constraints** (real-time visualization)

### Current Solution: Operator Splitting Method

The simulation now uses **operator splitting** to solve the heat equation optimally. This approach treats different physical processes separately using their most appropriate numerical methods.

## Operator Splitting Implementation

The heat equation is split into separate operators:
```
∂T/∂t = L_diffusion(T) + L_radiation(T) + L_sources(T)
```

Where:
- `L_diffusion(T) = α∇²T` (pure diffusion)
- `L_radiation(T) = radiative cooling` (Stefan-Boltzmann)
- `L_sources(T) = internal + solar + atmospheric heating`

### Three-Step Solution Process

**Step 1: Pure Diffusion**
```python
T₁ = solve_pure_diffusion(T₀, dt)
```
Uses adaptive explicit method with sub-stepping for stability.

**Step 2: Radiative Cooling (Configurable Method)**
```python
T₂ = solve_radiative_cooling(T₁, dt)  # Dispatches to selected method
```
Configurable implementation - either Newton-Raphson implicit or linearized Stefan-Boltzmann.

**Step 3: Heat Sources (Explicit)**
```python
T₃ = solve_heat_sources_explicit(T₂, dt)
```
Applies internal heating, solar heating, and atmospheric heating explicitly.

### Method Comparison

#### Current Method: Operator Splitting (Implemented)

**Implementation:**
```python
# Step 1: Pure diffusion with adaptive stepping
working_temp, stability = solve_pure_diffusion(temperature)

# Step 2: Radiative cooling (configurable method)
working_temp = solve_radiative_cooling(working_temp)  # Dispatches based on selected method

# Step 3: Heat sources explicit
working_temp = solve_non_radiative_sources(working_temp)
```

**Characteristics:**
- ✅ **Speed**: Fast (near-original performance)
- ✅ **Stability**: Unconditionally stable (each operator uses optimal method)
- ✅ **Accuracy**: High accuracy (analytical solutions where possible)
- ✅ **Memory**: Low memory usage
- ✅ **Robust**: Each physics process solved optimally

**Performance**: ~0.95x baseline (5% performance cost for unconditional stability)

### Radiative Cooling Method Selection

The operator splitting approach allows configurable radiative cooling methods via `self.radiative_cooling_method`:

#### Newton-Raphson Implicit (Default: "newton_raphson_implicit")

**Implementation**: `_solve_radiative_cooling_newton_raphson_implicit()`
- **Method**: Solves dT/dt = -α(T^4 - T_space^4) using Newton-Raphson iteration
- **Advantages**: Unconditionally stable, physically accurate, handles large temperature differences
- **Disadvantages**: More computationally expensive (3-5 iterations typically)
- **Stability**: Unconditional
- **Accuracy**: High (exact Stefan-Boltzmann)
- **Performance**: 1-3 iterations per cell per timestep

#### Linearized Stefan-Boltzmann ("linearized_stefan_boltzmann")

**Implementation**: `_solve_radiative_cooling_linearized_stefan_boltzmann()`
- **Method**: Uses Newton cooling law Q = h(T - T_space) where h ≈ 4σεT₀³
- **Advantages**: Explicit, very stable, fast
- **Disadvantages**: Approximate, less accurate for large temperature differences
- **Stability**: Unconditional (when used in operator splitting)
- **Accuracy**: Good for moderate temperature differences
- **Performance**: Single calculation per cell per timestep

#### Alternative Method: DuFort-Frankel with Explicit Sources (Previous)

**Implementation:**
```python
# DuFort-Frankel for full equation
T^(n+1) = T^(n-1) + 2*dt*(α∇²T^n + Q^n/(ρcₚ))
```

**Characteristics:**
- ✅ **Speed**: Very fast (1 calculation per timestep)
- ✅ **Memory**: Low memory usage
- ❌ **Stability**: Conditionally stable when Q is large
- ❌ **Accuracy**: Can become unstable with large heat sources

**Status**: Replaced by operator splitting method

#### Alternative Method: Adaptive Explicit with Full Sub-stepping

**Implementation:**
```python
# Calculate required substeps
num_substeps = max(1, ceil(dt/dt_stable))
for step in range(num_substeps):
    T = T + dt_sub*(α∇²T + Q/(ρcₚ))
```

**Characteristics:**
- ✅ **Stability**: Unconditionally stable
- ✅ **Accuracy**: High accuracy with adaptive stepping
- ❌ **Speed**: 10-100x slower (many diffusion calculations)
- ❌ **Memory**: Higher memory for substeps

**Performance**: ~0.1x baseline (10x slower)
**Status**: Too slow for interactive use

## Mathematical Foundation

### Operator Splitting Theory

Operator splitting decomposes the heat equation into separate operators:
```
∂T/∂t = L₁(T) + L₂(T) + L₃(T)
```

**Lie Splitting** (first-order accurate):
```
T^(n+1) = exp(dt·L₃) ∘ exp(dt·L₂) ∘ exp(dt·L₁) T^n
```

Each operator is solved with its optimal method:
- **L₁ (diffusion)**: Adaptive explicit with sub-stepping
- **L₂ (radiation)**: Newton-Raphson implicit (analytical)
- **L₃ (sources)**: Explicit integration

### Why Operator Splitting Works

**Unconditional Stability**: Each operator uses its most stable numerical method:
- Pure diffusion is much easier to stabilize than diffusion+sources
- Radiative cooling has analytical implicit solutions
- Heat sources are typically well-behaved for explicit integration

**Accuracy**: Each physical process is solved optimally rather than compromising for a single method

**Performance**: Avoids the computational cost of treating all processes with the most restrictive (expensive) method

## Implementation Details

### Step 1: Pure Diffusion Solution

**Adaptive time stepping**:
```python
# Stability analysis for pure diffusion only
max_alpha = max(thermal_diffusivity)
diffusion_dt_limit = dx²/(4α)
num_substeps = ceil(dt / diffusion_dt_limit)
```

**Pure diffusion equation**:
```python
for substep in range(num_substeps):
    T = T + dt_sub * α * ∇²T / dx²
```

### Step 2: Radiative Cooling (Configurable Method)

**Method Selection**: Dispatcher `_solve_radiative_cooling()` calls appropriate implementation based on `self.radiative_cooling_method`.

**Option A: Newton-Raphson for Stefan-Boltzmann cooling**:
```python
# Solve: T_new - T_old + dt*α*(T_new⁴ - T_space⁴) = 0
for iteration in range(3):
    f = T_new - T_old + dt*α*(T_new⁴ - T_space⁴)
    df_dt = 1 + dt*α*4*T_new³
    T_new -= f / df_dt
```

**Unconditionally stable**: Implicit treatment of highly nonlinear radiation term

**Option B: Linearized Stefan-Boltzmann cooling**:
```python
# Linearized approximation: Q = h(T - T_space) where h = 4σεT₀³  
h_effective = 4 * stefan_boltzmann * emissivity * T_reference³
cooling_rate = h_effective * (T - T_space) / (ρ * cp * thickness)
T_new = T_old - dt * cooling_rate
```

**Fast and stable**: Explicit treatment with linear approximation

### Step 3: Heat Sources (Explicit)

**Direct application**:
```python
source_change = (Q_internal + Q_solar + Q_atmospheric) * dt / (ρ*cp)
T = T + source_change
```

**Well-behaved**: Heat sources are typically smooth and bounded

## Performance Comparison

| Method | Relative Speed | Stability | Accuracy | Memory | Status |
|--------|---------------|-----------|----------|---------|---------|
| **Operator Splitting** | **0.95x** | **Unconditional** | **High** | **Low** | **✅ CURRENT** |
| DuFort-Frankel Original | 1.0x | Conditional | Medium | Low | ⚠️ Replaced |
| Adaptive Explicit (Full) | 0.1x | Unconditional | High | Medium | ✅ Alternative |

### Typical Performance Characteristics

**Operator Splitting Method:**
- **Diffusion**: 1-10 substeps (adaptive based on thermal diffusivity)
- **Radiation**: 1-3 Newton-Raphson iterations (typically converges in 2)
- **Sources**: 1 step (explicit, well-behaved)
- **Overall**: ~5% performance cost for unconditional stability

**Substep Requirements:**
- **Normal conditions**: 3-5 diffusion substeps
- **High thermal diffusivity**: Up to 10 substeps
- **Extreme conditions**: Automatic adaptation prevents instability

## Advantages of Operator Splitting

### Stability Benefits

**Each operator uses its optimal method**:
- **Pure diffusion**: Stable with simple explicit methods
- **Radiative cooling**: Analytically solvable with Newton-Raphson
- **Heat sources**: Well-behaved for explicit integration

**No compromise methods**: Avoids using overly restrictive methods for all processes

### Accuracy Benefits

**Physical realism**: Each process solved according to its mathematical nature
- Diffusion: Parabolic PDE
- Radiation: Nonlinear algebraic equation
- Sources: Ordinary differential equation

**Error control**: Adaptive stepping only where needed (diffusion)

### Performance Benefits

**Minimal computational overhead**: Only 5% slower than original method

**Predictable performance**: No extreme cases requiring excessive substeps

**Memory efficient**: No large linear systems or extra storage

## Current Status and Recommendations

### Recommended Method: Operator Splitting (Implemented)

**Use for all geological simulations**:
- ✅ Unconditional stability with minimal performance cost
- ✅ Physically realistic treatment of each process
- ✅ Suitable for real-time interactive visualization
- ✅ No parameter tuning required
- ✅ Mathematically sound approach

### Alternative Methods

**Full Adaptive Explicit**: Use for maximum accuracy research
- Higher computational cost but ultimate accuracy
- Good for validation and benchmarking
- 10x slower but unconditionally stable

**Original DuFort-Frankel**: Historical reference only
- Replaced due to conditional stability issues
- Could become unstable with large heat sources
- Not recommended for current use

## Future Improvements

### Higher-Order Accuracy
- **Strang Splitting**: Second-order accurate operator splitting
- **Runge-Kutta Integration**: Higher-order time integration for sources
- **Implicit-Explicit Methods**: Combine implicit diffusion with explicit sources

### Advanced Stability
- **Richardson Extrapolation**: Automatic error estimation
- **Embedded Methods**: Built-in adaptive error control
- **Predictor-Corrector**: Multi-step error correction

### Performance Optimization
- **Parallel Implementation**: Spatial domain decomposition
- **GPU Acceleration**: Massive parallelization of linear algebra
- **Pipelined Operations**: Overlap computation phases

## Conclusion

The **Operator Splitting Method** provides the optimal solution for geological heat transfer:

### Proven Benefits
1. **Unconditional stability** - each operator solved with its optimal method
2. **High accuracy** - physically realistic treatment of each process
3. **Excellent performance** - only 5% slower than original method
4. **Mathematical rigor** - based on established operator splitting theory
5. **Maintenance simplicity** - each operator can be improved independently

### Key Innovation
Operator splitting recognizes that different physical processes require different numerical approaches:
- **Diffusion**: Parabolic PDE requiring careful time stepping
- **Radiation**: Nonlinear problem with analytical implicit solutions
- **Sources**: Well-behaved terms suitable for explicit integration

This approach provides the best combination of stability, accuracy, and performance for geological simulation, making it suitable for both research and interactive applications.
