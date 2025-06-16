# PHYSICS REFERENCE - 2D Geological Simulation

This document serves as the authoritative reference for all physical processes, laws, and equations implemented in the 2D geological simulation engine.

## GOLDEN RULES

1. Do not add artificial limits, e.g. minimum or maximum temperatures, or minimum or maximum temperature changes per step. These are artificial limits that can obscure bugs in the code or true physical phenomena. The only exception is that temperature can not go below absolute zero (0 K). If it does that indicates a problem with the model.

2. Do not keep legacy code or legacy interfaces to maintain compatibility. The code for this project is self-contained. There are no external callers of this code.

3. Use a dispatcher design pattern to select between multiple physics implementation options.

4. When in doubt, add traceback logging so that any error sources are correctly identified.


## TABLE OF CONTENTS

- [Kinematic Equation & Force Balance](#kinematic-equation--force-balance)
- [Cell-Swapping Mechanics](#cell-swapping-mechanics)
- [Heat Transfer Physics](#heat-transfer-physics)
  - [Heat Diffusion Methods](#heat-diffusion-methods)
  - [Operator Splitting Implementation](#operator-splitting-implementation)
- [Gravitational Physics](#gravitational-physics)
- [Pressure Calculations](#pressure-calculations)
  - [Pressure Solver Options](#pressure-solver-options)
- [Surface Tension](#surface-tension)
- [Enhanced Solid Mechanics](#enhanced-solid-mechanics)
- [Density-Driven Motion and Fluid Dynamics](#density-driven-motion-and-fluid-dynamics)
- [Material Properties & Transitions](#material-properties--transitions)
- [Atmospheric Physics](#atmospheric-physics)
- [Solar & Radiative Physics](#solar--radiative-physics)
- [Geological Processes](#geological-processes)
- [Units & Constants](#units--constants)
- [Numerical Methods](#numerical-methods)
  - [Spatial Kernels & Isotropy](#spatial-kernels--isotropy)
- [Physical Assumptions](#physical-assumptions)
- [Cell Conservation Exceptions](#cell-conservation-exceptions)
- [Motion Physics Improvements](#motion-physics-improvements)
- [Implementation Roadmap](#implementation-roadmap)
- [Open Items](#open-items)

---

## KINEMATIC EQUATION & FORCE BALANCE

The simulation evolves velocity `v` using the total force per unit mass acting on each cell:

```
dv/dt = (F_gravity + F_buoyancy + F_pressure + F_viscosity) / m
```

where

- `m = ρ × V_cell` is the mass of the cell
- `V_cell` = cell volume
- `F_gravity` = gravitational force toward the centre of mass
- `F_buoyancy` = upward force from density differences
- `F_pressure` = pressure gradient force from fluid flow, gravity
- `F_viscosity` = diffusive momentum transport (internal friction)

Each force component is detailed in the sections that follow.

Temperature enters via several forces. For example, the buoyancy force (`F_buoyancy`) and via the temperature-dependent binding strength used in the Cell-Swapping Mechanics.

---

## CELL-SWAPPING MECHANICS

The rules below govern all exchanges of mass, momentum and energy between neighbouring cells. They replace the former density-ratio swap criterion.

1. Net-force test – For each cell pair compute the magnitude of the cumulative force vector
   |F_net|. If |F_net| ≤ B_thresh, the pair is considered bound and no motion is attempted.
2. Velocity-difference test – When the binding test fails, the cells may swap if the relative velocity satisfies
   |v_A − v_B| ≥ Δv_thresh (typically 0.1 m s⁻¹).
3. Force threshold matrix – Threshold forces are material-pair dependent, for example:


   |          | Fluid         | Solid |
   |----------|---------------|---------------|
   | Fluid    | 0             | 0             |
   | Solid    | 0             | Ft_ss         |

A practical functional form for solid cohesion is

```
Ft = Fth0 * T_factor * W_factor
```

where:
- `Fth0` is the reference rock cohesion
- `T_factor` is the temperature factor
- `W_factor` is the weathering factor

If the net force on a cell is smaller than the force threshold, the cell is considered bound and no motion is attempted. If it exceeds the threshold, the cells may swap if the relative velocity satisfies |v_A − v_B| ≥ Δv_thresh.

### Implementation Status and Lessons Learned

**Status**: Successfully implemented and replaces all density-ratio swapping logic.

#### Key Implementation Insights

1. **Simplified Iteration Approach**: Direct cell-by-cell iteration with neighbor checking is more reliable than complex vectorized slicing operations that can introduce coordinate mapping errors.

2. **Directional Force Logic**: For surface tension and expansion effects, use `abs(proj_src) > src_bind` rather than `proj_src > src_bind` to allow forces pointing away from targets (negative projections).

3. **Velocity Threshold Tuning**: 
   - Original: 0.1 m/s (too high for surface tension effects)
   - Surface tension: 0.001 m/s (allows low-velocity cohesive swaps)
   - Recommendation: Material-dependent thresholds

4. **Binding Force Matrix**: Pre-computed lookup table with temperature scaling:
   ```python
   # Fluid-fluid: 0 N (no binding)
   # Fluid-solid: 0.5 × base_force × temp_factor
   # Solid-solid: 1.0 × base_force × temp_factor
   ```

5. **Asymmetric Swap Criteria**: Only source cell needs to overcome binding when target is non-rigid (fluids/space). Both cells must overcome binding for solid-solid swaps.

#### Force Field Assembly

Total force per unit volume:
```
F_total = F_gravity + F_pressure + F_buoyancy + F_viscosity
```

Where:
- `F_gravity = ρ × g` (gravitational body force)
- `F_pressure = -∇P` (pressure gradient force)
- `F_buoyancy` = local density contrast effects
- `F_viscosity` = momentum diffusion (future implementation)

#### Critical Implementation Details

1. **Pressure Force Calculation**: Use face-normal differences rather than central differences for stronger interface forces:
   ```python
   # X-direction forces
   fx_pressure[:, :-1] -= (P[:, :-1] - P[:, 1:]) / dx
   fx_pressure[:, 1:]  += (P[:, :-1] - P[:, 1:]) / dx
   ```

2. **Neighbor Restriction**: Use 4-connected neighbors only (no diagonals) to prevent unrealistic diagonal swaps.

3. **Deduplication**: Essential to prevent conflicting swaps when multiple cells target the same location.

#### Integration with Density Settling

**Current Issue**: Traditional density-based settling passes can undo force-based swaps, particularly surface tension effects.

**Solution Needed**: Modify settling logic to respect force-based criteria rather than pure density differences. Consider force thresholds in settling decisions.

Implementation pseudo-code:

```python
for each neighbour pair (A,B):
    F_net = forces[A] - forces[B]          # N
    B_thresh = binding_lookup[A.type, B.type]
    if abs(F_net) > B_thresh:
        if norm(v[A] - v[B]) >= dv_thresh:
            swap(A, B)
```

This replaces the old density-ratio swapping logic with physics-based force criteria that naturally incorporate surface tension effects.

### Centre of Mass Calculation  (`_calculate_center_of_mass`)
Gravity in every mass-movement routine points toward a dynamically updated centre-of-mass (COM):

1. Cell masses – `m = ρ · V` where `V = cell_size³` and `ρ` already includes thermal-expansion corrections.
2. Coordinates – build `x_idx` and `y_idx` arrays (0 … width-1 / height-1) representing cell centres.
3. Mask – exclude `MaterialType.SPACE`; vacuum has zero mass.
4. First moments  
   `Σm   = mass[mask].sum()`  
   `Σmx  = (mass * x_idx)[mask].sum()`  
   `Σmy  = (mass * y_idx)[mask].sum()`
5. COM  
   `COM_x = Σmx / Σm`  
   `COM_y = Σmy / Σm`

The coordinates are stored in `self.center_of_mass` (floats).  All gravity-driven 
algorithms use the vector pointing from a voxel to this COM as the inward
"down" direction.  Because density updates every macro-step, large magma bodies
or buoyant plumes can shift the COM and slightly re-orient gravity, giving a
first-order coupling between thermal/density anomalies and the gravitational
field without solving Poisson's equation.

#### Full Poisson Formulation (variable-density)

The analytic depth formula above assumes a column with piece-wise constant
density.  When large compositional or thermal anomalies are present the
density field varies laterally and the 1-D approximation under-estimates
pressure contrasts.  A more general approach starts from hydrostatic balance

∇P = ρ g

Taking the divergence and assuming the gravitational acceleration g points
everywhere toward the planet's centre (magnitude g constant on the small
scales of the model) gives

∇·((1/ρ) ∇P) = −ρ g r̂

On the discrete grid this is solved each macro-step with a Successive
Over-Relaxation (SOR) scheme:

```python
# RHS from local density and radial unit vector (toward COM)
rhs = -rho * g_r   # g_r = (dx_to_COM*unit + dy_to_COM*unit)
P = np.zeros_like(rho)
for iteration in range(max_iter):
    P_new = 0.25 * (np.roll(P,+1,0)+np.roll(P,-1,0)+
                    np.roll(P,+1,1)+np.roll(P,-1,1) - rhs*dx*dx)
    P[:] = omega * P_new + (1-omega) * P         # ω≈1.7
    if max_residual < 1e-3:                       # < 1 kPa
        break
# Vacuum boundary: P = 0 in SPACE cells
P[material_types == SPACE] = 0.0
```

The solver converges in ~50 iterations on a 100 × 100 grid and exactly
reproduces the analytic depth-law when \(\rho\) is uniform.

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

Diffusivity may be locally enhanced in fluids or at fluid-solid interfaces to model convection.

### Heat Source Terms

The total heat source `Q_total` is:

```
Q_total = Q_internal + Q_solar − Q_radiative
```

Where:

1. Internal heating (`Q_internal`)
2. Solar heating (`Q_solar`)
3. Radiative cooling sink (`Q_radiative`)

#### Internal Heat Generation

Internal planet heat sources are computed as:

Q_internal = Q_crust + Q_core

where crust heating may arise from tidal forces and core heating from radioactive decay.

```
Q_crust(d) = q0_crust * exp(-(1 − d)/λ_crust)
```

where
- `d` = relative depth (0 at surface, 1 at centre)
- `q0_crust`, `λ_crust` = tunable parameters defined in `heat_transfer.py`.

Q_core(d) = q0_core * exp(−(d/σ_core)²)

where
- `q0_core` = core heating rate at the center (W/m³)
- `σ_core` = core heating decay length (m)

#### Internal Heating Implementation

Geothermal energy is injected every step by `_calculate_internal_heating_source`:
* Exponential depth-dependent profile:  
  `Q = Q0 * exp(-depth / core_heating_depth_scale)`  (W m⁻³).  
* Adds heat explicitly in operator-split Step 3; contributes to `power_density` bookkeeping.

#### Solar Heating

Solar insolation adds heat primarily at the surface and decays with depth due to absorption. A simple empirical form is used:

```
Q_solar = S0 * (1 - albedo) * cos(latitude) * exp(-d/λ_solar)
```

where:
- `S0` is the solar constant at the top of the atmosphere (W m⁻²)
- `albedo` is the local reflectance (0–1)
- `latitude` sets the zenith angle (`cos(latitude)` ≈ insolation factor)
- `d` is depth measured from the surface (m)
- `λ_solar` is the attenuation length (m)

#### Radiative Cooling

Black body radiation removes energy from the planet.

```
Q_radiative = ε σ (T⁴ - T_space⁴)
```

where:
- `ε` is the thermal emissivity of the cell (0–1)
- `σ` = 5.67 × 10⁻⁸ W m⁻² K⁻⁴ is the Stefan-Boltzmann constant
- `T_space` = 2.7 K is the cosmic background temperature

The term is negative in the energy balance and thus acts as a sink in `Q_total`.

### Heat Diffusion Methods

This section compares different numerical methods for solving the heat diffusion equation with source terms in the geology simulator.

#### The Problem

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
1. Large heat sources (solar, internal heating, radiative cooling)
2. Multiple time scales (diffusion: years, sources: seconds)
3. Stability requirements for long-term evolution
4. Performance constraints (real-time visualization)

#### Current Solution: Operator Splitting Method

The simulation now uses operator splitting to solve the heat equation optimally. This approach treats different physical processes separately using their most appropriate numerical methods.

### Operator Splitting Implementation

The heat equation is split into separate operators:
```
∂T/∂t = L_diffusion(T) + L_radiation(T) + L_sources(T)
```

Where:
- `L_diffusion(T) = α∇²T` (pure diffusion)
- `L_radiation(T) = radiative cooling` (Stefan-Boltzmann)
- `L_sources(T) = internal + solar + atmospheric heating`

#### Three-Step Solution Process

Step 1: Pure Diffusion
```python
T₁ = solve_pure_diffusion(T₀, dt)
```
Uses adaptive explicit method with sub-stepping for stability.

Step 2: Radiative Cooling (Configurable Method)
```python
T₂ = solve_radiative_cooling(T₁, dt)  # Dispatches to selected method
```
Configurable implementation - either Newton-Raphson implicit or linearized Stefan-Boltzmann.

Step 3: Heat Sources (Explicit)
```python
T₃ = solve_heat_sources_explicit(T₂, dt)
```
Applies internal heating, solar heating, and atmospheric heating explicitly.

#### Method Comparison

##### Current Method: Operator Splitting (Implemented)

Implementation:
```python
# Step 1: Pure diffusion with adaptive stepping
working_temp, stability = solve_pure_diffusion(temperature)

# Step 2: Radiative cooling (configurable method)
working_temp = solve_radiative_cooling(working_temp)  # Dispatches based on selected method

# Step 3: Heat sources explicit
working_temp = solve_non_radiative_sources(working_temp)
```

Characteristics:
- Speed: Fast (near-original performance)
- Stability: Unconditionally stable (each operator uses optimal method)
- Accuracy: High accuracy (analytical solutions where possible)
- Memory: Low memory usage
- Robust: Each physics process solved optimally

Performance: ~0.95x baseline (5% performance cost for unconditional stability)

#### Radiative Cooling Method Selection

The operator splitting approach allows configurable radiative cooling methods via `self.radiative_cooling_method`:

##### Newton-Raphson Implicit (Default: "newton_raphson_implicit")

Implementation: `_solve_radiative_cooling_newton_raphson_implicit()`
- Method: Solves dT/dt = -α(T^4 - T_space^4) using Newton-Raphson iteration
- Advantages: Unconditionally stable, physically accurate, handles large temperature differences
- Disadvantages: More computationally expensive (3-5 iterations typically)
- Stability: Unconditional
- Accuracy: High (exact Stefan-Boltzmann)
- Performance: 1-3 iterations per cell per timestep

##### Linearized Stefan-Boltzmann ("linearized_stefan_boltzmann")

Implementation: `_solve_radiative_cooling_linearized_stefan_boltzmann()`
- Method: Uses Newton cooling law Q = h(T - T_space) where h ≈ 4σεT₀³
- Advantages: Explicit, very stable, fast
- Disadvantages: Approximate, less accurate for large temperature differences
- Stability: Unconditional (when used in operator splitting)
- Accuracy: Good for moderate temperature differences
- Performance: Single calculation per cell per timestep

#### Mathematical Foundation

##### Operator Splitting Theory

Operator splitting decomposes the heat equation into separate operators:
```
∂T/∂t = L₁(T) + L₂(T) + L₃(T)
```

Lie Splitting (first-order accurate):
```
T^(n+1) = exp(dt·L₃) ∘ exp(dt·L₂) ∘ exp(dt·L₁) T^n
```

Each operator is solved with its optimal method:
- L₁ (diffusion): Adaptive explicit with sub-stepping
- L₂ (radiation): Newton-Raphson implicit (analytical)
- L₃ (sources): Explicit integration

##### Why Operator Splitting Works

Unconditional Stability: Each operator uses its most stable numerical method:
- Pure diffusion is much easier to stabilize than diffusion+sources
- Radiative cooling has analytical implicit solutions
- Heat sources are typically well-behaved for explicit integration

Accuracy: Each physical process is solved optimally rather than compromising for a single method

Performance: Avoids the computational cost of treating all processes with the most restrictive (expensive) method

#### Implementation Details

##### Step 1: Pure Diffusion Solution

Adaptive time stepping:
```python
# Stability analysis for pure diffusion only
max_alpha = max(thermal_diffusivity)
diffusion_dt_limit = dx²/(4α)
num_substeps = ceil(dt / diffusion_dt_limit)
```

Pure diffusion equation:
```python
for substep in range(num_substeps):
    T = T + dt_sub * α * ∇²T / dx²
```

##### Step 2: Radiative Cooling (Configurable Method)

Method Selection: Dispatcher `_solve_radiative_cooling()` calls appropriate implementation based on `self.radiative_cooling_method`.

Option A: Newton-Raphson for Stefan-Boltzmann cooling:
```python
# Solve: T_new - T_old + dt*α*(T_new⁴ - T_space⁴) = 0
for iteration in range(3):
    f = T_new - T_old + dt*α*(T_new⁴ - T_space⁴)
    df_dt = 1 + dt*α*4*T_new³
    T_new -= f / df_dt
```

Unconditionally stable: Implicit treatment of highly nonlinear radiation term

Option B: Linearized Stefan-Boltzmann cooling:
```python
# Linearized approximation: Q = h(T - T_space) where h = 4σεT₀³  
h_effective = 4 * stefan_boltzmann * emissivity * T_reference³
cooling_rate = h_effective * (T - T_space) / (ρ * cp * thickness)
T_new = T_old - dt * cooling_rate
```

Fast and stable: Explicit treatment with linear approximation

##### Step 3: Heat Sources (Explicit)

Direct application:
```python
source_change = (Q_internal + Q_solar + Q_atmospheric) * dt / (ρ*cp)
T = T + source_change
```

Well-behaved: Heat sources are typically smooth and bounded

#### Performance Comparison

| Method | Relative Speed | Stability | Accuracy | Memory | Status |
|--------|---------------|-----------|----------|---------|---------|
| Operator Splitting | 0.95x | Unconditional | High | Low | CURRENT |
| DuFort-Frankel Original | 1.0x | Conditional | Medium | Low | Replaced |
| Adaptive Explicit (Full) | 0.1x | Unconditional | High | Medium | Alternative |

##### Typical Performance Characteristics

Operator Splitting Method:
- Diffusion: 1-10 substeps (adaptive based on thermal diffusivity)
- Radiation: 1-3 Newton-Raphson iterations (typically converges in 2)
- Sources: 1 step (explicit, well-behaved)
- Overall: ~5% performance cost for unconditional stability

Substep Requirements:
- Normal conditions: 3-5 diffusion substeps
- High thermal diffusivity: Up to 10 substeps
- Extreme conditions: Automatic adaptation prevents instability

#### Advantages of Operator Splitting

##### Stability Benefits

Each operator uses its optimal method:
- Pure diffusion: Stable with simple explicit methods
- Radiative cooling: Analytically solvable with Newton-Raphson
- Heat sources: Well-behaved for explicit integration

No compromise methods: Avoids using overly restrictive methods for all processes

##### Accuracy Benefits

Physical realism: Each process solved according to its mathematical nature
- Diffusion: Parabolic PDE
- Radiation: Nonlinear algebraic equation
- Sources: Ordinary differential equation

Error control: Adaptive stepping only where needed (diffusion)

##### Performance Benefits

Minimal computational overhead: Only 5% slower than original method

Predictable performance: No extreme cases requiring excessive substeps

Memory efficient: No large linear systems or extra storage

#### Current Status and Recommendations

##### Recommended Method: Operator Splitting (Implemented)

Use for all geological simulations:
- Unconditional stability with minimal performance cost
- Physically realistic treatment of each process
- Suitable for real-time interactive visualization
- No parameter tuning required
- Mathematically sound approach

##### Alternative Methods

Full Adaptive Explicit: Use for maximum accuracy research
- Higher computational cost but ultimate accuracy
- Good for validation and benchmarking
- 10x slower but unconditionally stable

Original DuFort-Frankel: Historical reference only
- Replaced due to conditional stability issues
- Could become unstable with large heat sources
- Not recommended for current use

#### Future Improvements

##### Higher-Order Accuracy
- Strang Splitting: Second-order accurate operator splitting
- Runge-Kutta Integration: Higher-order time integration for sources
- Implicit-Explicit Methods: Combine implicit diffusion with explicit sources

##### Advanced Stability
- Richardson Extrapolation: Automatic error estimation
- Embedded Methods: Built-in adaptive error control
- Predictor-Corrector: Multi-step error correction

##### Performance Optimization
- Parallel Implementation: Spatial domain decomposition
- GPU Acceleration: Massive parallelization of linear algebra
- Pipelined Operations: Overlap computation phases

#### Conclusion

The Operator Splitting Method provides the optimal solution for geological heat transfer:

##### Proven Benefits
1. Unconditional stability - each operator solved with its optimal method
2. High accuracy - physically realistic treatment of each process
3. Excellent performance - only 5% slower than original method
4. Mathematical rigor - based on established operator splitting theory
5. Maintenance simplicity - each operator can be improved independently

##### Key Innovation
Operator splitting recognizes that different physical processes require different numerical approaches:
- Diffusion: Parabolic PDE requiring careful time stepping
- Radiation: Nonlinear problem with analytical implicit solutions
- Sources: Well-behaved terms suitable for explicit integration

This approach provides the best combination of stability, accuracy, and performance for geological simulation, making it suitable for both research and interactive applications.

---

## GRAVITATIONAL PHYSICS

### Gravity field calculation

The gravitational acceleration field g is obtained from a scalar potential *Φ* that satisfies the Poisson equation for a continuous mass distribution:

```
∇²Φ = 4 π G ρ
```

Once Φ is known, the acceleration acting on each cell is simply

```
g = -∇Φ
```

The density field changes every step (temperature change, cell migration) the Poisson problem must be re-solved frequently, so a fast numerical method is important.

### Poisson solver

Two fast numerical schemes are available:

FFT Poisson solver
  - Fast O(N log N) for uniform Cartesian grids with periodic or free-space boundary handling by zeroing the k=0 mode.
  - Recommended for standard rectangular planets.
  - Not recommended when density varies sharply or the domain is non-rectangular
  - Thus we do not use this method

Geometric multigrid
   - Robust for arbitrary boundary conditions or masked domains; typically converges in 6–8 sweeps using a V-cycle (our current default).
   - An F-cycle (extra coarse-grid visits) can further accelerate convergence for highly heterogeneous density fields at the cost of ~1.3× work per solve.
   - Recommended for irregular domains or when density varies sharply
   - We use this method with a TBD mix of V-cycle vs F-cycle

To mitigate ringing from sharp density jumps we optionally smooth ρ with a small Gaussian kernel (σ ≈ 0.5 cell) before the gravity solve; the full-resolution density is retained for all other physics.

The solver interface returns (Φ, g_x, g_y) and caches spectral coefficients so that subsequent solves after minor density updates cost <50 % of the first call.

A brief comparison of the options is below.

| Solver                    | Grid size 80×80 | Time per solve (ms) | Typical iterations | Memory cost | Notes |
|---------------------------|-----------------|---------------------|--------------------|-------------|-------|
| Centre-of-mass (analytic) | –               | 0.0                 | –                  | none        | Gravity fixed, inaccurate for lumpy ρ |
| FFT (periodic)            | 0.5             | 0.5                 | 1 (direct)         | O(N)        | Fast but needs rectangular, moderate ringing, inaccurate for lumpy ρ |
| Multigrid V-cycle         | 6–8 sweeps      | 2.0                 | 6–8                | O(N)        | Default; good smooth-error reduction |
| Multigrid F-cycle         | 9–10 sweeps     | 2.6                 | 9–10               | O(N)        | +30 % work, 1.5–2× faster residual drop on rough ρ |

### Geometric multigrid details

Geometric multigrid attacks low-frequency (smooth) error on coarser grids and high-frequency error on finer grids.

Relax (smooth): perform a few Gauss-Seidel or Jacobi iterations on the current grid to damp the high-frequency error components.
Restrict (⇩): project the residual from a fine grid to the next coarser grid, usually by 2× decimation with weighted averaging.
Prolong (⇧) / correct: interpolate the coarse-grid correction back up to the finer grid and update the solution.

A cycle is the pattern in which the solver moves down (restrict) and up (prolong) through this grid hierarchy.

```
V-cycle (3 levels shown)

fine L0  ── relax ──▶ restrict
              │
              ▼
     coarse L1 ── relax ──▶ restrict
                        │
                        ▼
             coarse L2 (coarsest) – relax a few times
                        │
                        ▲  prolong + correct
     fine  L1 ◀─────────┘ relax
              ▲  prolong + correct
fine  L0 ◀────┘ relax
```

The solver visits the coarsest level once per cycle – like the letter V.  This is usually enough when the right-hand-side (density field) is smooth.

An F-cycle is more aggressive:

```
F-cycle (4 levels shown)

L0 → L1 → L2 → L3
       ▲    │
       │    └── back down to L2, relax, then up
       └────────── up to L1, relax, down again
finally back to L0
```

Think of drawing the letter F: you go down to the bottom, part-way back up, down again, then all the way up.  This re-visits the coarser grids multiple times, scrubbing out stubborn smooth error that appears when the density field has sharp contrasts.

Why not always use the F-cycle?  It does ~30 % more relaxation work.  In practice we monitor the residual; if it stagnates after one V-cycle we switch to an F-cycle for the next step, then fall back once convergence is healthy.

A concise mental model is:
- Jacobi (or red–black Gauss–Seidel) smoothing damps high-frequency error; plain Gauss–Seidel converges roughly twice as fast but is less parallel-friendly.
- Multigrid then transfers the remaining smooth error to coarser levels where it appears high-frequency again and is removed cheaply.
- The V-cycle is the minimal single-pass walk through the hierarchy.
- The F-cycle is a double-scrub that revisits coarse grids for extra smoothing.

### Buoyancy force

Materials move based on buoyancy forces in the gravitational field:

Effective density with thermal expansion:
```
ρ_eff = ρ₀ / (1 + β(T - T₀))
```

Where:
- `ρ₀` = reference density (kg/m³)
- `β` = volumetric thermal expansion coefficient (1/K)
- `T₀` = reference temperature (273.15 K)

Buoyancy conditions:
- Rising: Less dense material closer to center than denser material farther out
- Sinking: Denser material farther from center than less dense material closer in

---

## PRESSURE CALCULATIONS

### Pressure Distribution

Space: `P = 0` (vacuum)

Atmospheric pressure (exponential decay):
```
P_atmo = P_surface × exp(-h/H)
```
Where:
- `P_surface = 0.1 MPa`
- `h` = height above surface (m)
- `H = 8400 m` (scale height)

Hydrostatic pressure (fluids):
```
P_fluid = max(P_surface, ρ_fluid × g × depth / 10⁶)
```
Where:
- `ρ_fluid = 2000 kg/m³`
- `g = 9.81 m/s²`

Lithostatic pressure (solids):
```
P_solid = max(P_surface, ρ_solid × g × depth / 10⁶)
```
Where:
- `ρ_solid = 3000 kg/m³`

## SURFACE TENSION

**Status**: Implemented with physics-based cohesive force model achieving 50-100+ swaps per timestep.

Surface tension minimizes the surface area of fluid-vacuum interfaces through cohesive forces between fluid particles. This is now implemented using local curvature-based forces that allow bulk interface processing.

### Physical Model

Surface tension emerges from cohesive forces between fluid cells:
```
F_cohesion = σ × (n_max - n_current) × direction_to_neighbors
```

Where:
- `σ` = surface tension strength coefficient
- `n_max` = maximum possible neighbors (8 for 2D grid)
- `n_current` = current count of fluid neighbors
- Cells with fewer neighbors (higher curvature) experience stronger inward forces

### Implementation: Physics-Based Bulk Processing

The new `apply_physics_based_surface_tension()` method processes entire interfaces simultaneously:

1. **Interface Detection**: Identify all fluid cells adjacent to vacuum/space
2. **Curvature Calculation**: Count fluid neighbors for each interface cell
   - 1-2 neighbors = high curvature (sharp protrusion)
   - 3-5 neighbors = moderate curvature 
   - 6-8 neighbors = low curvature (flat or internal)
3. **Multi-Pass Processing**: Up to 3 passes of 50 swaps each per timestep
4. **Smart Target Selection**:
   - Find vacuum cells with most fluid neighbors (gaps to fill)
   - Move high-curvature fluid cells to low-curvature positions
   - Preserve momentum during swaps

### Performance Improvements

**Previous Limitations**:
- Only 3 swaps per timestep (sequential processing)
- Ad-hoc shape analysis (aspect ratios, COM calculations)
- No physical basis for movement decisions

**Current Performance**:
- 50-100+ swaps per timestep (bulk parallel processing)
- Physics-based curvature forces drive motion
- Water line collapses from 20:1 to ~2:1 aspect ratio in 10 steps
- Momentum-conserving material swaps
- Natural emergence of circular shapes from local rules

### Force-Based Integration

Surface tension is integrated with the unified kinematics system:

```python
# Calculate effective surface tension force
curvature = 8 - num_fluid_neighbors  # 0-7 scale
F_surface_tension = strength * curvature * direction_to_center

# Combined with other forces
F_total = F_gravity + F_buoyancy + F_surface_tension

# Swap when total force exceeds binding threshold
if |F_total| > binding_threshold:
    perform_swap()
```

### Key Technical Details

1. **Parallel Processing**: Process all interface cells simultaneously rather than sequentially
2. **Curvature-Based Priority**: High-curvature cells (protrusions) move first
3. **Gap Filling**: Actively identify and fill interior gaps in fluid bodies
4. **Momentum Conservation**: Swap velocities along with materials
5. **No Hard-Coded Shapes**: Circular/spherical shapes emerge naturally from local curvature minimization

### Remaining Challenges

1. **Discrete Grid Effects**: Some water conservation issues (~10-15% loss) due to discrete swapping
2. **Lateral Movement**: Sometimes needs encouragement through gap-filling logic
3. **Competition with Settling**: Gravity settling can interfere with surface tension reshaping

---

## ENHANCED SOLID MECHANICS

The simulation now properly handles rigid body dynamics and solid material interactions:

### Force-Based Swapping with Material Thresholds

The `apply_force_based_swapping()` method now correctly implements material-specific binding thresholds:

```python
# Check if source can overcome its own binding
src_bind = compute_binding_threshold(mt[sy, sx], temp[sy, sx])
if |F_src·direction| > src_bind:
    # Also check target binding for solids
    if is_solid(mt[ty, tx]):
        tgt_bind = compute_binding_threshold(mt[ty, tx], temp[ty, tx])
        if |F_net| > tgt_bind:
            perform_swap()
```

This ensures solids don't flow through each other like liquids.

### Rigid Body Group Dynamics

New methods identify and move connected rigid materials as coherent units:

1. **Group Identification**: `identify_rigid_groups()` uses connected component labeling
2. **Net Force Calculation**: Sum forces over entire group including buoyancy
3. **Coherent Motion**: `apply_group_dynamics()` moves groups as units
4. **Momentum Transfer**: Groups exchange momentum on collision

Benefits:
- Ice maintains shape while floating
- Rock groups fall and tumble realistically  
- Proper momentum transfer between colliding bodies
- Prevents artificial fragmentation of solid structures

## MATERIAL PROPERTIES & TRANSITIONS

### Metamorphism

General transition system: Each material type can have multiple P-T dependent transitions

Transition conditions:
```
T_min ≤ T ≤ T_max  AND  P_min ≤ P ≤ P_max
```

Examples:
- Rock → Magma (high temperature)
- Magma → Rock (cooling)
- Water → Ice (low temperature)
- Water → Water vapor (high temperature/low pressure)

### Phase Transitions During Convection

Water evaporation:
- Condition: `T > 350 K` (≈77°C)
- Probability: 5% per timestep

Water vapor condensation:
- Condition: `T < 320 K` (≈47°C)  
- Probability: 5% per timestep

Phase changes are data-driven via `MaterialDatabase`:
* Each `MaterialProperties` entry contains a list of `TransitionRule`(target, T/P window).
* `_apply_metamorphism` scans all non-space cells each macro-step and replaces materials whose local T-P falls inside a rule.
* Melting, crystallisation and gas ↔ liquid ↔ solid transitions (e.g., ICE ⇌ WATER ⇌ VAPOR, MAGMA → BASALT/GRANITE/OBSIDIAN) are all executed in-place – cells retain position & volume.

---

## ATMOSPHERIC PHYSICS
### Atmospheric Convection

Fast mixing process for atmospheric materials (Air, Water vapor)

Mixing equation:
```
T_new = T_old + f × (T_avg_neighbors - T_old)
```
Where `f = 0.3` (mixing fraction)

Neighbor calculation: Only atmospheric cells participate in averaging

`_apply_atmospheric_convection` performs a simple vertical mixing pass:
* For each AIR or WATER_VAPOR cell, it mixes a fraction `atmospheric_convection_mixing` of the temperature difference with the cell directly above – a cheap way to mimic day-time convection.

### Directional-Sweep Atmospheric Absorption (default)

The simulator now uses a single-pass Amanatides & Woo DDA sweep that marches
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

   `k` is a per-material absorption coefficient from `materials.py`:
   AIR 0.001, WATER_VAPOR 0.005, WATER 0.02, ICE 0.01, others 1.0.
5. The absorbed energy is converted to volumetric power density and split into
   `solar_input` (surface/solid/liquid cells) or `atmospheric_heating`
   (gas cells) for diagnostics.
6. The ray terminates if `k ≥ 1` (opaque) or when the remaining flux is zero.

Advantages:
- Accurate day/night shadowing for any solar angle.
- Eliminates the 10³–10⁵× flux spikes of the old radial scheme.
- Runs in linear time; suitable for real-time visualisation.

---

## SOLAR & RADIATIVE PHYSICS

### Solar Heating

Latitude-dependent intensity:
```
I_solar = I₀ × cos(latitude) × distance_factor
```
Where:
- `I₀ = 1361 W/m²` (solar constant)
- `distance_factor = 1×10⁻⁵`
- `latitude` = distance from equatorial plane

Albedo effects:
```
I_effective = I_solar × (1 - albedo)
```

Material albedos stored in material database

### Solar Heating & Greenhouse Effect
Incoming stellar flux is handled in two stages:
1. Raw insolation – `_calculate_solar_heating_source` projects a solar vector, applies distance factor & cosine-law shading, then multiplies by material albedo.
2. Atmospheric absorption – `_solve_atmospheric_absorption` (directional sweep) attenuates the beam through AIR / WATER_VAPOR columns; absorption coefficient comes from `MaterialDatabase._init_optical_absorption`.  
   *Greenhouse*: the outgoing long-wave cooling constant is multiplied by `(1 – greenhouse_factor)` where
   
  `greenhouse_factor = base + (max-base) * tanh( ln(1+M_vapor/scale) / 10 )`

### Radiative Cooling

Stefan-Boltzmann Law:
```
P_radiated = ε × σ × A × (T⁴ - T_space⁴)
```

Where:
- `ε` = emissivity (material-dependent)
- `σ = 5.67×10⁻⁸ W/(m²⋅K⁴)` (Stefan-Boltzmann constant)
- `T_space = 2.7 K` (cosmic background)

Greenhouse effect:
```
σ_eff = σ × (1 - greenhouse_factor)
```

Dynamic greenhouse:
```
greenhouse_factor = base + (max - base) × tanh(vapor_factor)
```
Where vapor_factor depends on atmospheric water vapor content

---

## GEOLOGICAL PROCESSES

### Weathering

Chemical weathering (Arrhenius-like):
```
Rate_chemical = exp((T - 15)/14.4) × water_factor
```
Where `water_factor = 3.0` if adjacent to water

Physical weathering:
- Freeze-thaw: Max effectiveness at 0°C
- Thermal stress: High temperature extremes

Weathering products: Material-specific, defined in material database

Surface chemistry is approximated by `_apply_weathering` (optional flag):
* Operates on the outermost crust layer (`surface_radiation_depth_fraction`).
* Converts rocks to their listed weathering products (e.g., GRANITE → SANDSTONE) at a slow stochastic rate, modelling mechanical & chemical erosion.

---

## UNITS & CONSTANTS

### Base Units
- Length: meters (m)
- Time: years (converted to seconds internally)
- Temperature: Kelvin (K)
- Pressure: Pascal (Pa), displayed as MPa
- Power: Watts (W)

### Key Constants
```
seconds_per_year = 365.25 × 24 × 3600 = 31,557,600 s
stefan_boltzmann_geological = 5.67×10⁻⁸ × seconds_per_year J/(year⋅m²⋅K⁴)
space_temperature = 2.7 K
reference_temperature = 273.15 K
average_gravity = 9.81 m/s²
```

### Typical Material Properties
- Density: 1000-8000 kg/m³
- Thermal conductivity: 0.1-400 W/(m⋅K)  
- Specific heat: 400-4200 J/(kg⋅K)
- Thermal expansion: 1×10⁻⁶ - 3×10⁻⁴ 1/K

---

## NUMERICAL METHODS

TODO: Add a brief summary of the equations that we need to solve

### Time Stepping

Adaptive timestepping: Currently fixed at `dt = 1 year`

DuFort-Frankel stability: Unconditionally stable for any timestep

### Spatial Discretization

Grid: Uniform Cartesian with square cells
Cell size: Typically 50 m per cell
Boundary conditions: Insulating (no-flux) at material-space interfaces

### Vectorization

NumPy arrays: All operations vectorized for performance
Morphological operations: Used for fast neighbor calculations
Boolean masking: Efficient material-type specific operations

### Performance Optimization

Quality levels:
1. Full (100% accuracy): Process all cells
2. Balanced (50% accuracy): Process 50% of cells  
3. Fast (20-33% accuracy): Process 20-33% of cells

Caching: Material property lookups cached for performance

Neighbor shuffling: Randomized to prevent grid artifacts

### Multigrid smoothers
The Poisson solvers (pressure, velocity projection) use a geometric multigrid V-cycle.  We currently employ *red-black Gauss–Seidel* (RB-GS) as the smoother because it damps high-frequency error roughly twice as fast per sweep as weighted Jacobi, particularly when the variable coefficient 1/ρ spans many orders of magnitude (air versus basalt).  Any convergent smoother would work – weighted-Jacobi, lexicographic Gauss-Seidel, Chebyshev, even a few conjugate-gradient iterations – the grid hierarchy is unchanged.  RB-GS was chosen for code reuse and robustness; swapping in a different smoother only requires a few lines in `pressure_solver.py`.

### Spatial Kernels & Isotropy
To minimise axial artefacts the engine uses pre-computed circular kernels for all morphological operations.

| Kernel | Size | Purpose |
|--------|------|---------|
| `_circular_kernel_3x3` | 3 × 3 (8-neighbour) | Fast neighbour look-ups (e.g., atmospheric absorption) – default when `neighbor_count = 8` |
| `_circular_kernel_5x5` | 5 × 5 (includes radius 2 offsets) | Isotropic candidate gathering for collapse, buoyancy, stratification – always used |
| `_collapse_kernel_4`   | 3 × 3 cross-shape | Strict 4-neighbour collapse for Manhattan-style movement – used when `neighbor_count = 4` (set automatically for `quality = 3`) |
| `_collapse_kernel_8`   | 3 × 3 full ring | Allows diagonal collapse moves – default (`neighbor_count = 8`, quality 1-2) |
| `_laplacian_kernel_radius1` (implicit) | 3 × 3 | Classic 8-neighbour Laplacian (explicit diffusion, fast) – selected when `diffusion_stencil = "radius1"` |
| `_laplacian_kernel_radius2` | 5 × 5, 13-point | Nearly isotropic Laplacian – default (`diffusion_stencil = "radius2"`) |

These kernels are generated once on startup and reused everywhere, ensuring that gravitational collapse, fluid migration and diffusion all respect circular symmetry on a Cartesian grid.

> Tip – any new morphological rule should reuse one of the existing kernels to preserve numerical isotropy.

---

## PHYSICAL ASSUMPTIONS

### Simplifications

1. 2D geometry: All processes assumed cylindrically symmetric
2. Incompressible flow: Density changes only via thermal expansion
3. Local thermodynamic equilibrium: No heat diffusion lag
4. Idealized materials: Properties constant within material types
5. No elasticity: Instantaneous stress relaxation
6. Simplified radiative transfer: No scattering, only absorption/emission

### Scaling

The simulation uses enhanced parameters for visibility:
- Internal heating rates ~1000× real values
- Simplified atmospheric physics
- Accelerated geological processes
- Enhanced thermal diffusivity for stability

This allows geological timescales (millions of years) to be observable in human timescales (minutes) while preserving physical relationships.

---

## CELL CONSERVATION EXCEPTIONS

In almost every numerical update the simulator treats each grid cell as an indestructible voxel – matter is merely moved or its phase changes _in-situ_.  For long-term stability we want all physics operators to preserve the count of MaterialType.SPACE cells (vacuum) unless something explicitly vents gas to space or accretes material from space.

The following operators currently violate that conservation principle by either turning non-SPACE material into SPACE, or by pulling existing SPACE inward so the outer vacuum region grows.  They should be revisited:

| Operator / Routine | Location | Trigger | How cell count changes |
|--------------------|----------|---------|------------------------|
| Gravitational collapse (`_apply_gravitational_collapse_vectorized`) | `simulation_engine.py` | Solid cell adjacent to a cavity (AIR / WATER / SPACE) moves into that cavity | If the chosen cavity is SPACE, the solid and vacuum swap positions – global SPACE count is unchanged, but vacuum is pulled inward (planet appears eroded). We still list it here because repeated swaps reshape the planet; safest long-term fix is to forbid swapping with SPACE and instead swap with AIR |
| Unsupported-cell settling (`_settle_unsupported_cells`) | `simulation_engine.py` | Any denser material directly above a lighter fluid (AIR, WATER_VAPOR, WATER, MAGMA, SPACE) swaps one cell toward the centre of mass | Cells now swap materials/temperatures; the lighter fluid rises, the heavier sinks, so global SPACE count stays constant (no synthetic vacuum pockets). |
| Pyroclastic interaction (*water + magma*) | not yet explicit | Future rule might flash-boil water, expelling vapor upward and leaving behind SPACE | Would destroy a WATER cell |
| Exsolution / out-gassing at very low pressure | placeholder | Planned volcanic venting routine | Could convert MAGMA → SPACE + AIR if vent blows out material |

### Why this matters
Mass and volume conservation are critical for numerical stability and for keeping the planet from being "eaten" by its vacuum surroundings.  Each of the above rules either:
1. Needs an alternate implementation that moves an equal & opposite amount of AIR (or other filler) so total SPACE remains constant, or
2. Must be accompanied by a replenishment mechanism (e.g., accretion of interstellar dust) so the net SPACE budget is balanced over time.

### Next steps
* Short-term safeguard: automated regression test (`test_space_integrity.py`) now fails if SPACE cells increase by more than a small tolerance.
* Medium term: refactor the two mechanical-movement routines so they swap with AIR instead of SPACE, preserving vacuum volume.
* Long term: audit any new thermochemical reactions using the checklist below before merging:
  1. Does the operator ever set a cell to `MaterialType.SPACE`?  If so, why?
  2. Could two cells merge?  If so, where does the "extra" cell go?
  3. Does the rule have an inverse that can fill the lost volume elsewhere?

> Keeping this table up-to-date will help us rapidly spot and fix future regressions.

> Swap conflicts: When two proposed swaps target the same cell (or each other) the helper `_dedupe_swap_pairs` keeps one swap and silently drops the others—no cell is cleared or set to SPACE. This guarantees cell-count preservation during mass movement passes.

---

## MOTION PHYSICS IMPROVEMENTS

### Previous Limitations (Now Addressed)

The original cell-swapping approach had fundamental limitations that prevented realistic fluid dynamics and rigid body behavior:

1. **Rate-Limited Individual Swaps**: FIXED - Now achieves 50-100+ swaps per timestep through bulk processing
2. **No Momentum Conservation**: FIXED - Material swaps now preserve velocities
3. **Lack of Coherent Motion**: FIXED - Rigid groups move as coherent units
4. **Sequential Processing**: FIXED - Bulk operations process many cells simultaneously

### Implemented Solutions

#### Phase 1: Enhanced Surface Tension (COMPLETED)
- Implemented bulk interface processing in `apply_physics_based_surface_tension()`
- Processes 50-100+ simultaneous swaps per timestep
- Curvature-based forces drive natural shape evolution
- Water lines collapse from 20:1 to ~2:1 aspect ratio in 10 steps

#### Phase 3: Group Dynamics (COMPLETED)
- Implemented `identify_rigid_groups()` using connected component labeling
- `apply_group_dynamics()` moves rigid bodies as coherent units
- Net forces calculated over entire groups including buoyancy
- Ice maintains shape while floating, rocks transfer momentum on impact

### Remaining Implementation Phases

#### Phase 2: Velocity Fields (NOT YET IMPLEMENTED)
Add continuous velocity fields alongside discrete cell states:

```python
# Continuous fields
velocity_x = np.zeros((height, width))
velocity_y = np.zeros((height, width))

# Update velocities based on forces
velocity += force / density * dt

# Use velocities for transport decisions
if |velocity| > binding_threshold:
    transport_material()
```

Benefits:
- Natural momentum conservation
- Smooth acceleration/deceleration
- Foundation for full fluid dynamics

#### Phase 4: Full Unified Kinematics (FUTURE WORK)
- Semi-Lagrangian advection
- Pressure projection for incompressibility
- Complete momentum conservation framework

### Current Implementation Details

#### Force-Based Swapping System
The `apply_force_based_swapping()` method now properly implements:
- Material-specific binding thresholds
- Momentum-conserving swaps
- Proper solid mechanics (solids don't flow through each other)

#### Surface Tension System
The `apply_physics_based_surface_tension()` method implements:
- Curvature-based cohesive forces
- Multi-pass bulk processing (3 passes × 50 swaps)
- Smart target selection for gap filling
- Natural emergence of circular shapes

#### Group Dynamics System
The rigid body system includes:
- Connected component analysis for material groups
- Net force integration over groups
- Coherent group motion with proper physics
- Maintains rigid body integrity during motion

### Key Physics Principles (Implemented)

1. **Conservation Laws**: Mass and momentum conserved in all operations
2. **Collective Behavior**: Connected rigid cells move together
3. **Parallel Processing**: Bulk operations on independent cells
4. **Force Integration**: Forces properly calculated and applied
5. **Binding Thresholds**: Materials resist motion based on physical properties

### Achieved Improvements

- Surface tension: Water lines collapse to circles in <10 timesteps (vs never)
- Rigid bodies: Ice maintains shape while floating and responds to forces
- Collisions: Proper force-based interactions between materials
- Performance: Bulk operations achieve 50-100+ swaps per timestep
- Physics-based: Removed ad-hoc rules in favor of physical principles

### Future Work

The velocity field implementation (Phase 2) and full unified kinematics (Phase 4) remain as future enhancements that would provide:
- Continuous velocity tracking
- Full Navier-Stokes fluid dynamics
- Complete pressure-velocity coupling
- Advanced fluid flow patterns

---

## IMPLEMENTATION ROADMAP

### Performance Strategy (≤ 16 ms on 100 × 100)

1. Add velocity fields `vx, vy` (float64, shape (h,w)).  Initialise to 0.
2. Provisional Velocity – compute all forces *except* pressure, vectorised NumPy:  
   `vx += Δt * ax`, `vy += Δt * ay`.
3. Pressure Solve – 15–25 Jacobi/SOR iterations:  
   ```python
   for iter in range(max_iter):
       P[1:-1,1:-1] = 0.25*(P[:-2,1:-1]+P[2:,1:-1]+P[1:-1,:-2]+P[1:-1,2:] 
                              - rhs*dx*dx)
   ```
   • `rhs = ρ/Δt * divergence(vx*,vy*)`  
   • Stop early when max residual < 1 Pa.
4. Velocity Projection – subtract gradient:  
   `vx -= Δt/ρ * (P[:,2:]-P[:,:-2])/(2Δx)` (analogous in y).
5. Material Advection – use *semi-Lagrangian* back-trace (two bilinear probes per cell) → stable at large Δt.
6. Phase / density update – reuse existing metamorphism functions; recompute ρ, ν.
7. Sparse Updates – keep a boolean `active_mask` (cells where |𝐯|, |Ṫ|, or material change > ε).  Only those and their 1-cell halo enter steps 2–5.
8. Quality Levels – reuse existing `quality` flag:  
   • Full: whole grid every step.  
   • Balanced: update `active_mask` only.  
   • Fast: subsample active cells (e.g., every other cell) each frame.
9. Solver Optimisation  
   • Pre-compute 1/ρ where possible.  
   • Use `numba.njit(parallel=True)` or move the Poisson solve to Cython.
10. Frame-Time Budget (100×100)  
    | Stage | Target Time | Notes |
    |-------|-------------|-------|
    | Force assembly        | ≤ 1 ms | vectorised NumPy |
    | Poisson (20 iter)     | ≤ 7 ms | SOR ω≈1.7, early-out |
    | Projection            | ≤ 1 ms | simple gradients |
    | Advection             | ≤ 4 ms | semi-Lagrangian, only active cells |
    | Misc/Book-keeping     | ≤ 3 ms | phase, IO, logging |
    Total ≈ 16 ms → 60 fps (safety margin included).

11. Validation Tests
    • Rising bubble test (air in water).  
    • Dam-break pressure surge.  
    • Rock-on-ice melt collapse.  
    • Hydrostatic rest ‑ zero velocity residual.

12. Staged Roll-Out
    a. Implement velocity & pressure arrays (no movement yet).  
    b. Enable gravity + buoyancy; verify static pressure.  
    c. Add pressure solve & projection.  
    d. Replace density-stratification / collapse with velocity-driven advection.  
    e. Benchmark & tune `active_mask` heuristics.

13. Maintenance
    • Keep the old three-pass system behind a feature flag for regression comparison.  
    • Unit-test the Poisson solver separately.  
    • Plot residual vs iteration each CI run to catch performance drifts.

> With these steps we gain a single, physically self-consistent motion model while preserving interactive frame rates on modest grids.

---

## OPEN ITEMS

These tasks have been agreed during the refactor sessions but are not yet implemented.  They are listed here so that any contributor can pick them up without digging through chat history.

### Immediate (blocking) 
1. Complete SI sweep – purge any remaining `seconds_per_year` maths in *tests* and documentation examples; delete the placeholder attribute from `simulation_engine_original.py` once reference tests pass.
2. FFT / DST pressure projection – replace the multigrid Poisson solver in `fluid_dynamics.py` with a frequency-space implementation for O(N log N) performance and predictable convergence.
3. Energy conservation regression – add an automated test that steps an isolated closed system for ≥10 years and asserts that total internal energy changes < 0.1 %.  This will guard against future source / sink sign errors.
4. Material property cache validation – convert the ad-hoc debug script into a pytest that randomly deletes materials and checks that `_material_props_cache` is perfectly pruned.

### Short-term enhancements
- Temperature-dependent viscosity: damp velocities as a smooth function of local melt fraction; this replaces the temporary solid drag factor 0.2.
- Variable cell-size support: allow `cell_size` ≠ 50 m so small-scale phenomena (lava tubes, glaciers) can be simulated in separate runs.
- Greenhouse coupling: link water-vapour mass directly to `atmospheric_processes.calculate_greenhouse_effect()` instead of the current heuristic.
- Moist-convective rainfall: precipitate WATER when saturated vapour cools below the Clausius-Clapeyron curve; feeds erosion module.

### Research backlog (nice-to-have)
- Coupled erosion & sediment transport (height-field + fluvial flow).
- Partial melt phase diagram for silicates: returns melt fraction and latent heat sink.
- GPU kernels for heat diffusion and Poisson solves via CuPy (optional acceleration path).
- 3-D extrusion prototype: prove that 2-D solver generalises to shallow-layer quasi-3-D without re-architecting.

# TODO:
- Talk about kernel sizes and isotropy

## Spatial Kernels & Isotropy
To minimise axial artefacts the engine uses pre-computed circular kernels for all morphological operations.

| Kernel | Size | Purpose |
|--------|------|---------|
| `_circular_kernel_3x3` | 3 × 3 (8-neighbour) | Fast neighbour look-ups (e.g., atmospheric absorption) – default when `neighbor_count = 8` |
| `_circular_kernel_5x5` | 5 × 5 (includes radius 2 offsets) | Isotropic candidate gathering for collapse, buoyancy, stratification – always used |
| `_collapse_kernel_4`   | 3 × 3 cross-shape | Strict 4-neighbour collapse for Manhattan-style movement – used when `neighbor_count = 4` (set automatically for `quality = 
3`) |
| `_collapse_kernel_8`   | 3 × 3 full ring | Allows diagonal collapse moves – default (`neighbor_count = 8`, quality 1-2) |
| `_laplacian_kernel_radius1` (implicit) | 3 × 3 | Classic 8-neighbour Laplacian (explicit diffusion, fast) – selected when `diffusion_stencil = "radius1"` |
| `_laplacian_kernel_radius2` | 5 × 5, 13-point | Nearly isotropic Laplacian – default (`diffusion_stencil = "radius2"`) |

These kernels are generated once on startup and reused everywhere, ensuring that gravitational collapse, fluid migration and diffusion all respect circular symmetry on a 
Cartesian grid.
