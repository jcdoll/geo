# PHYSICS REFERENCE - 2D Geological Simulation

This document serves as the authoritative reference for all physical processes, laws, and equations implemented in the 2D geological simulation engine.

## GOLDEN RULES

1. Do not add artificial limits, e.g. minimum or maximum temperatures, or minimum or maximum temperature changes per step. These are artificial limits that can obscure bugs in the code or true physical phenomena. The only exception is that temperature can not go below absolute zero (0 K). If it does that indicates a problem with the model.

2. Do not keep legacy code or legacy interfaces to maintain compatibility. The code for this project is self-contained. There are no external callers of this code.

3. Use a dispatcher design pattern to select between multiple physics implementation options.

4. When in doubt, add traceback logging so that any error sources are correctly identified.


## TABLE OF CONTENTS

1. [Kinematic Equation & Force Balance](#kinematic-equation--force-balance)
2. [Cell-Swapping Mechanics](#cell-swapping-mechanics)
3. [Heat Transfer Physics](#heat-transfer-physics)
4. [Gravitational Physics](#gravitational-physics)
5. [Pressure Calculations](#pressure-calculations)
6. [Material Properties & Transitions](#material-properties--transitions)
7. [Fluid Dynamics](#fluid-dynamics)
8. [Atmospheric Physics](#atmospheric-physics)
9. [Solar & Radiative Physics](#solar--radiative-physics)
10. [Geological Processes](#geological-processes)
11. [Units & Constants](#units--constants)
12. [Numerical Methods](#numerical-methods)

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

---

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

---

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

## NUMERICAL METHODS

TODO: Add a brief summary of the equations that we need to solve

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

• Relax (smooth) – perform a few Gauss-Seidel or Jacobi iterations on the current grid to damp the high-frequency error components.
• Restrict (⇩) – project the residual from a fine grid to the next coarser grid, usually by 2× decimation with weighted averaging.
• Prolong (⇧) / correct – interpolate the coarse-grid correction back up to the finer grid and update the solution.

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

### Surface Tension Implementation

**Status**: Implemented and functional, but rate-limited by discrete cell approach.

Surface tension creates cohesive forces that minimize the surface area of fluid-vacuum interfaces. This is implemented through a pressure-based approach that adds internal pressure to fluid cells at interfaces.

#### Physical Model

Surface tension pressure source term:
```
P_surface = σ × N_interfaces × (2/dx)
```

Where:
- `σ` = surface tension coefficient (Pa·m) - currently 50,000 Pa·m for strong discrete effects
- `N_interfaces` = number of neighboring cells with significantly lower density
- `dx` = cell size (m)
- Factor of 2/dx provides correct dimensional scaling (Pa·m / m = Pa)

#### Implementation Details

1. **Interface Detection**: For each fluid cell, count 4-connected neighbors with density < 0.5 × fluid_density
2. **Pressure Source**: Add cohesive pressure proportional to interface exposure
3. **Poisson Solve**: Include surface tension as source term in pressure equation:
   ```
   ∇²P = -ρ∇·g + P_surface_tension/1e6  (Pa → MPa)
   ```
4. **Force Calculation**: Pressure gradients create outward forces at fluid-space boundaries
5. **Force-Based Swapping**: Allow fluid→space swaps when |F_net| > binding_threshold

#### Current Performance

**Working Components**:
- ✅ Surface tension pressure correctly computed (400-600 Pa at interfaces)
- ✅ Strong pressure forces generated (10,000+ N/m³ at boundaries)
- ✅ Force-based swapping detects and executes fluid→space swaps
- ✅ 3 water→space swaps per timestep consistently achieved

**Rate Limitation**:
- ❌ Only 3 swaps per timestep insufficient for dramatic shape change
- ❌ Aspect ratio improvement very gradual (7.50 → 7.75 over 120 steps)
- ❌ Test expects 7.5 → <1.6 aspect ratio in 120 timesteps (unrealistic for discrete approach)

#### Key Technical Insights

1. **Pressure Boundary Conditions**: Must NOT zero out space cell pressures after Poisson solve - this destroys the pressure gradients needed for surface tension forces

2. **Force Direction Logic**: Use `abs(proj_src) > src_bind` rather than `proj_src > src_bind` to allow outward expansion forces (negative projections)

3. **Velocity Threshold**: Reduced from 0.1 m/s to 0.001 m/s to allow low-velocity surface tension swaps

4. **Density Settling Interference**: Traditional density-based settling passes undo surface tension swaps because water >> space density

5. **Discrete vs Continuous**: Real surface tension acts on entire interfaces simultaneously; discrete cell-by-cell swapping is inherently slower

#### Future Improvements Needed

1. **Bulk Interface Processing**: Implement simultaneous swaps across entire fluid-space boundaries rather than individual cells

2. **Integrated Settling Logic**: Modify density-based settling to respect surface tension forces rather than pure density differences

3. **Multi-Cell Expansion**: Allow coordinated expansion patterns that create new interfaces for subsequent swaps

4. **Rate Optimization**: Increase effective swap rate through multiple iterations with pressure recalculation between iterations

#### Force-Based Swapping Integration

Surface tension integrates with the unified force-based swapping system:

```python
# Surface tension only allows fluid→space swaps (outward expansion)
if src_is_fluid and tgt_is_space:
    F_net = hypot(fsrc - ftgt)  # Total force difference
    proj_src = fsrc·direction   # Directional force projection
    
    # Conditions for swap
    cond_force = F_net > binding_threshold(src, tgt)
    cond_direction = abs(proj_src) > binding_threshold(src, reference_solid)
    cond_velocity = |v_src - v_tgt| >= dv_thresh
    
    if cond_force and cond_direction and cond_velocity:
        swap(src, tgt)
```

This replaces the old density-ratio swapping logic with physics-based force criteria that naturally incorporate surface tension effects.

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

---

## FLUID DYNAMICS

### Material Mobility Classification

Gases: Air, Water vapor
Liquids: Water, Magma  
Hot solids: Solid materials with `T > 1200 K`

### Air Migration (Buoyancy)

Direction: Away from planetary center (upward buoyancy)

Migration conditions:
- Target material is porous (`porosity > 0.1`) OR non-solid
- Move toward lower gravitational potential

Migration probability: 30%

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
1. Large heat sources (solar, internal heating, radiative cooling)
2. Multiple time scales (diffusion: years, sources: seconds)
3. Stability requirements for long-term evolution
4. Performance constraints (real-time visualization)

### Current Solution: Operator Splitting Method

The simulation now uses operator splitting to solve the heat equation optimally. This approach treats different physical processes separately using their most appropriate numerical methods.

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

### Method Comparison

#### Current Method: Operator Splitting (Implemented)

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
- ✅ Speed: Fast (near-original performance)
- ✅ Stability: Unconditionally stable (each operator uses optimal method)
- ✅ Accuracy: High accuracy (analytical solutions where possible)
- ✅ Memory: Low memory usage
- ✅ Robust: Each physics process solved optimally

Performance: ~0.95x baseline (5% performance cost for unconditional stability)

### Radiative Cooling Method Selection

The operator splitting approach allows configurable radiative cooling methods via `self.radiative_cooling_method`:

#### Newton-Raphson Implicit (Default: "newton_raphson_implicit")

Implementation: `_solve_radiative_cooling_newton_raphson_implicit()`
- Method: Solves dT/dt = -α(T^4 - T_space^4) using Newton-Raphson iteration
- Advantages: Unconditionally stable, physically accurate, handles large temperature differences
- Disadvantages: More computationally expensive (3-5 iterations typically)
- Stability: Unconditional
- Accuracy: High (exact Stefan-Boltzmann)
- Performance: 1-3 iterations per cell per timestep

#### Linearized Stefan-Boltzmann ("linearized_stefan_boltzmann")

Implementation: `_solve_radiative_cooling_linearized_stefan_boltzmann()`
- Method: Uses Newton cooling law Q = h(T - T_space) where h ≈ 4σεT₀³
- Advantages: Explicit, very stable, fast
- Disadvantages: Approximate, less accurate for large temperature differences
- Stability: Unconditional (when used in operator splitting)
- Accuracy: Good for moderate temperature differences
- Performance: Single calculation per cell per timestep

#### Alternative Method: DuFort-Frankel with Explicit Sources (Previous)

Implementation:
```python
# DuFort-Frankel for full equation
T^(n+1) = T^(n-1) + 2*dt*(α∇²T^n + Q^n/(ρcₚ))
```

Characteristics:
- ✅ Speed: Very fast (1 calculation per timestep)
- ✅ Memory: Low memory usage
- ❌ Stability: Conditionally stable when Q is large
- ❌ Accuracy: Can become unstable with large heat sources

Status: Replaced by operator splitting method

#### Alternative Method: Adaptive Explicit with Full Sub-stepping

Implementation:
```python
# Calculate required substeps
num_substeps = max(1, ceil(dt/dt_stable))
for step in range(num_substeps):
    T = T + dt_sub*(α∇²T + Q/(ρcₚ))
```

Characteristics:
- ✅ Stability: Unconditionally stable
- ✅ Accuracy: High accuracy with adaptive stepping
- ❌ Speed: 10-100x slower (many diffusion calculations)
- ❌ Memory: Higher memory for substeps

Performance: ~0.1x baseline (10x slower)
Status: Too slow for interactive use

## Mathematical Foundation

### Operator Splitting Theory

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

### Why Operator Splitting Works

Unconditional Stability: Each operator uses its most stable numerical method:
- Pure diffusion is much easier to stabilize than diffusion+sources
- Radiative cooling has analytical implicit solutions
- Heat sources are typically well-behaved for explicit integration

Accuracy: Each physical process is solved optimally rather than compromising for a single method

Performance: Avoids the computational cost of treating all processes with the most restrictive (expensive) method

## Implementation Details

### Step 1: Pure Diffusion Solution

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

### Step 2: Radiative Cooling (Configurable Method)

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

### Step 3: Heat Sources (Explicit)

Direct application:
```python
source_change = (Q_internal + Q_solar + Q_atmospheric) * dt / (ρ*cp)
T = T + source_change
```

Well-behaved: Heat sources are typically smooth and bounded

## Performance Comparison

| Method | Relative Speed | Stability | Accuracy | Memory | Status |
|--------|---------------|-----------|----------|---------|---------|
| Operator Splitting | 0.95x | Unconditional | High | Low | ✅ CURRENT |
| DuFort-Frankel Original | 1.0x | Conditional | Medium | Low | ⚠️ Replaced |
| Adaptive Explicit (Full) | 0.1x | Unconditional | High | Medium | ✅ Alternative |

### Typical Performance Characteristics

Operator Splitting Method:
- Diffusion: 1-10 substeps (adaptive based on thermal diffusivity)
- Radiation: 1-3 Newton-Raphson iterations (typically converges in 2)
- Sources: 1 step (explicit, well-behaved)
- Overall: ~5% performance cost for unconditional stability

Substep Requirements:
- Normal conditions: 3-5 diffusion substeps
- High thermal diffusivity: Up to 10 substeps
- Extreme conditions: Automatic adaptation prevents instability

## Advantages of Operator Splitting

### Stability Benefits

Each operator uses its optimal method:
- Pure diffusion: Stable with simple explicit methods
- Radiative cooling: Analytically solvable with Newton-Raphson
- Heat sources: Well-behaved for explicit integration

No compromise methods: Avoids using overly restrictive methods for all processes

### Accuracy Benefits

Physical realism: Each process solved according to its mathematical nature
- Diffusion: Parabolic PDE
- Radiation: Nonlinear algebraic equation
- Sources: Ordinary differential equation

Error control: Adaptive stepping only where needed (diffusion)

### Performance Benefits

Minimal computational overhead: Only 5% slower than original method

Predictable performance: No extreme cases requiring excessive substeps

Memory efficient: No large linear systems or extra storage

## Current Status and Recommendations

### Recommended Method: Operator Splitting (Implemented)

Use for all geological simulations:
- Unconditional stability with minimal performance cost
- Physically realistic treatment of each process
- Suitable for real-time interactive visualization
- No parameter tuning required
- Mathematically sound approach

### Alternative Methods

Full Adaptive Explicit: Use for maximum accuracy research
- Higher computational cost but ultimate accuracy
- Good for validation and benchmarking
- 10x slower but unconditionally stable

Original DuFort-Frankel: Historical reference only
- Replaced due to conditional stability issues
- Could become unstable with large heat sources
- Not recommended for current use

## Future Improvements

### Higher-Order Accuracy
- Strang Splitting: Second-order accurate operator splitting
- Runge-Kutta Integration: Higher-order time integration for sources
- Implicit-Explicit Methods: Combine implicit diffusion with explicit sources

### Advanced Stability
- Richardson Extrapolation: Automatic error estimation
- Embedded Methods: Built-in adaptive error control
- Predictor-Corrector: Multi-step error correction

### Performance Optimization
- Parallel Implementation: Spatial domain decomposition
- GPU Acceleration: Massive parallelization of linear algebra
- Pipelined Operations: Overlap computation phases

## Conclusion

The Operator Splitting Method provides the optimal solution for geological heat transfer:

### Proven Benefits
1. Unconditional stability - each operator solved with its optimal method
2. High accuracy - physically realistic treatment of each process
3. Excellent performance - only 5% slower than original method
4. Mathematical rigor - based on established operator splitting theory
5. Maintenance simplicity - each operator can be improved independently

### Key Innovation
Operator splitting recognizes that different physical processes require different numerical approaches:
- Diffusion: Parabolic PDE requiring careful time stepping
- Radiation: Nonlinear problem with analytical implicit solutions
- Sources: Well-behaved terms suitable for explicit integration

This approach provides the best combination of stability, accuracy, and performance for geological simulation, making it suitable for both research and interactive applications.

## Cell Conservation Exceptions  🚧
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

## Density-Driven Motion and Fluid Dynamics
The simulator separates mass movement into three complementary passes that together honour gravity, buoyancy and fluid behaviour:

### 1. Density-Stratification Pass  (`_apply_density_stratification_local_vectorized`)
* Scope – Operates on *mobile* materials only:  gases (AIR, WATER VAPOR), liquids (WATER), hot solids (> 1200 K), and low-density cold solids (ICE, PUMICE).  
* Rule – Using an isotropic 5 × 5 neighbour list it compares *effective* density (ρ corrected for thermal expansion) between each sampled cell and a neighbour that is one or two cells closer to / farther from the centre of mass.  
* Action – If the outer cell is denser it swaps inward; if lighter it swaps outward.  This creates mantle convection rolls, vapour plumes, and lets ice rise through magma or sink through air as appropriate.


### 2. Unsupported-Cell Settling (`_settle_unsupported_cells`)
* Scope – All solids.  
* Rule – Looks only in the inward gravitational direction (one cell toward COM).  If the destination voxel is a *fluid* (AIR, WATER VAPOR, WATER, MAGMA or even SPACE) and is less dense than the source, the two voxels swap.  
* Outcome – Rockfalls into caves, snowflakes dropping through air, basalt sinking into magma pools.  The lighter fluid rises, preserving mass and space counts.

### 3. Fluid Migration / Vacuum Buoyancy (`_apply_fluid_dynamics_vectorized`)
* Scope – All low-density fluids (AIR, WATER VAPOR, WATER, MAGMA, SPACE).  
* Rule – For each fluid cell adjacent to any non-space material, test neighbours within radius 2. If the neighbour is denser and farther from the surface, swap (Monte-Carlo throttled by `fluid_migration_probability`).  
* Outcome – Magma diapirs, steam bubbles, and trapped vacuum pockets rise toward the planetary surface.

Together these passes realise both behaviours you outlined:
* Hot, ductile mantle rock participates in large-scale convection (Pass 1).
* Any voxel that finds itself resting on something lighter will fall (Pass 2), while light fluids drift upward (Pass 3).

---

## Spatial Kernels & Isotropy
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

## Internal Heating
Geothermal energy is injected every step by `_calculate_internal_heating_source`.
* Exponential depth-dependent profile:  
  `Q = Q0 * exp(-depth / core_heating_depth_scale)`  (W m⁻³).  
* Adds heat explicitly in operator-split Step 3; contributes to `power_density` bookkeeping.

## Solar Heating & Greenhouse Effect
Incoming stellar flux is handled in two stages:
1. Raw insolation – `_calculate_solar_heating_source` projects a solar vector, applies distance factor & cosine-law shading, then multiplies by material albedo.
2. Atmospheric absorption – `_solve_atmospheric_absorption` (directional sweep) attenuates the beam through AIR / WATER_VAPOR columns; absorption coefficient comes from `MaterialDatabase._init_optical_absorption`.  
   *Greenhouse*: the outgoing long-wave cooling constant is multiplied by `(1 – greenhouse_factor)` where
   
  `greenhouse_factor = base + (max-base) * tanh( ln(1+M_vapor/scale) / 10 )`

## Atmospheric Convection
`_apply_atmospheric_convection` performs a simple vertical mixing pass:
* For each AIR or WATER_VAPOR cell, it mixes a fraction `atmospheric_convection_mixing` of the temperature difference with the cell directly above – a cheap way to mimic day-time convection.

## Metamorphism & Phase Transitions
Phase changes are data-driven via `MaterialDatabase`.
* Each `MaterialProperties` entry contains a list of `TransitionRule`(target, T/P window).
* `_apply_metamorphism` scans all non-space cells each macro-step and replaces materials whose local T-P falls inside a rule.
* Melting, crystallisation and gas ↔ liquid ↔ solid transitions (e.g., ICE ⇌ WATER ⇌ VAPOR, MAGMA → BASALT/GRANITE/OBSIDIAN) are all executed in-place – cells retain position & volume.

## Weathering
Surface chemistry is approximated by `_apply_weathering` (optional flag):
* Operates on the outermost crust layer (`surface_radiation_depth_fraction`).
* Converts rocks to their listed weathering products (e.g., GRANITE → SANDSTONE) at a slow stochastic rate, modelling mechanical & chemical erosion.

## Pressure Model
Gravitational lithostatic pressure is recalculated every macro-step by `_calculate_planetary_pressure`:
* Starting at the surface pressure (`surface_pressure`), pressure increments downward with depth using average gravity and density:  
  `ΔP = ρ * g * Δh`.
* Atmospheric pressure decays exponentially with altitude using `atmospheric_scale_height`.
* User-applied tectonic stress is added via `pressure_offset`.

These additions round out the documentation so every major physical subsystem now has a corresponding description in PHYSICS.md.

### Why Three Separate Passes?
Having one monolithic "swap anything with anything" routine would indeed be simpler conceptually, but splitting the work into targeted passes yields a far better speed / accuracy trade-off:

| Pass | Candidate cells (80×80 planet) | Typical samples checked* | Complexity per sample | Dominant memory access |
|------|--------------------------------|-------------------------|-----------------------|------------------------|
| Stratification (1) | Gases, liquids, hot rocks, light solids ≈ 5–10 % | *density_sample_fraction* ≈ 1 000 | ~10 neighbour densities | Sparse, cache-friendly |
| Unsupported settling (2) | All solids but only those directly above a fluid: ≈ 1–2 % | deterministic | 1 density compare | Straight slice, vectorised |
| Fluid migration (3) | AIR/WATER/MAGMA/SPACE ≈ 3 % | *process_fraction_air* ≈ 500 | up to 12 neighbour checks | Contiguous chunks |

\*measured on 80×80 default planet; percentages scale with planet mass.

Performance advantages:
1. Early culling – Each pass quickly masks out ~90 % of the grid that cannot move under that rule, so arithmetic and random-sampling happen on small arrays.
2. Specialised neighbourhoods –  Pass 2 needs only the single voxel inward; Pass 3 needs radius-2 isotropy; Pass 1 needs full 5×5 but just for the sampled mobiles.  A unified pass would have to evaluate the heaviest case for every cell → 5–10× slower.
3. Directional semantics – Unsupported settling is 1-D (*inward only*).  Embedding that into the isotropic swap logic would require extra per-candidate branching and reduce vectorisation.
4. Stronger physical fidelity –  The mantle convection pass allows sideways exchange that would incorrectly mix atmosphere if merged with fluid buoyancy; conversely the fluid-only pass has extra porosity / probability checks irrelevant to rock.

Empirically, profiling shows:
* 3-pass scheme: ~3–4 ms per macro-step on 80×80 grid (Python+NumPy).  
* Single isotropic "swap if heavier" prototype: ~20 ms with identical physics but no early masking.

Hence the current architecture is both faster and clearer, while still producing physically plausible results.  Each pass can be toggled or refined independently without risking cross-coupling bugs.

## Motion Physics Improvements and Recommendations

### Current Limitations Analysis

The existing cell-swapping approach has fundamental limitations that prevent realistic fluid dynamics and rigid body behavior:

1. **Rate-Limited Individual Swaps**: Current implementation achieves only ~3 swaps per timestep, insufficient for phenomena like surface tension-driven shape changes (e.g., water line collapsing to a circle).

2. **No Momentum Conservation**: Cells swap positions without transferring momentum, preventing realistic collision responses and buoyancy oscillations.

3. **Lack of Coherent Motion**: Individual cells move independently without concept of connected groups, preventing rigid body behavior for ice, rock, or other solid structures.

4. **Sequential Processing**: Cell-by-cell evaluation creates artificial ordering dependencies and limits parallelism.

### Recommended Solutions

#### 1. Multi-Cell Group Operations

Identify and move coherent groups of cells as units:

```python
# Identify connected components for rigid materials
groups = connected_component_labeling(material_types)

# Move entire groups based on net forces
for group_id in unique_groups:
    group_mask = (groups == group_id)
    net_force = sum(forces[group_mask])
    group_velocity += net_force / group_mass * dt
    move_group_coherently(group_mask, group_velocity)
```

Benefits:
- Preserves rigid body shapes during motion
- Allows proper momentum transfer between bodies
- Enables realistic iceberg/floating object behavior

#### 2. Bulk Interface Processing

Process entire fluid-vacuum interfaces simultaneously:

```python
# Find ALL interface cells at once
interface_cells = fluid_mask & has_vacuum_neighbor
curvature = calculate_local_curvature(interface_cells)

# Move many cells simultaneously based on curvature
cells_to_move = interface_cells & (curvature > threshold)
bulk_contract_interface(cells_to_move)
```

Benefits:
- Increases swap rate from ~3 to 50-100+ per timestep
- Enables rapid surface tension effects
- More physically accurate interface evolution

#### 3. Velocity Field Integration

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

#### 4. Momentum-Conserving Collisions

Every material exchange must conserve linear momentum:

```python
# Before swap
p1 = m1 * v1
p2 = m2 * v2
p_total = p1 + p2

# After swap (positions exchanged, velocities adjusted)
v1_new = (p_total - m2*v2_old) / m1
v2_new = (p_total - m1*v1_old) / m2
```

### Implementation Strategy

#### Phase 1: Enhanced Surface Tension
- Implement bulk interface processing
- Allow multiple simultaneous swaps per timestep
- Target: 50-100 swaps per timestep for surface tension

#### Phase 2: Velocity Fields
- Add velocity_x, velocity_y arrays
- Update velocities based on force fields
- Use velocity thresholds for swap decisions

#### Phase 3: Group Dynamics
- Implement connected component labeling
- Move rigid bodies as coherent units
- Add inter-group momentum transfer

#### Phase 4: Full Unified Kinematics
- Semi-Lagrangian advection
- Pressure projection for incompressibility
- Complete momentum conservation

### Key Physics Principles

1. **Conservation Laws**: Every operation must conserve mass, momentum, and energy
2. **Collective Behavior**: Connected cells should move together when appropriate
3. **Parallel Processing**: Operations on independent cells should happen simultaneously
4. **Force Integration**: Forces accumulate into velocities, velocities drive motion
5. **Binding Thresholds**: Materials resist motion until forces exceed thresholds

### Expected Improvements

- Surface tension: Water lines collapse to circles in <50 timesteps (vs never)
- Rigid bodies: Ice maintains shape while floating and bobbing
- Collisions: Proper momentum transfer and response
- Performance: Bulk operations more efficient than individual swaps
- Stability: Conservation laws prevent energy/mass drift

## Unified Kinematics: Pressure- and Density-Driven Mass Motion

The previous sections document separate routines for gravitational collapse, density stratification, and fluid migration.  These capture many first-order behaviours but do not yet model:
• lateral flow from pressure gradients (e.g.
  water squirting through a fissure)
• dynamic buoyancy in a *single* momentum framework
• feedback between velocity, pressure, and material state.

This section outlines a single kinematic equation that subsumes those effects while remaining suitable for a cellular-automata engine.

### Governing Momentum Equation (2-D Cartesian grid)
```
∂𝐯/∂t =  -∇P / ρ                         ⏤ pressure-gradient acceleration
         + 𝐠                              ⏤ body-force of gravity (toward COM)
         + ν ∇²𝐯                          ⏤ viscous / numerical diffusion
         + 𝐅_buoyancy                    ⏤ Archimedes term (density contrast)
         + 𝐅_material                    ⏤ material strength & drag
```
Where
• 𝐯(x,y,t)   cell-centred velocity vector (m s⁻¹)  
• P(x,y,t)    scalar pressure field (Pa)            
• ρ(x,y,t)    *effective* density (includes thermal expansion) (kg m⁻³)  
• ν           kinematic viscosity (m² s⁻¹) – piecewise per material  
• 𝐠(x,y)      gravity vector pointing to COM  

Buoyancy is written explicitly:
```
𝐅_buoyancy =  (ρ_ref − ρ) / ρ   · 𝐠
```
with ρ_ref equal to the local average density of the surrounding fluid envelope (air, water, magma, etc.).

For solids, a *drag / rigidity* term suppresses flow so they behave quasi-static:
```
𝐅_material = -k_solid · 𝐯         (k_solid ≫ 1 for competent rock)
```
Liquids and gases set k_solid ≈ 0.

### Pressure Closure (Pseudo-Incompressible)
To stay inexpensive we adopt the pseudo-incompressible assumption (density changes via temperature/phase, not acoustic waves).  Enforcing ∇·𝐯 = 0 yields a Poisson equation each macro-step:
```
∇²P = ρ / Δt · ∇·𝐯* ,             with 𝐯* the provisional velocity without the −∇P term.
```
We solve this with Successive-Over-Relaxation (SOR) or Jacobi iterations until the divergence is below a tolerance (≲10⁻³).

### Discretisation
• Grid spacing Δx = Δy = cell_size (usually 50 m).  
• Central differences for ∇P and ∇²𝐯.  
• Forward Euler or semi-implicit step for viscosity.  
• CFL constraint: Δt ≤ min(Δx / |𝐯|) with a safety factor.

### Boundary Conditions
• Cells bordering SPACE use P = 0 (vacuum).  
• No-slip (𝐯 = 0) at solid boundaries unless cracked/open.  
• Open vents/fissures inherit the neighbour pressure for outflow.

### Expected Behaviours Captured
1. Gravity: body force term.  
2. Low-density rise / high-density sink: buoyancy term.  
3. Fluid outflow / lateral seepage: −∇P / ρ term.  
4. Collapse when support melts: rigidity term drops as T→melt ⇒ 𝐅_material →0 so the object accelerates downward.

---
## Implementation Roadmap & Performance Strategy (≤ 16 ms on 100 × 100)

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
7. Sparse Updates – keep a boolean `active_mask` (cells where |𝐯|, |Ṫ|, or material change > ε).  Only those and their 1-cell halo enter steps 2–5.
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

## Pressure Solver Options & Solver Roadmap

> Current implementation – The planetary pressure is solved with red-black Successive Over-Relaxation (SOR) (see `fluid_dynamics.calculate_planetary_pressure`).  A parity loop (`for parity in (0,1)`) updates the chess-board subsets, so the algorithm is literally classic RB-SOR with over-relaxation factor ω ≈ 1.7.  Because of coarse grids and large density jumps a small *analytic* radial correction (quadratic + linear) is applied after the iterations to enforce a monotonic inward pressure gradient.

The table below compares alternative solvers that would remove that empirical patch while keeping – or improving – performance.

| ID | Solver | Accuracy | Typical Convergence (80×80 grid) | Cost per Step (Python/NumPy) | Pros | Cons |
|----|--------|----------|----------------------------------|------------------------------|------|------|
| A₁ | RB-SOR + patch (status-quo) | ★☆☆ | ~200 sweeps (≈ 1 k iterations per cell) | 4–5 ms | • Very simple  • Works with variable ρ | • Needs empirical patch  • O(N²) sweeps for high accuracy |
| A₂ | RB-SOR + pre-conditioning (Jacobi, Chebyshev) | ★★☆ | 3× faster than A₁ | 2 ms | • Minimal code change | • Still grid-dependent  • Tuning ω / preconditioner |
| B | Geometric Multigrid (V-cycle) | ★★★ | Residual ↓ 10⁻⁶ in 3–4 V-cycles (≈ O(N)) | 1–2 ms | • Grid-independent speed  • Handles variable ρ exactly | • Need hierarchy & prolong/restrict code (≈ 150 LOC) |
| C | FFT / DST Poisson (constant ρ) | ★★★ | Exact (machine precision) in 1 sweep | 0.5 ms | • Blazing fast with `scipy.fft`  • Simple | • Assumes constant ρ – needs Picard loop or damping  • Hard Dirichlet SPACE mask requires padding |
| D | PCG + AMG preconditioner | ★★★ | 10–15 iterations | 1 ms | • Sparse-matrix libraries available (`pyamg`) | • Matrix assembly each step  • Extra dependency |
| E | Self-gravity potential Φ → P | ★★★ | Exact (given Φ) | 1 ms (2 FFTs) | • Physically correct for non-circular planets  • Gives full gravity field | • Requires solving ∇²Φ = 4πGρ  • Adds complexity & memory |

### Recommendation
1. Short term (≤ 1 day) Replace the patch with *RB-SOR + Jacobi pre-conditioner* (option A₂).  Zero refactor risk, immediate residual drop ~3×.
2. Medium term (≤ 1 week) Implement Geometric Multigrid (option B).  Pure-NumPy V-cycle is < 200 LOC and removes grid-size dependence entirely.
3. Long term (R&D) Adopt self-gravity potential workflow (option E).  Gives correct pressure for arbitrary shapes and unlocks tidal / spin effects.  Multigrid can still serve as the Φ- and P-solver if FFT boundaries become awkward.

```text
Roadmap
———
[ v0.9 ]  RB-SOR + Jacobi (drop patch)   → CI residual ↘
[ v1.0 ]  Multigrid V-cycle, variable ρ  → regression tests green, <2 ms P-solve
[ v2.0 ]  Φ-based self-gravity           → full hydrostatic & tidal modelling
```

The current RB-SOR implementation is adequate for gameplay-scale grids, but Multigrid (or FFT where applicable) will give the same answer faster and in a fully theoretical framework – no empirical corrections necessary.

## ROADMAP & OPEN ITEMS

These tasks have been agreed during the refactor sessions but are not yet implemented.  They are listed here so that any contributor can pick them up without digging through chat history.

### Immediate (blocking) 
1. Complete SI sweep – purge any remaining `seconds_per_year` maths in *tests* and documentation examples; delete the placeholder attribute from `simulation_engine_original.py` once reference tests pass.
2. FFT / DST pressure projection – replace the multigrid Poisson solver in `fluid_dynamics.py` with a frequency-space implementation for O(N log N) performance and predictable convergence.
3. Energy conservation regression – add an automated test that steps an isolated closed system for ≥10 years and asserts that total internal energy changes < 0.1 %.  This will guard against future source / sink sign errors.
4. Material property cache validation – convert the ad-hoc debug script into a pytest that randomly deletes materials and checks that `_material_props_cache` is perfectly pruned.

### Short-term enhancements
• Temperature-dependent viscosity – damp velocities as a smooth function of local melt fraction; this replaces the temporary solid drag factor 0.2.
• Variable cell-size support – allow `cell_size` ≠ 50 m so small-scale phenomena (lava tubes, glaciers) can be simulated in separate runs.
• Greenhouse coupling – link water-vapour mass directly to `atmospheric_processes.calculate_greenhouse_effect()` instead of the current heuristic.
• Moist-convective rainfall – precipitate WATER when saturated vapour cools below the Clausius-Clapeyron curve; feeds erosion module.

### Research backlog (nice-to-have)
• Coupled erosion & sediment transport (height-field + fluvial flow).
• Partial melt phase diagram for silicates – returns melt fraction and latent heat sink.
• GPU kernels for heat diffusion and Poisson solves via CuPy (optional acceleration path).
• 3-D extrusion prototype – prove that the 2-D solver generalises to shallow-layer quasi-3-D without re-architecting.

Contributors should update this list (and cross-reference issue numbers) whenever an item is started or completed.

---
