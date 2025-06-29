# PHYSICS REFERENCE - CELLULAR AUTOMATA (DEPRECATD)

This document serves as the authoritative reference for all physical processes, laws, and equations implemented in the 2D geological simulation engine.

This document is **DEPRECATED**. This approach was ofund to have fundamental flaws that are address in the new flux based Eulerian approach.

## GOLDEN RULES

- Review AGENTS.md before starting work.

- Do not add artificial limits, e.g. minimum or maximum temperatures, or minimum or maximum temperature changes per step. These are artificial limits that can obscure bugs in the code or true physical phenomena. The only exception is that temperature can not go below absolute zero (0 K). If it does that indicates a problem with the model.

- Do not keep legacy code or legacy interfaces to maintain compatibility. The code for this project is self-contained. There are no external callers of this code.

- Use a dispatcher design pattern to select between multiple physics implementation options.

- When in doubt, add traceback logging so that any error sources are correctly identified.

---

## Overview

This project is a 2D cellular automata simulation of a planet. All processes are based on simplified physical laws. Ad-hoc rules and heuristics are not used. For example, the gravity field is calculated dynamically every step rather than setting it at the center of the simulation. In the following sections we lay out the physical laws we follow and the numerical implementation details.

### Goals

Here are some of the high level goals for our simulation:
- Fast: approximately 30-60 frames per second for a 128x128 simulation area running on a CPU
- Physics-based: all rules should be physics based as much as possible, for example cell swapping should be based on physical rules and not heuristics
- Simple: keep things as simple as possible
- Clear: for each subject, we should define the physical laws and then discuss their numerical implementation

The units for the simulation are meters, kilograms, and seconds (MKS).

### Performance Achievements

Current performance metrics (100x60 grid):
- **Total step time**: ~45ms (22 FPS) after optimizations
- **Movement calculation**: 2.5ms (was 19ms) - 7.7x speedup via vectorization
- **Time series recording**: 17ms (was 20ms) - removed unused planet_radius calculation
- **Overall improvement**: 1.5x speedup from 66ms to 45ms

Key optimizations:
1. **Vectorized movement**: Process cells in batches by movement direction
2. **Pre-allocated buffers**: Reuse memory for movement operations
3. **Removed planet_radius**: Eliminated O(n) distance calculation that wasn't used
4. **Material count caching**: Only recalculate when materials change

### Behavior requirements

The simulation captures these simplified behaviors for fast, fun planetary simulation:

Gravity:
- Gravitational force is computed dynamically using self-gravity
- Materials fall naturally in gravity fields
- No assumptions about "down" - gravity emerges from mass distribution

Fluid mechanics:
- ALL materials flow based on forces and material viscosity
- High viscosity materials (rocks) flow slowly, low viscosity (water) flows quickly
- No rigid bodies - everything flows at geological time scales
- Pressure drives fluid movement naturally

Thermal mechanics:
- Every cell has temperature with realistic heat diffusion
- Heat sources: material-based (uranium), solar radiation
- Heat sinks: radiative cooling to space
- Temperature affects density through thermal expansion

Buoyancy:
- Emerges naturally from pressure gradients and density differences
- Less dense materials rise, denser materials sink
- No special rules - just physics

Water cycle:
- Phase transitions: ice ↔ water ↔ vapor based on temperature/pressure
- Vapor rises due to low density
- Condensation when cooled
- Natural circulation emerges

Material properties:
- Viscosity determines flow resistance (rocks=high, water=low)
- Density drives buoyancy
- Phase transitions at specific temperature/pressure conditions
- Simple per-material rules create complex emergent behavior

Conservation:
- Cell count is exactly conserved (no creation/destruction)
- Mass is not conserved during phase transitions (realistic)
- Reflecting boundaries prevent material loss

Key simplifications for speed:
- No rigid body groups or cohesion
- No surface tension (meaningless at 50m scale)
- Simple velocity-based movement
- Material properties determine all behavior

---


## MATERIAL MOVEMENT MECHANICS

### THEORY

All materials flow based on forces and material properties. NO RIGID BODIES.

**Basic principles:**
- Every material moves according to F = ma
- Viscosity provides resistance to motion (material-dependent damping)
- Simple velocity-based movement with collision rules
- No binding forces or cohesion

**Movement algorithm:**
1. Calculate net force on each cell: F = F_gravity + F_pressure + F_buoyancy
2. Update velocity: v += (F/m) * dt
3. Apply material viscosity damping: v *= (1 - viscosity * dt)
4. Move materials based on velocity
5. Handle collisions with simple momentum conservation

**Material viscosity values:**
- Space: 0.0 (no resistance)
- Air/vapor: 0.005 (very low)
- Water: 0.01 (flows easily)
- Magma: 0.05 (flows with some resistance)
- Sand: 0.1 (granular flow)
- Ice: 0.15 (slow flow)
- Clay: 0.2 (plastic flow)
- Sedimentary rocks: 0.25-0.35 (slow geological flow)
- Igneous/metamorphic rocks: 0.35-0.4 (very slow geological flow)

**Collision rules:**
- When two materials try to occupy same space, higher density wins
- Momentum is conserved in collisions
- No special cases for "solid" vs "fluid" - just viscosity differences

### IMPLEMENTATION

The simplified movement system uses pure velocity-based physics with vectorized operations:

1. **Force calculation**: Gravity, pressure gradients, and buoyancy create forces
2. **Velocity update**: Explicit Euler integration with material damping
3. **Movement**: Cells move based on velocity, handling collisions simply
4. **No rigid bodies**: All materials flow, just at different rates

**Vectorized movement algorithm (7.7x faster):**
1. Calculate displacement for all cells: dx = v * dt / cell_size
2. Group cells by quantized movement dir  ection (e.g., all cells moving (+1, 0))
3. Process each movement group in parallel:
   - Check bounds and target occupancy
   - Perform swaps for valid moves
   - Handle collisions for blocked moves
4. Use pre-allocated buffers to avoid memory allocation

Key implementation details:
- Batch processing reduces Python loop overhead
- Pre-allocated movement buffers for all properties
- Vectorized collision handling for multiple cell pairs
- Movement threshold prevents tiny movements (0.01 cell minimum)


---



## KINEMATIC EQUATION

### THEORY

The simulation evolves velocity v using the total force per unit mass acting on each cell:

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

### IMPLEMENTATION

Forces are first accumulated into float32 arrays `F_x` and `F_y`. With mass `m = ρ V_cell` known per-cell the velocity update is performed in a single NumPy broadcast:

```
v += dt * F / m
```

The method used here is explicit first-order Euler. We only compute the acceleration at our initial acceleration and so there will be some error. There are higher order and implicit methods that can reduce the error.

Options are:

- Explicit Euler
   - First order accurate explicit
   - No velocity clamp is imposed, because friction adds numerical damping to prevent runaway velocity
    - With strong damping, explicit Euler is naturally stable (velocity decays exponentially)
    - Velocity is optionally clamped in `fluid_dynamics.py` (disabled by default)
- Velocity-Verlet
   - Second order accurate, explicit
   - Method
      - Compute acceleration
      - Apply half velocity update
      - Compute acceleration after half velocity update
      - Apply half velocity update
- Heun's Method:
   - Second order accurate, explicit
   - Method
      - Compute acceleration
      - Compute estimated velocity
      - Compute acceleration at estimated velocity
      - Compute velocity from average of the two accelerations
- Semi-implicit Euler

AIDEV-TODO: WE DON'T ACTUALLY USE DISPLACEMENT FOR ANYTHING THOUGH SO WHY COMPUTE IT

---

## FLUID DYNAMICS

### THEORY

We do not handle fluids in any special manner besides having zero binding force. In the future we may consider a full Navier-Stokes fluid dynamics simulation. This section is kept for future reference.

---

## GRAVITATIONAL PHYSICS

### THEORY

The gravitational acceleration field g is obtained from a scalar potential *Φ* that satisfies the Poisson equation for a continuous mass distribution:

```
∇²Φ = 4 π G ρ
```

Once Φ is known, the acceleration acting on each cell is simply

```
g = -∇Φ
```

The density field changes every step (temperature change, cell migration) the Poisson problem must be re-solved frequently, so a fast numerical method is important.

### CENTER OF MASS

Earlier versions of this project used a simple center of mass calculation (COM). That does not capture the system accurately enough for our taste. Details are as follows for reference but this is not currently used:

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

The coordinates are stored in `self.center_of_mass` (floats). All gravity-driven algorithms use the vector pointing from a voxel to this COM as the inward "down" direction. Because density updates every macro-step, large magma bodies or buoyant plumes can shift the COM and slightly re-orient gravity, giving a first-order coupling between thermal/density anomalies and the gravitational field without solving Poisson's equation.

### IMPLEMENTATION

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

### GEOMETRIC MULTIGRID

We use geometric multigrid for several problems, and so we wanted to go through it in detail once in this section.

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

The solver visits the coarsest level once per cycle – like the letter V. This is usually enough when the right-hand-side (density field) is smooth.

An F-cycle is more aggressive:

```
F-cycle (4 levels shown)

L0 → L1 → L2 → L3
       ▲    │
       │    └── back down to L2, relax, then up
       └────────── up to L1, relax, down again
finally back to L0
```

Think of drawing the letter F: you go down to the bottom, part-way back up, down again, then all the way up. This re-visits the coarser grids multiple times, scrubbing out stubborn smooth error that appears when the density field has sharp contrasts.

Why not always use the F-cycle?  It does ~30 % more relaxation work. In practice we monitor the residual; if it stagnates after one V-cycle we switch to an F-cycle for the next step, then fall back once convergence is healthy.

A concise mental model is:
- Jacobi (or red–black Gauss–Seidel) smoothing damps high-frequency error; plain Gauss–Seidel converges roughly twice as fast but is less parallel-friendly.
- Multigrid then transfers the remaining smooth error to coarser levels where it appears high-frequency again and is removed cheaply.
- The V-cycle is the minimal single-pass walk through the hierarchy.
- The F-cycle is a double-scrub that revisits coarse grids for extra smoothing.

## BUOYANCY FORCE

### THEORY

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

### IMPLEMENTATION

Implementation: The buoyancy solver builds a smoothed density field `ρ̃` (Gaussian σ=0.5 cell) to avoid checker-boarding. For every face between cells A and B we compute
```
F_buoy = (ρ̃_B − ρ̃_A) * g_mag * n_hat
```
where `n_hat` is the outward normal from A to B. Equal and opposite forces are added to the per-cell force arrays. This continuous force approach avoids any special-case "swap because lighter" rule—cells move only when the net force exceeds their binding thresholds.

---


## PRESSURE PHYSICS

### THEORY

The pressure field is computed by solving the full Poisson equation with variable density, starting from force balance:


```
F = 0 = -∇P + ρg
```

we can obtain the hydrostatic balance equation:

```
∇P = ρg
```

Taking the divergence and rearranging:

```
∇²P = ∇·(ρg)
```

This must be expanded properly for variable density:
```
∇·(ρg) = ρ(∇·g) + g·(∇ρ)
```

For self-gravity with variable density:
```
∇²P = ρ(∇·g_self) + g_self·(∇ρ)
```
Where ∇·g_self = -4πGρ from the gravitational Poisson equation. 

The second term is significant for our case, because the density variation can be extreme (granite:air = 2700:1).

For external gravity (uniform field), since ∇·g_ext = 0:
```
∇²P = g_ext·(∇ρ)
```

This formulation correctly handles:
- Variable density fields (compositional/thermal variations)
- Arbitrary geometries and material distributions
- Both self-gravity and external gravity contributions
- Proper boundary conditions (P = 0 in vacuum/space cells)

### IMPLEMENTATION

The pressure field is solved using a geometric multigrid method, which is an optimal O(N) algorithm for solving the Poisson equation ∇²P = RHS.

#### What is Multigrid?

The key insight: iterative methods like Gauss-Seidel quickly smooth high-frequency errors but are slow at reducing low-frequency (smooth) errors. Multigrid accelerates convergence by:
1. Smoothing on the fine grid to reduce high-frequency errors
2. Transferring the problem to a coarser grid where low-frequency errors become high-frequency
3. Solving on the coarse grid (where it's cheaper)
4. Transferring the correction back to the fine grid

#### The V-Cycle Algorithm

Our implementation uses a V-cycle, which looks like:
```
Fine grid (100×100):    Smooth → Restrict ↘
                                           ↘
Coarse grid (50×50):                        Smooth → Restrict ↘
                                                               ↘
Coarsest grid (25×25):                                          Solve
                                                               ↗
Coarse grid (50×50):                        Smooth ← Prolong ↗
                                           ↗
Fine grid (100×100):    Smooth ← Prolong ↗
```

#### Components of our Multigrid Solver

1. **Smoother: Red-Black Gauss-Seidel**
   - Updates grid points in a checkerboard pattern (red points, then black points)
   - Each point updated using: `P[i,j] = (P[i±1,j] + P[i,j±1] - h²·RHS[i,j])/4`
   - Chosen because it parallelizes well and has good smoothing properties

2. **Restriction: Full-Weighting** (fine → coarse)
   - Averages 2×2 blocks of fine grid cells to create one coarse cell
   - `P_coarse[i,j] = (P_fine[2i,2j] + P_fine[2i+1,2j] + P_fine[2i,2j+1] + P_fine[2i+1,2j+1])/4`
   - Properly handles odd-sized grids by padding

3. **Prolongation: Bilinear Interpolation** (coarse → fine)
   - Interpolates coarse grid values back to fine grid
   - Direct injection at coincident points
   - Linear interpolation along edges
   - Bilinear interpolation at cell centers

4. **Convergence**: Typically 10-20 V-cycles for 1e-6 relative error

#### Right-Hand Side Construction

Before solving, we must build the RHS of ∇²P = RHS:

1. **Compute gravity field** (if self-gravity enabled via `gravity_solver.py`)
2. **Calculate spatial derivatives** using central differences:
   ```python
   # Divergence of gravity
   div_g = (gx[i,j+1] - gx[i,j-1])/(2*dx) + (gy[i+1,j] - gy[i-1,j])/(2*dx)
   
   # Density gradient  
   grad_rho_x = (rho[i,j+1] - rho[i,j-1])/(2*dx)
   grad_rho_y = (rho[i+1,j] - rho[i-1,j])/(2*dx)
   ```
3. **Build RHS** combining terms:
   - Self-gravity: `RHS = -ρ(∇·g)`
   - External gravity: `RHS = g·∇ρ`

#### Boundary Conditions

- **Dirichlet boundaries**: P = 0 at grid edges (open boundaries)
- **Material boundaries**: No special treatment needed - the variable density in RHS naturally handles interfaces
- **Vacuum/Space**: Forced to P = 0 after solving

#### Why Multigrid is Fast

- **Direct methods** (Gaussian elimination): O(N³) operations
- **Simple iterative** (Gauss-Seidel alone): O(N²) operations  
- **Multigrid**: O(N) operations - optimal!

For a 100×100 grid:
- Direct: ~1,000,000,000 operations
- Gauss-Seidel: ~1,000,000 operations per iteration × many iterations
- Multigrid: ~10,000 operations per V-cycle × 10-20 cycles = ~200,000 total

#### Current Limitations and Design Philosophy

The pressure solver is designed for **dynamic simulations**, not static equilibrium:

1. **Captures pressure-driven flows**: The solver correctly handles buoyancy, density stratification, and material interfaces
2. **Not designed for perfect hydrostatics**: Static fluid columns don't achieve f = 0 equilibrium

**Why hydrostatic equilibrium is imperfect:**

1. **Discrete grid effects**: On coarse grids (50-100m cells), material interfaces are step functions. The pressure equation ∇²P = ∇·(ρg) has delta-function forcing at boundaries, poorly represented on discrete grids.

2. **Inconsistent discretizations**: The Poisson solver uses one discretization of ∇², while force calculation uses a different discretization of ∇. These need to be "adjoint" for perfect conservation.

3. **Interface forces**: At material boundaries (e.g., water/air), the pressure gradient calculation creates spurious forces. Water at interfaces experiences ~13,000 N/m³ instead of the expected 10,000 N/m³.

4. **Theoretical limits**: With sharp density jumps, the vector field ρg is not curl-free at interfaces. This means no exact pressure field satisfies ∇P = ρg everywhere on the discrete grid.

**Design trade-offs:**

- **Speed**: The multigrid solver is O(N) optimal, achieving milliseconds for 128×128 grids
- **Simplicity**: Avoids complex interface treatments like ghost fluid methods
- **Dynamic accuracy**: Correctly captures the physics that matters for geological evolution

**Possible improvements:**

1. **Ghost Fluid Method**: Modify pressure gradients at interfaces to account for density jumps
2. **Consistent discretization**: Ensure gradient operator is adjoint to divergence operator
3. **Higher resolution**: Smoother interfaces reduce numerical artifacts

For now, we accept these limitations. The solver successfully drives the intended physics: heavy materials sink, light materials rise, and the system evolves toward realistic configurations.

### INCOMPRESSIBLE FLOW AND DYNAMIC PRESSURE

For incompressible fluids, the pressure must enforce the divergence-free constraint:

```
∇·v = 0  (incompressibility)
```

This leads to an additional pressure equation:
```
∇²P_dynamic = ρ∇·(v·∇v) - ∂(∇·v)/∂t
```

For steady incompressible flow, this simplifies to:
```
∇²P_dynamic = -ρ∇·(v·∇v)
```

**Why this matters:**
- When materials move through fluids, they displace fluid that must flow around them
- The pressure field must adjust to maintain ∇·v = 0
- This creates dynamic pressure variations beyond the hydrostatic pressure
- Without this term, fluids can artificially compress/expand at boundaries

**Current limitation:** Our pressure solver only computes hydrostatic pressure from gravity. This means:
- Fluids can compress/expand numerically when interacting with moving materials
- Dynamic pressure forces may be underestimated
- High-speed flows lack proper pressure-velocity coupling

**Future implementation:** Add a projection step to enforce incompressibility:
1. Compute provisional velocity from forces
2. Solve for dynamic pressure that makes velocity divergence-free
3. Project velocity: v_new = v_provisional - ∇P_dynamic/ρ

## SURFACE TENSION

Surface tension has been removed from the codebase because it is physically meaningless at geological scales.

### THEORETICAL APPROACHES

1. Continuum Surface Force (CSF) Model: `f = σ κ n |∇c|`
   - σ = surface tension coefficient (0.072 N/m for water)
   - κ = interface curvature
   - n = interface normal
   - c = smoothed color function

2. Young-Laplace Pressure: `ΔP = σ/R` (2D)

3. Energy Minimization: `E = σ ∫ dA`

### DIMENSIONAL ANALYSIS

The key insight is that surface tension becomes negligible at large scales:

- Surface forces: F_surface ∝ σL (scales with perimeter)
- Inertial forces: F_inertia ∝ ρL³ (scales with volume)
- Force ratio: F_inertia/F_surface ∝ L²

At 50m grid cells:
- Scale ratio: ~10¹⁰ times larger than molecular scales
- Inertia dominates surface tension by ~10¹² 

### WHY IT FAILS ON COARSE GRIDS

1. No meaningful curvature: 50m cells create step-function interfaces
2. Discrete materials: Binary WATER/AIR transitions, no smooth interfaces
3. Numerical instability: Curvature calculations on discrete grids produce noise

### RECOMMENDATIONS

- Geological scales (≥50m): Use bulk fluid behavior (gravity, pressure)
- Fine scales (≤1m): Surface tension can be implemented with CSF/level-set methods
- Accept visual artifacts: Fluids may fragment but behave correctly in bulk

## HEAT TRANSFER PHYSICS

### THEORY

Before talking about specific forces, we will outline the theory behind the heat transfer model.

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

The thermal diffusivity is calculated from

```
α = k/(ρ × cp)
```

Where:
- `k` = thermal conductivity (W/(m⋅K))

Diffusivity may be locally enhanced in fluids or at fluid-solid interfaces to model convection.

```
Q_total = Q_internal + Q_solar − Q_radiative
```

## IMPLEMENTATION

This section compares different numerical methods for solving the heat diffusion equation with source terms in the geology simulator.

The challenge is that geological systems have:
1. Large heat sources (solar, internal heating, radiative cooling)
2. Multiple time scales (diffusion: years, sources: seconds)
3. Stability requirements for long-term evolution
4. Performance constraints (real-time visualization)

The simulation now uses operator splitting to solve the heat equation optimally. This approach treats different physical processes separately using their most appropriate numerical methods.

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


## INTERNAL HEATING

### THEORY

Internal heat generation is now entirely material-based, eliminating the previous planetary geometry assumptions. Each material can have its own volumetric heat generation rate.

```
Q_internal = q_material
```

where `q_material` is the heat generation rate (W/m³) specified in the material properties.

This approach:
- Works for arbitrary geometries (not just spherical planets)
- Allows localized heat sources (uranium deposits, radioactive materials)
- Preserves heat generation through the simulation (no dependency on position)

### IMPLEMENTATION

The material-based heating is implemented in `heat_transfer.py::_calculate_internal_heating_source()`:

1. **Material Property Lookup**: Each material type has a `heat_generation` property (W/m³)
2. **Grid Application**: For each cell, the heat generation rate is determined by its material type
3. **Temperature Conversion**: Heat generation is converted to temperature change:
   ```
   dT/dt = q_material / (ρ × cp)
   ```
4. **Time Integration**: The temperature change is multiplied by the timestep

Example materials:
- **Uranium**: 5×10⁻⁴ W/m³ (enhanced for simulation visibility)
- **Regular rocks**: 0 W/m³ (no intrinsic heat generation)
- **Future materials**: Can add other radioactive materials with appropriate heat generation rates

The previous depth-based heating model (crustal and core heating based on distance from planet center) has been completely removed.

---

## SOLAR HEATING

### THEORY

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


### IMPLEMENTATION

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

## RADIATIVE COOLING

### THEORY

Black body radiation removes energy from the planet.

```
Q_radiative = ε σ (T⁴ - T_space⁴)
```

where:
- `ε` is the thermal emissivity of the cell (0–1)
- `σ` = 5.67 × 10⁻⁸ W m⁻² K⁻⁴ is the Stefan-Boltzmann constant
- `T_space` = 2.7 K is the cosmic background temperature

The term is negative in the energy balance and thus acts as a sink in `Q_total`.

### IMPLEMENTATION

---

## ATMOSPHERIC PHYSICS

### THEORY

AIDEV-TODO: ADD DETAILS HERE

The atmospheric layer is approximated as a well-mixed gas column whose state variables (temperature, water-vapour mass) are stored per voxel. Two bulk processes are currently modelled:

1. **Greenhouse trapping** – Outgoing long-wave cooling coefficient σ is reduced by the factor `(1 − g)` where
   `g = base + (max − base) · tanh( vapour_column / scale )`. This couples surface temperature directly to atmospheric humidity.
2. **Enhanced diffusion** – For AIR and WATER_VAPOR cells the numerical viscosity and thermal diffusivity are multiplied by 3× to mimic turbulent eddies that are not resolved on the grid.

These heuristics conserve energy, impose negligible cost, and keep surface gradients physically plausible. A full 2-D Navier–Stokes solve is left for future work.

### IMPLEMENTATION

AIDEV-TODO

Implementation: Each macro-step the engine computes the column water-vapour mass `M_v` for every atmospheric column. The greenhouse factor is then
```
g = base + (max - base) * tanh( ln(1 + M_v/scale) / 10 )
σ_eff = σ * (1 - g)
```
where `base = 0.1`, `max = 0.6`, and `scale = 1 kg m⁻²` are tunable but planet-independent. The modified Stefan-Boltzmann cooling term uses `σ_eff`. Enhanced mixing is applied by multiplying the diffusivity and viscosity arrays by `3` in voxels whose material id is `AIR` or `WATER_VAPOR`.

---

## WEATHERING

### THEORY

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


### IMPLEMENTATION

`_apply_weathering()` iterates over surface-exposed crust cells each macro-step. For each voxel it evaluates:

```
P_chem = dt * k_chem * exp((T - 288 K)/14.4) * water_factor
P_phys = dt * k_freeze * max(0, 1 - |T|/5 K)
P_total = clamp(P_chem + P_phys, 0, 1)
```

A Bernoulli draw with probability `P_total` decides whether the voxel is replaced by the `weathering_product` listed in `MaterialDatabase`. Voxels are conserved during this process.


---

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


### Directional-Sweep Atmospheric Absorption (default)



---

### Solar Heating & Greenhouse Effect
Incoming stellar flux is handled in two stages:
1. Raw insolation – `_calculate_solar_heating_source` projects a solar vector, applies distance factor & cosine-law shading, then multiplies by material albedo.
2. Atmospheric absorption – `_solve_atmospheric_absorption` (directional sweep) attenuates the beam through AIR / WATER_VAPOR columns; absorption coefficient comes from `MaterialDatabase._init_optical_absorption`. 
   *Greenhouse*: the outgoing long-wave cooling constant is multiplied by `(1 – greenhouse_factor)` where
   
  `greenhouse_factor = base + (max-base) * tanh( ln(1+M_vapor/scale) / 10 )`

---

## OTHER IMPLEMENTATION DETAILS

Other important numerical implementation details are noted in this section.

### TIME STEPPING

Adaptive timestep:
- The solver picks `dt = safety * min(dt_diff, dt_cfl)` where `dt_diff = 0.25 Δx² / max(α)` (explicit diffusion limit) and `dt_cfl = 0.5 Δx / max(|v|)`.
- A safety factor of 0.8 gives ample margin.
- If visualisation demands a larger wall-clock frame time the engine runs multiple micro-steps per rendered frame.

### SPATIAL DISCRETIZATION

Grid: Uniform Cartesian with square cells
Cell size: Typically 50 m per cell
Boundary conditions: Insulating (no-flux) at material-space interfaces

### VECTORIZATION

NumPy arrays: All operations vectorized for performance
Morphological operations: Used for fast neighbor calculations
Boolean masking: Efficient material-type specific operations

### CACHING

`materials.py` maintains an LRU cache keyed by `(material_id, T_bin)` for density, heat capacity and optical absorption.
The cache holds ≤10 k entries (<1 MB) and is flushed only when `MaterialDatabase.reload()` is invoked by tests, guaranteeing deterministic physics without memory growth.

### MULTIGRID SMOOTHERS

The Poisson solvers (pressure, velocity projection) use a geometric multigrid V-cycle. We currently employ *red-black Gauss–Seidel* (RB-GS) as the smoother because it damps high-frequency error roughly twice as fast per sweep as weighted Jacobi, particularly when the variable coefficient 1/ρ spans many orders of magnitude (air versus basalt). Any convergent smoother would work – weighted-Jacobi, lexicographic Gauss-Seidel, Chebyshev, even a few conjugate-gradient iterations – the grid hierarchy is unchanged. RB-GS was chosen for code reuse and robustness; swapping in a different smoother only requires a few lines in `pressure_solver.py`.

### SPATIAL KERNELS AND ISOTROPY
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

Any new morphological rule should reuse one of the existing kernels to preserve numerical isotropy.

### CELL CONSERVATION

In almost every numerical update the simulator treats each grid cell as an indestructible voxel – matter is merely moved or its phase changes in-situ. For long-term stability we want all physics operators to preserve the count of MaterialType.SPACE cells (vacuum) unless something explicitly vents gas to space or accretes material from space.

---

## AI DEVELOPMENT NOTES (AIDEV-NOTE)

Important AI development findings are noted below, corresponding to each of the sections above.

### CELL-SWAPPING MECHANICS

1. **Simplified Iteration Approach**: Direct cell-by-cell iteration with neighbor checking is more reliable than complex vectorized slicing operations that can introduce coordinate mapping errors.

2. **Directional Force Logic**: For surface tension and expansion effects, use `abs(proj_src) > src_bind` rather than `proj_src > src_bind` to allow forces pointing away from targets (negative projections).

3. **Velocity Threshold Tuning**: 
   - Original: 0.1 m/s (too high for surface tension effects)
   - Surface tension: 0.001 m/s (allows low-velocity cohesive swaps)
   - Recommendation: Material-dependent thresholds

4. **Binding Force Matrix**: Pre-computed lookup table with temperature scaling:

5. **Material-based movement**: All materials flow based on viscosity - no special rigid/fluid distinction.

### IMMEDIATE ENHANCEMENTS
1. Complete SI sweep – purge any remaining `seconds_per_year` maths in *tests* and documentation examples; delete the placeholder attribute from `simulation_engine_original.py` once reference tests pass.
2. FFT / DST pressure projection – replace the multigrid Poisson solver in `fluid_dynamics.py` with a frequency-space implementation for O(N log N) performance and predictable convergence.
3. Energy conservation regression – add an automated test that steps an isolated closed system for ≥10 years and asserts that total internal energy changes < 0.1 %. This will guard against future source / sink sign errors.
4. Material property cache validation – convert the ad-hoc debug script into a pytest that randomly deletes materials and checks that `_material_props_cache` is perfectly pruned.

### FUTURE ENHANCEMENTS
- Temperature-dependent viscosity: damp velocities as a smooth function of local melt fraction; this replaces the temporary solid drag factor 0.2.
- Variable cell-size support: allow `cell_size` ≠ 50 m so small-scale phenomena (lava tubes, glaciers) can be simulated in separate runs.
- Greenhouse coupling: link water-vapour mass directly to `atmospheric_processes.calculate_greenhouse_effect()` instead of the current heuristic.
- Moist-convective rainfall: precipitate WATER when saturated vapour cools below the Clausius-Clapeyron curve; feeds erosion module.

### VERY LONG TERM ENHANCEMENTS
- Coupled erosion & sediment transport (height-field + fluvial flow).
- Partial melt phase diagram for silicates: returns melt fraction and latent heat sink.
- GPU kernels for heat diffusion and Poisson solves via CuPy (optional acceleration path).
- 3-D extrusion prototype: prove that 2-D solver generalises to shallow-layer quasi-3-D without re-architecting.

---

## PERFORMANCE OPTIMIZATIONS

### Force-Based Swapping Vectorization (2025-01-21)

**Issue**: Unified kinematics taking 376.5ms (78% of total time), preventing 30-60 FPS target.

**Solution**: Vectorized `apply_force_based_swapping()` using numpy array operations:
- Replaced triple-nested loops with shifted array operations
- Process all 4 neighbor directions simultaneously
- Achieved 24.1x speedup (120.5ms → 5.0ms)

**Result**: Overall simulation improved from 4.3 FPS to 8.0 FPS (1.8x speedup).

**Remaining work**:
- Profile other bottlenecks (heat diffusion, gravity solver, pressure solver)
- Consider numba JIT compilation for remaining loops
- Investigate GPU acceleration for field solvers
- Need additional 22+ FPS to reach 30 FPS target
