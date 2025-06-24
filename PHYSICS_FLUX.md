# PHYSICS REFERENCE - FLUX BASED EULERIAN

## Overview

This document describes a flux-based approach for simulating planetary geology that solves the fundamental issues with cellular automata (CA) approaches while maintaining performance targets of 30+ FPS for a 128×128 grid.

## GOLDEN RULES

- Review AGENTS.md before starting work.

- Do not add artificial limits, e.g. minimum or maximum temperatures, or minimum or maximum temperature changes per step. These are artificial limits that can obscure bugs in the code or true physical phenomena. The only exception is that temperature can not go below absolute zero (0 K). If it does that indicates a problem with the model.

- Do not keep legacy code or legacy interfaces to maintain compatibility. The code for this project is self-contained. There are no external callers of this code.

- Use a dispatcher design pattern to select between multiple physics implementation options.

- When in doubt, add traceback logging so that any error sources are correctly identified.

## Goals

TODO: Update this section

The simulation captures these simplified behaviors:

Physics-based:
- Behaviors are based on physical laws and correct theory
- Clamping, artificial limits, ad-hoc methods are not used

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

#### 1. Transport (Mass, Momentum, and Multi-Material)

The foundation of our simulation is the coupled transport of mass, momentum, and multiple materials through the grid. These conservation laws ensure physically correct behavior.

Mass Conservation (Continuity Equation)
```
∂ρ/∂t + ∇·(ρv) = 0
```

In flux form:
```
∂ρ/∂t = -[∂(ρvx)/∂x + ∂(ρvy)/∂y]
```

This ensures that mass is neither created nor destroyed, only transported between cells.

Momentum Conservation
```
∂(ρv)/∂t + ∇·(ρvv) = -∇P + ρg
```

Where g includes both self-gravity and any external gravity fields. This governs how velocity evolves due to pressure gradients, gravity, and advection.

Multi-Material Transport
Each material's volume fraction evolves according to:
```
∂φᵢ/∂t + ∇·(φᵢv) = 0
```

With the constraint: Σφᵢ = 1 (volume fractions sum to unity)

#### Conservative Energy Transport

The thermal energy conservation equation for multi-material flows requires careful treatment:
```
∂(φᵢρᵢcₚᵢT)/∂t + ∇·(φᵢρᵢcₚᵢTv) = 0
```

This must be solved for EACH material separately to ensure exact energy conservation. The final temperature is then computed from:
```
T = Σ(φᵢρᵢcₚᵢT)ᵢ / Σ(φᵢρᵢcₚᵢ)ᵢ
```

**Implementation Note**: A simpler but INCORRECT approach would be to advect the mixed energy density E = ρ_mix·cp_mix·T. This fails catastrophically at material interfaces where thermal mass changes dramatically (e.g., hot rock flowing into cold air). The per-material approach is ~9x slower but physically correct.

**Historical Note**: Early versions used the mixed energy approach for performance, resulting in temperature explosions exceeding 100,000K at material interfaces. The conservative per-material method eliminates these unphysical spikes.

#### 2. Self-Gravity
The gravitational potential Φ satisfies Poisson's equation:
```
∇²Φ = 4πGρ
```

With gravitational acceleration:
```
g = -∇Φ
```

#### 3. Pressure and Incompressibility

The naive approach to pressure would be:

```
F = 0 = -∇P + ρg
∇P = ρg
∇²P = ∇·(ρg) = ρ(∇·g) + g·(∇ρ)
```

But consider a fluid with uniform density and uniform gravity. If we solve `∇²P = 0` there is no guarantee that the pressure will vary linearly with depth to balance gravity, reaching hydrostatic equilibrium. Consider buoyancy. Adding a second material (water over rock) only introduces `g·∇ρ ≠ 0` at one grid-cell row, and otherwise the pressure in the fluid will be constant.

We need a different approach.

Instead, we use the marker-and-cell (MAC) staggered grid velocity projection method - the standard approach in incompressible CFD.

Incompressible Navier-Stokes is:
```
∂v/∂t  +  (v·∇)v  =  – (1/ρ) ∇P  +  g  +  ν ∇²v
∇·v = 0
```

which has components:

- ∂v/∂t
    - local acceleration
    - how fast the velocity at one grid point changes with time
- (v·∇) v
    - convective acceleration
    - velocity change due to fluid flow
    - allows fast moving material to pull along slower moving fluid
- (1/ρ) ∇P
    - pressure gradient force
    - pushes fluid from high pressure to low pressure
    - provides buoyancy
- g
    - body force
    - external acceleration applied everywhere (gravity, Coriois, etc)
    - force that leads to pressure gradient which provides buoyancy
- ν ∇²v
    - viscous diffusion
    - internal friction that smooths out velocity differences

The operators and symbols are:

* v = (vₓ, vᵧ) – velocity vector field in 2-D.
* ∇ – gradient; `∇f = (∂f/∂x, ∂f/∂y)`.
* (v·∇)v – dot product first, then gradient: `vₓ ∂v/∂x + vᵧ ∂v/∂y`.
* ∇² – Laplacian; sum of second derivatives: `∂²v/∂x² + ∂²v/∂y²`.
* ρ(x,y,t) – density (may vary between materials).
* P(x,y,t) – pressure field.
* g – gravity (may vary in magnitude and direction)
* ν – kinematic viscosity (0 → inviscid, large → viscous).

The core idea is:
* Predict velocity with every force except pressure → provisional v★.
* Measure how much ∇·v★ deviates from zero (it should be zero with incompressibility).
* Solve a Poisson-type equation for a scalar φ; its gradient, divided by ρ, removes that deviation in one shot.

In more detail - we perform a forward Euler step with all of the non-pressure forces to obtain:

```
v★ = vⁿ + Δt [ –(v·∇)v + g + ν∇²v ]                (predictor)
```

We now look for a scalar potential (φ) such that:

```
vⁿ⁺¹ = v★ – Δt · β ∇φ                              (corrector)
β = 1/ρ
```

where β is computed at cell faces using harmonic averaging.

To ensure that φ yields an incompressible fluid, we take the divergence and set it to zero (`∇vⁿ⁺¹ = 0`).

```
∇·(β ∇φ) = (1/Δt) ∇·v★                             (Poisson equation)
```

We solve for φ via standard Poisson equation methods (multigrid) and apply the correction to the stored pressure:

```
Pⁿ⁺¹ = Pⁿ + φ                                     (pressure update)
```

In summary, the projection loop is:
1. Predictor: advance cell face velocities with all forces except pressure.
2. Divergence: compute ∇·u* at cell centres.
3. Poisson equation: solve ∇·(β ∇φ)=∇·u*/Δt (β=1/ρ) on cell faces using the harmonic mean.
4. Corrector: subtract β ∇φ from each face velocity, making the new field divergence-free.
5. Update pressure: P ← P + φ.

MAC staggered grid means that we use the velocity components on the faces vs the pressure/density/temperature at the cell centers.

#### 4. Heat Transfer
Energy conservation with advection and diffusion:
```
∂(ρcₚT)/∂t + ∇·(ρcₚTv) = ∇·(k∇T) + Q
```

Where:
- cₚ = specific heat capacity
- k = thermal conductivity  
- Q = heat sources (radioactive decay, etc.)

This is solved using operator splitting:
1. Advection: Solve `∂(ρcₚT)/∂t + ∇·(ρcₚTv) = 0` together with material transport
2. Diffusion: Solve `∂(ρcₚT)/∂t = ∇·(k∇T) + Q` using alternating direction implicit (ADI) or multigrid methods

The advection step conserves total thermal energy E = Σ(φᵢρᵢcₚᵢT) while the diffusion step redistributes it.

#### 5. Solar Heating
Solar radiation provides the primary external energy input:
```
Q_solar = I₀ × (1 - albedo) × absorption_coefficient
```

Where:
- I₀ = incident solar flux (W/m²)
- albedo = material reflectance (0-1)
- absorption_coefficient = material-dependent absorption

#### 6. Radiative Cooling
All materials emit thermal radiation according to Stefan-Boltzmann law:
```
Q_radiative = -ε σ (T⁴ - T_space⁴)
```

Where:
- ε = thermal emissivity (material property)
- σ = 5.67×10⁻⁸ W/(m²·K⁴) (Stefan-Boltzmann constant)
- T = material temperature (K)
- T_space = 2.7 K (cosmic background)

#### 7. Material Phase Transitions
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

### 8. Mixture Properties
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

Cell in-plane dimensions: Δx = Δy = dx
Cell out-of-plane dimensions: Δz = thickness
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
    
    # 2. Update momentum with MAC projection
    # This handles the predictor-corrector split internally:
    # - Predictor: v* = v + dt*(advection + gravity + viscosity)
    # - Corrector: Project v* to divergence-free
    physics.update_momentum(gx, gy, dt)
    
    # 3. Advection (flux-based transport)
    advect_materials_flux(state, dt)
    
    # 4. Thermal diffusion (flux-based)
    diffuse_heat_flux(state, dt)
    
    # 5. Solar heating
    apply_solar_heating(state, solar_angle)
    
    # 6. Radiative cooling
    apply_radiative_cooling(state)
    
    # 7. Phase transitions
    apply_phase_changes(state)
```

### Geometric Multigrid

We use geometric multigrid for several problems, and so we wanted to go through it in detail once in this section.

The key insight: iterative methods like Gauss-Seidel quickly smooth high-frequency errors but are slow at reducing low-frequency (smooth) errors. Multigrid accelerates convergence by:
1. Smoothing on the fine grid to reduce high-frequency errors
2. Transferring the problem to a coarser grid where low-frequency errors become high-frequency
3. Solving on the coarse grid (where it's cheaper)
4. Transferring the correction back to the fine grid

Geometric multigrid attacks low-frequency (smooth) error on coarser grids and high-frequency error on finer grids.

A cycle is the pattern in which the solver moves down (restrict) and up (prolong) through this grid hierarchy.

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

The components in this cycle are:

1. Smoother: Red-Black Gauss-Seidel or Jacobi
   - Updates grid points in a checkerboard pattern (red points, then black points)
   - Gauss-Seidel updates per: `P[i,j] = (P[i±1,j] + P[i,j±1] - h²·RHS[i,j])/4`
   - Chosen because it parallelizes well and has good smoothing properties

2. Restriction: Full-Weighting (fine → coarse)
   - Averages 2×2 blocks of fine grid cells to create one coarse cell
   - `P_coarse[i,j] = (P_fine[2i,2j] + P_fine[2i+1,2j] + P_fine[2i,2j+1] + P_fine[2i+1,2j+1])/4`
   - Properly handles odd-sized grids by padding

3. Prolongation: Bilinear Interpolation (coarse → fine)
   - Interpolates coarse grid values back to fine grid
   - Direct injection at coincident points
   - Linear interpolation along edges
   - Bilinear interpolation at cell centers

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

The F-cycle re-visits the coarser grids multiple times, scrubbing out stubborn smooth error that appears when the density field has sharp contrasts. The F-cycle does ~30 % more relaxation work, so we don't use it by default.

A concise mental model is:
- Jacobi (or red–black Gauss–Seidel) smoothing damps high-frequency error; plain Gauss–Seidel converges roughly twice as fast but is less parallel-friendly.
- Multigrid then transfers the remaining smooth error to coarser levels where it appears high-frequency again and is removed cheaply.
- The V-cycle is the minimal single-pass walk through the hierarchy.
- The F-cycle is a double-scrub that revisits coarse grids for extra smoothing.

Multigrid is fast because ite requires fewer operations:
- Direct methods (Gaussian elimination): O(N³) operations
- Simple iterative (Gauss-Seidel alone): O(N²) operations  
- Multigrid: O(N) operations - optimal!

For a 100×100 grid:
- Direct: ~1,000,000,000 operations
- Gauss-Seidel: ~1,000,000 operations per iteration × many iterations
- Multigrid: ~10,000 operations per V-cycle × 10-20 cycles = ~200,000 total

### Transport Implementation

The transport implementation combines mass conservation, momentum conservation, and multi-material advection into a unified flux-based approach.

Flux is computed at cell faces:

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

Material is advected with the flow:

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

Temperature advection requires conservative treatment of each material's energy:

```python
def advect_temperature_conservative(state, dt, material_fluxes):
    # Store OLD volume fractions before material advection
    old_phi = state.vol_frac.copy()
    
    # Compute initial energy in each cell
    E_initial = sum(old_phi[i] * rho[i] * cp[i] * T for i in materials)
    
    # Compute energy flux for EACH material using same fluxes as volume
    for mat in materials:
        # Energy flux = volume flux * density * cp * T_upwind
        energy_flux_x[mat] = volume_flux_x[mat] * rho[mat] * cp[mat] * T_upwind
        energy_flux_y[mat] = volume_flux_y[mat] * rho[mat] * cp[mat] * T_upwind
    
    # Total energy change from all material fluxes
    E_change = -sum(divergence(energy_flux[mat]) for mat in materials)
    E_final = E_initial + E_change
    
    # NEW thermal mass after materials have moved
    thermal_mass_new = sum(phi_new[i] * rho[i] * cp[i] for i in materials)
    T_new = E_final / thermal_mass_new  # Energy conserved exactly!
```

**Critical**: This MUST use the same material fluxes and be done synchronously with material advection. Using stale volume fractions causes catastrophic errors.

This unified approach ensures:
- Exact mass conservation through flux divergence
- Momentum conservation through velocity advection
- Energy conservation through temperature advection
- Consistent treatment of all transported quantities

### Self-Gravity

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


### Pressure and Incompressibility

Theory and notation were covered earlier; below is only the practical implementation and pitfall.

Solver flow:
1. Predictor – advance face velocities with all forces except pressure
2. Build RHS – compute `(∇·v★)/Δt` at cell centres
3. Poisson – solve `∇·(β ∇φ) = RHS` (β = 1/ρ on faces) with the same multigrid used for gravity
4. Corrector – subtract `Δt β ∇φ` from u, v on faces
5. Pressure update – `P ← P + φ`
6. Move scalars – advect density, temperature, volume fractions, etc.

Minimum data layout (MAC stagger)

| Location on grid          | Stored      |
| ------------------------- | ----------- |
| Cell centre `(i,j)`       | P, ρ, T, φᵢ |
| Vertical face `(i+½,j)`   | u, βₓ (1/ρ) |
| Horizontal face `(i,j+½)` | v, βᵧ (1/ρ) |

Harmonic averaging for β on faces

```python
beta_x[i,j] = 2.0 / (rho[i,j] + rho[i-1,j])
beta_y[i,j] = 2.0 / (rho[i,j] + rho[i,j-1])
```

Pseudo-code:

```python
def advance_one_step(state, dt):
    # ----- 1. predictor ------------------------------------------------------
    # explicit advection + body forces on faces
    ax, ay = convective_accel(state)          # (v·∇)v
    state.u_face += dt * (ax + state.gx)
    state.v_face += dt * (ay + state.gy)
    apply_viscous_diffusion(state, dt)

    # ----- 2. RHS for Poisson ------------------------------------------------
    div_star = divergence_faces_to_cells(state.u_face, state.v_face, state.dx)
    rhs      = div_star / dt                 # (∇·v★)/Δt

    # ----- 3. Poisson solve --------------------------------------------------
    phi = multigrid_poisson(
            rhs,
            beta_x = state.beta_x,           # face-centred 1/ρ
            beta_y = state.beta_y,
            dx     = state.dx,
            bc     = "neumann",              # closed domain default
            tol    = 1e-7)

    # ----- 4. projection -----------------------------------------------------
    gx_phi, gy_phi = gradient_cells_to_faces(phi, state.dx)
    state.u_face -= dt * state.beta_x * gx_phi
    state.v_face -= dt * state.beta_y * gy_phi

    # ----- 5. update pressure & cell velocities ------------------------------
    state.P_cell += phi
    cell_average_velocities(state)           # for output or scalar advection
```

Boundary-conditions:

| Domain edge   | φ BC                | When to use                  |
| ------------- | ------------------- | ---------------------------- |
| Solid wall    | Neumann `∂φ/∂n = 0` | Closed planet crust          |
| Open to space | Dirichlet `φ = 0`   | Top of atmosphere            |

Null-space note: if all faces are Neumann, subtract the mean of φ each V-cycle or the Poisson matrix is singular.

Timestep rule of thumb:

```
Δt ≤ 0.5 · dx / max( |v| , √(g·dx) )      # includes gravity-wave speed
```

Bigger grids or stiffer gravity ⇒ smaller Δt.

Potential pitfalls:

| Symptom                          | Root cause                           | Fix                         |
| -------------------------------- | ------------------------------------ | --------------------------- |
| Checkerboard pattern in pressure | u,v collocated with P                | switch to MAC staggering    |
| Very slow drift in static column | β stored at centres                  | put harmonic β on faces     |
| Ringing at rock/water interface  | arithmetic averaging of β            | harmonic averaging          |
| Poisson stalls or explodes       | null space not removed (Neumann all) | subtract mean φ every cycle |
| Residual divergence after step   | solver tolerance too loose           | tighten tol to ≤ 1 × 10⁻⁷   |

Summary:
* Face-centred β guarantees hydrostatic equilibrium cell-by-cell.
* MAC staggering removes the pressure-velocity decoupling (“checkerboards”).
* One multigrid solve per step keeps cost O(N) and fully re-uses the gravity code.
* Projection enforces `∇·v = 0` to machine precision, so buoyancy emerges entirely from density differences—no hand-coded hacks.

### Heat Transfer

The heat equation combines advective and diffusive transport of thermal energy. We use operator splitting to solve these terms separately for efficiency and stability.

For the diffusion term `∂T/∂t = ∇·(k∇T)`, we have two main approaches:

1. ADI (Alternating Direction Implicit) method
The ADI method splits the 2D diffusion into alternating 1D implicit solves:

```python
# First half-step: implicit in x, explicit in y
for j in range(ny):
    # Solve tridiagonal system for row j
    solve_tridiagonal(a, b, c, d)  # O(nx) per row

# Second half-step: implicit in y, explicit in x  
for i in range(nx):
    # Solve tridiagonal system for column i
    solve_tridiagonal(a, b, c, d)  # O(ny) per column
```

**Advantages:**
- Unconditionally stable (can use large Δt)
- O(N) complexity per timestep
- Well-suited for parabolic PDEs like heat diffusion
- Simple implementation with banded matrix solver

**Disadvantages:**
- Splitting error (though second-order accurate)
- Less efficient for highly anisotropic conductivity

2. Multigrid method

Multigrid solves the implicit system `(I - Δt∇·(k∇))T^{n+1} = T^n + ΔtQ` directly:

```python
# Full implicit discretization leads to linear system
# A * T_new = T_old + dt * Q
# where A = I - dt * L (L is discrete Laplacian)

def multigrid_heat_solver(T_old, dt, k, Q):
    # V-cycle with smoothing at each level
    return multigrid_v_cycle(A, T_old + dt*Q)
```

**Advantages:**
- Handles variable/anisotropic conductivity naturally
- Can achieve higher accuracy
- Scales well to very large grids

**Disadvantages:**
- More complex implementation
- Higher computational cost per timestep (~10x slower than ADI)
- Overkill for simple diffusion problems

3. Explicit Methods (not used)
Standard explicit finite difference: `T^{n+1} = T^n + Δt·α·∇²T^n`

**Why rejected:**
- Severe timestep restriction: Δt < 0.25·Δx²/α (CFL condition)
- For typical parameters: Δt < 0.001s (impractical)
- Would require 1000x more timesteps than implicit methods

4. Spectral Methods (FFT) (not used)
Solve in Fourier space where derivatives become multiplications.

**Why rejected:**
- Requires periodic boundary conditions (unrealistic for geology)
- Cannot handle variable material properties
- Poor performance with mixed materials

Temperature advection is handled together with material transport as described in the Transport Implementation section above.

#### Timestep Selection

The overall timestep is limited by the fastest physical process:

```
Δt = min(
    CFL_advection,     # 0.5 * dx / max(|v|)
    CFL_gravity_wave,  # 0.5 * dx / sqrt(g*H)
    diffusion_limit    # Not needed for implicit methods
)
```

With ADI or multigrid, thermal diffusion doesn't limit the timestep, allowing much larger steps than explicit methods.

### Solar Heating

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

### Radiative Cooling

Radiative cooling implements the Stefan-Boltzmann law for thermal radiation from materials to space. We offer two numerical methods for solving this highly nonlinear cooling term, both integrated within an operator-splitting framework for stability.

#### Mathematical Formulation

The radiative cooling rate follows the Stefan-Boltzmann law noted earlier. There are two methods that we implement for solving the problem.

1. Newton-Raphson Implicit (Default)

Solves the nonlinear equation implicitly:
```
T_new - T_old + dt·α·(T_new⁴ - T_space⁴) = 0
```

Where α = ε·σ / (ρ·cp·thickness)

Implementation:
```python
def solve_radiative_cooling_newton_raphson(T_old, dt, emissivity):
    T_new = T_old  # Initial guess
    for iteration in range(3):  # Usually converges in 2-3 iterations
        f = T_new - T_old + dt*alpha*(T_new**4 - T_space**4)
        df_dT = 1 + dt*alpha*4*T_new**3
        T_new -= f / df_dT
        if abs(f) < tolerance:
            break
    return T_new
```

Advantages:
- Unconditionally stable for any timestep
- Exact solution of Stefan-Boltzmann law
- Handles large temperature differences correctly

Disadvantages:
- Requires 2-3 iterations per cell
- More computationally expensive than linearized method

2. Linearized Stefan-Boltzmann

Approximates the nonlinear cooling term using Taylor expansion around a reference temperature:
```
Q ≈ h·(T - T_space)
```

Where h = 4·σ·ε·T_ref³ is the linearized heat transfer coefficient.

The reference temperature T_ref is chosen dynamically:
```
T_ref = max(T_current, 300K)
```

This choice optimizes linearization accuracy:
- For T > 300K: Uses actual temperature, giving exact derivative at the operating point
- For T < 300K: Uses 300K floor to maintain reasonable cooling rates for cold cells

The 300K floor prevents the heat transfer coefficient from becoming vanishingly small for very cold cells, ensuring meaningful cooling rates even as temperatures approach T_space.

Implementation:
```python
def solve_radiative_cooling_linearized(T_old, dt, emissivity):
    T_ref = max(T_old, 300.0)  # Dynamic reference temperature
    h_effective = 4 * STEFAN_BOLTZMANN * emissivity * T_ref**3
    cooling_rate = h_effective * (T_old - T_space) / (rho * cp * thickness)
    return T_old - dt * cooling_rate
```

Advantages:
- Single calculation per cell (no iterations)
- Very fast execution
- Stable for reasonable timesteps

Disadvantages:
- Less accurate for large temperature differences
- May underestimate cooling at very high temperatures

We modulate the radiative cooling by the greenhouse effect. The atmosphere traps some outgoing radiation based on water vapor content:

```python
def apply_greenhouse_effect(cooling, i, j):
    # Calculate water vapor column above this cell
    vapor_column = compute_vapor_column_above(state, i, j)
    
    # Greenhouse factor: 0.1 (min) to 0.6 (max) absorption
    greenhouse_factor = 0.1 + 0.5 * tanh(log(1 + vapor_column/1.0) / 10)
    
    # Reduce cooling by trapped radiation
    effective_cooling = cooling * (1 - greenhouse_factor)
    return effective_cooling
```

The tanh function provides a smooth transition from minimal greenhouse effect (dry atmosphere) to significant trapping (humid atmosphere).

Both methods are integrated into the operator-splitting framework, ensuring overall stability of the heat equation solver.

### Material Phase Transitions

Material phase transitions implement state changes based on temperature and pressure conditions, with proper handling of latent heat and volume conservation.

The phase transition system uses a rule-based approach where each material can define multiple transition pathways:

```python
@dataclass
class TransitionRule:
    target: MaterialType          # Destination material
    temp_min: float              # Minimum temperature (K)
    temp_max: float              # Maximum temperature (K)  
    pressure_min: float          # Minimum pressure (Pa)
    pressure_max: float          # Maximum pressure (Pa)
    rate: float                  # Transition rate (fraction/second)
    latent_heat: float = 0.0    # J/kg (positive = exothermic, negative = endothermic)
    water_required: bool = False # Requires water presence
    description: str = ""        # Human-readable description
```

The phase transition system processes all materials each timestep:

1. Condition Checking: For each material with defined transitions, check if T-P conditions are met
2. Rate Calculation: Apply transition rate limited by available material
3. Volume Transfer: Move volume fractions between materials
4. Latent Heat: Apply energy changes to maintain conservation
5. Normalization: Ensure volume fractions sum to 1.0

```python
def apply_transitions(state, dt):
    # Check each material's transition rules
    for source_type, rules in transitions.items():
        for rule in rules:
            # Find cells meeting conditions
            mask = (T >= rule.temp_min) & (T <= rule.temp_max) &
                   (P >= rule.pressure_min) & (P <= rule.pressure_max) &
                   (vol_frac[source] > 0)
            
            # Calculate transition amount
            rate = min(rule.rate * dt, 1.0)
            amount = vol_frac[source] * rate
            
            # Apply transition
            vol_frac[source][mask] -= amount[mask]
            vol_frac[target][mask] += amount[mask]
            
            # Apply latent heat
            heat = rule.latent_heat * density[source] * amount
            temperature += heat / (density * specific_heat)
```

The water system demonstrates the full complexity of phase transitions:

Water → Ice (Freezing)
- **Conditions**: T < 273.15K
- **Rate**: 0.1/s (10% per second)
- **Latent heat**: +334 kJ/kg (releases heat)
- **Physics**: Crystallization releases energy, warming surroundings

Ice → Water (Melting)
- **Conditions**: T > 273.15K
- **Rate**: 0.1/s
- **Latent heat**: -334 kJ/kg (absorbs heat)
- **Physics**: Breaking crystal structure requires energy

Water → Water Vapor (Evaporation)
- **Conditions**: T > 373.15K
- **Rate**: 0.05/s (5% per second)
- **Latent heat**: -2,260 kJ/kg (strongly endothermic)
- **Physics**: Phase change to gas requires significant energy

Water Vapor → Water (Condensation)
- **Conditions**: T < 373.15K
- **Rate**: 0.05/s
- **Latent heat**: +2,260 kJ/kg (releases heat)
- **Physics**: Condensation releases large amounts of energy

Simiilarly rock has several transitions.

Rock → Sand (Weathering)
- **Conditions**: Any temperature, P < 100 kPa, water required
- **Rate**: 1e-7/s (geological timescale)
- **Special**: Requires water presence for chemical weathering
- **Physics**: Chemical and mechanical breakdown of rock

Sand → Rock (Lithification)
- **Conditions**: T > 673K, P > 10 MPa
- **Rate**: 1e-8/s (very slow)
- **Physics**: Compaction and cementation under pressure

Rock → Magma (Melting)
- **Conditions**: T > 1473K
- **Rate**: 0.01/s
- **Latent heat**: -400 kJ/kg (endothermic)
- **Physics**: Solid to liquid transition

Magma → Rock (Crystallization)
- **Conditions**: T < 1273K
- **Rate**: 0.01/s
- **Latent heat**: +400 kJ/kg (exothermic)
- **Physics**: Cooling magma crystallizes, releasing heat

Latent heat is carefully tracked to ensure energy conservation:

1. **Endothermic transitions** (melting, evaporation) cool the cell
2. **Exothermic transitions** (freezing, condensation) warm the cell
3. Heat changes are applied based on the actual mass transitioning

This creates realistic thermal effects:
- Ice formation releases heat, slowing further freezing
- Evaporation cools surfaces (evaporative cooling)
- Magma crystallization releases heat, creating thermal aureoles

The system ensures stability through:
- Rate limiting to prevent overshooting (max 100% transition per timestep)
- Volume fraction normalization after all transitions
- Proper handling of competing transitions
- Conservation checks for mass and energy

## Flux vs cellular automata (CA) approaches

Initially this project used a CA approach which had numerous problems.

1. Continuous Mass Transport

CA Problem - cell swapping is binary. This creates:
- Quantization noise in flow
- Unclear criteria for when to swap
- Mass not conserved (different materials have different densities)
- Difficulty achieving hydrostatic equilibrium

Flux solution: transport an arbitrary amount of mass.

```python
# CA approach (problematic)
if should_swap(cell1, cell2):
    swap(cell1, cell2)  # All or nothing!

# Flux approach (correct)
mass_to_move = flux * area * dt  # Continuous amount
cell1.mass -= mass_to_move
cell2.mass += mass_to_move
```

2. Consistent Pressure-Velocity Coupling

CA Problem: Pressure solver and movement rules use different discretizations:
- Pressure solver assumes continuous density field
- Movement rules operate on discrete cells
- Result: Forces don't balance at equilibrium

Flux Solution: Same discretization throughout:
- Pressure solver uses cell-centered values
- Flux computation uses same grid
- Forces exactly balance at equilibrium

3. Exact Conservation

The flux formulation guarantees:
```
Mass leaving cell (i,j) = Mass entering neighbor cells
```

This is exact to machine precision, unlike CA swapping.

4. Natural Multi-Material Handling

CA: Each cell has one material - interfaces are stepped

Flux: Each cell can contain multiple materials with smooth transitions:
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
| L | Enable logging |
| R | Reset to initial simulation state |
| ESC | Exit |
| TAB | Toggle through display modes |

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