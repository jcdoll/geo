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

#### 4. Pressure and Incompressibility

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

### Pressure Projection Implementation


We solve the Poisson equation using the same multigrid method used for gravity.

Key Implementation Details
1. Face-Centered Coefficients: β lives on faces, not cell centers
2. Staggered Grid: Ideally use MAC grid (velocities on faces)
3. Boundary Conditions: Neumann (∂φ/∂n = 0) for closed domains
4. Null Space: For pure Neumann, remove mean from φ

Common Pitfalls
- Using cell-centered β instead of face-centered → spurious currents
- Forgetting harmonic averaging → pressure oscillations at interfaces  
- Wrong BC type → unphysical flows at boundaries
- Tolerance too loose → residual divergence

The projection method is essential because:
- Automatically handles buoyancy at density interfaces
- Maintains exact incompressibility (∇·v = 0)
- No special cases for uniform/variable gravity
- Standard, well-tested approach used in all modern CFD codes

| Symptom                       | Cause                        | Fix                            |
| ----------------------------- | ---------------------------- | ------------------------------ |
| Checkerboard pressure         | velocities collocated with P | use MAC staggering             |
| Slow drift in “static” test   | β stored at centres          | move β to faces, harmonic mean |
| Pressure ringing at interface | arithmetic averaging of β    | use harmonic average           |
| Solver doesn’t converge       | null space not removed       | subtract mean φ every cycle    |


| ✔ Do this                                        | ✘ Don’t do this                  | Why                                 |
| ------------------------------------------------ | -------------------------------- | ----------------------------------- |
| β on faces, harmonic average                     | β at centres or arithmetic mean  | preserves hydrostatic balance       |
| Pure Neumann BC ⇒ subtract mean φ each V-cycle   | leave φ undefined                | avoids singular matrix              |
| Tight Poisson tolerance (≤10⁻⁷)                  | loose tolerance                  | residual divergence ⇒ creeping flow |
| Keep predictor and corrector on the same stagger | mix collocated & staggered forms | eliminates force imbalance          |

The MAC-grid projection method is implemented as follows:

Here is pseudo-code for the method:


```python
class FluxPhysics:
    def update_momentum(self, gx, gy, dt):
        """MAC-grid projection method for incompressible flow"""
        
        # Update face coefficients (β = 1/ρ with harmonic averaging)
        state.update_face_coefficients()
        state.update_face_velocities_from_cell()
        
        # A. Predictor: v* = v + dt*(advection + gravity + viscosity)
        # Convective acceleration
        ax_conv, ay_conv = self._compute_convective_acceleration()
        state.velocity_x += dt * (ax_conv + gx)
        state.velocity_y += dt * (ay_conv + gy)
        
        # Viscous damping (scaled by dx²)
        self.apply_viscous_damping(dt)
        
        # Update face velocities for projection
        state.update_face_velocities_from_cell()
        
        # B. Projection: solve ∇·(β∇φ) = ∇·v*/Δt
        solver = PressureSolver(state)
        phi = solver.project_velocity(dt, bc_type="neumann")
        # Face velocities corrected inside project_velocity()

class PressureSolver:
    def project_velocity(self, dt, bc_type="neumann"):
        """Project velocity to divergence-free field"""
        
        # Compute divergence of face velocities
        div = self._compute_divergence()  # ∇·v*
        rhs = div / dt
        
        # Solve variable-coefficient Poisson equation
        phi = solve_pressure_variable_coeff(
            rhs, 
            state.beta_x,  # Face-centered 1/ρ (x-faces)
            state.beta_y,  # Face-centered 1/ρ (y-faces)
            state.dx,
            tol=1e-6,
            bc_type=bc_type
        )
        
        # Update face velocities: v = v* - dt*β*∇φ
        self._update_face_velocities(phi, dt)
        
        # Update cell-centered velocities
        state.update_cell_velocities_from_face()
        
        return phi
```

Key Implementation Details

1. Face-Centered β: Computed using harmonic averaging at material interfaces
   ```python
   beta_x[j, i] = 2.0 / (rho[j, i-1] + rho[j, i])  # At vertical faces
   beta_y[j, i] = 2.0 / (rho[j-1, i] + rho[j, i])  # At horizontal faces
   ```

2. Staggered Grid Storage:
   - Cell centers: pressure, density, temperature
   - Vertical faces: x-velocity, beta_x (shape ny, nx+1)
   - Horizontal faces: y-velocity, beta_y (shape ny+1, nx)

3. Boundary Conditions:
    - Neumann (∂φ/∂n = 0) for closed domains
    - Dirichlet (φ = 0) for open domains.

4. CFL Timestep: Includes gravity wave speed
   ```python
   c_grav = sqrt(g * H)  # H = domain height
   dt <= 0.5 * dx / max(|v|, c_grav)
   ```


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