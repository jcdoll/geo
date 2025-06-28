# PHYSICS REFERENCE - SMOOTHED PARTICLE HYDRODYNAMICS (SPH)

## Overview

This document describes the Smoothed Particle Hydrodynamics (SPH) approach for simulating planetary geology. SPH naturally handles the extreme density ratios and material interactions that proved problematic in grid-based approaches, while enabling pseudo-rigid body behaviors through cohesive forces.

## Why SPH?

The flux-based Eulerian approach encountered fundamental issues:
- Extreme density ratios (190,000:1) caused numerical instabilities in pressure projection
- MAC staggered grids with β = 1/ρ coefficients became ill-conditioned
- Material/vacuum interfaces required special treatment that broke physics

SPH advantages for this project:
- **No pressure projection**: Pressure computed directly from particle density
- **Natural density handling**: 10^6:1 ratios are routine in SPH
- **Meshless**: No grid means no grid-based instabilities
- **Intuitive**: Particles naturally represent rocks, water, etc.
- **Cohesion**: Can implement tensile/cohesive forces for pseudo-rigid bodies
- **Conservation**: Momentum, mass, and energy naturally conserved

## GOLDEN RULES

- Review AGENTS.md before starting work
- No artificial limits on temperature, pressure, or other physical quantities (except T ≥ 0K)
- No legacy code - this is a clean implementation
- Use dispatcher pattern for multiple physics implementations
- Add traceback logging for error identification
- Test all code changes before considering complete

## Goals

Maintain the original simulation goals with SPH implementation:

**Performance**:
- Target: 30+ FPS for ~10,000 particles on CPU
- Use spatial hashing for O(N) neighbor searches
- Implement symplectic time integration for stability

**Physics-based behaviors**:
- Self-gravity computed from particle distribution
- Material properties determine flow behavior
- Phase transitions based on temperature/pressure
- Emergent phenomena from simple rules

**Key features**:
- Pseudo-rigid bodies through cohesive forces
- Fluid-solid interaction without special cases
- Natural handling of free surfaces and material interfaces
- Heat transfer and phase changes

## Theory

### SPH Fundamentals

SPH approximates continuous fields using discrete particles with smoothing kernels:

```
A(r) ≈ Σⱼ mⱼ/ρⱼ Aⱼ W(r - rⱼ, h)
```

Where:
- `mⱼ` = mass of particle j
- `ρⱼ` = density of particle j
- `Aⱼ` = field value at particle j
- `W` = smoothing kernel
- `h` = smoothing length

### State Variables

Each particle carries:
```
Position:     r = (x, y)           [m]
Velocity:     v = (vx, vy)         [m/s]
Mass:         m                    [kg]
Density:      ρ                    [kg/m³]
Pressure:     P                    [Pa]
Temperature:  T                    [K]
Material:     mat_type             [enum]
Stress:       σ = [[σxx, σxy],     [Pa]
                   [σyx, σyy]]
```

### Governing Equations

#### 1. Continuity (Mass Conservation)

SPH density from neighbors:
```
ρᵢ = Σⱼ mⱼ W(rᵢ - rⱼ, h)
```

Or using continuity equation:
```
dρᵢ/dt = Σⱼ mⱼ (vᵢ - vⱼ) · ∇W(rᵢ - rⱼ, h)
```

#### 2. Momentum Conservation

```
dvᵢ/dt = -Σⱼ mⱼ [Pᵢ/ρᵢ² + Pⱼ/ρⱼ² + Πᵢⱼ] ∇W(rᵢ - rⱼ, h) + gᵢ + Fᵢ/mᵢ
```

Where:
- `Pᵢ/ρᵢ² + Pⱼ/ρⱼ²` = pressure gradient term
- `Πᵢⱼ` = artificial viscosity for shocks
- `gᵢ` = gravitational acceleration
- `Fᵢ` = other forces (cohesion, etc.)

#### 3. Energy Conservation

```
dEᵢ/dt = Σⱼ mⱼ [Pᵢ/ρᵢ² + Pⱼ/ρⱼ² + Πᵢⱼ/2] (vᵢ - vⱼ) · ∇W(rᵢ - rⱼ, h) + Q̇ᵢ
```

Where `Q̇ᵢ` includes heat conduction, radiation, and heat generation.

### Equation of State

For geological materials, we use a stiffened gas EOS:
```
P = K₀[(ρ/ρ₀)^γ - 1] + P₀
```

Where:
- `K₀` = bulk modulus
- `ρ₀` = reference density
- `γ` = adiabatic index (typically 7 for liquids, 3-5 for solids)
- `P₀` = background pressure

For low-density materials (gases):
```
P = ρRT/M
```

### Pseudo-Rigid Body Mechanics

SPH can simulate solid-like behavior through stress tensors:

#### 1. Elastic Stress
For rocks and solid materials:
```
dσᵢⱼ/dt = 2G(ε̇ᵢⱼ - δᵢⱼε̇ₖₖ/3) + Kε̇ₖₖδᵢⱼ
```

Where:
- `G` = shear modulus
- `K` = bulk modulus
- `ε̇ᵢⱼ` = strain rate tensor

#### 2. Cohesive Forces
Particles can stick together through cohesive forces:
```
Fᶜᵒʰᵉˢⁱᵛᵉ = -k_cohesion * (|rᵢⱼ| - r₀) * r̂ᵢⱼ
```

Applied when:
- Same material type
- Distance < cohesion_radius
- Temperature < melting_point

#### 3. Fracture
Materials fracture when stress exceeds yield strength:
```
if |σ| > σ_yield:
    remove cohesive bonds
    add damping to simulate energy loss
```

### Material Properties

Each material type defines:
```
struct MaterialProperties {
    density: f32,              // Reference density [kg/m³]
    bulk_modulus: f32,         // K₀ for EOS [Pa]
    shear_modulus: f32,        // For solids [Pa]
    yield_strength: f32,       // Fracture threshold [Pa]
    cohesion_strength: f32,    // Adhesion force [N]
    cohesion_radius: f32,      // Bonding distance [m]
    viscosity: f32,            // Dynamic viscosity [Pa·s]
    thermal_conductivity: f32, // [W/(m·K)]
    specific_heat: f32,        // [J/(kg·K)]
    melting_point: f32,        // [K]
    boiling_point: f32,        // [K]
    latent_heat_fusion: f32,   // [J/kg]
    latent_heat_vapor: f32,    // [J/kg]
}
```

Example values:
- **Granite**: ρ=2700, K=50GPa, G=30GPa, σ_yield=200MPa
- **Water**: ρ=1000, K=2.2GPa, viscosity=0.001 Pa·s
- **Air**: ρ=1.2, ideal gas EOS
- **Space**: ρ=0.001, no interactions (ghost particles)

### Heat Transfer

SPH heat conduction:
```
dTᵢ/dt = Σⱼ (4mⱼkᵢkⱼ)/(ρᵢρⱼ(kᵢ+kⱼ)) (Tᵢ - Tⱼ) ∇W(rᵢ - rⱼ, h) · r̂ᵢⱼ / |rᵢⱼ|
```

Plus radiation and heat generation terms.

### Self-Gravity

Two approaches:

#### 1. Direct N-body (for small N)
```
gᵢ = -G Σⱼ≠ᵢ mⱼ(rᵢ - rⱼ)/|rᵢ - rⱼ|³
```

#### 2. Tree-code (Barnes-Hut) for large N
- Build quadtree of particles
- Use multipole approximations for distant groups
- O(N log N) complexity

### Phase Transitions

Temperature and pressure-driven transitions:
```
if T > melting_point and material == ROCK:
    material = MAGMA
    T -= latent_heat_fusion / specific_heat
    reduce cohesive bonds
```

## Implementation Strategy

### Core Components

1. **Particle System**
   - Particle data structure (SoA for cache efficiency)
   - Spatial hashing for neighbor searches
   - Dynamic particle management

2. **Physics Modules**
   - Density computation
   - Force calculation (pressure, viscosity, cohesion)
   - Time integration (Leap-frog or Verlet)
   - Thermal evolution

3. **Material System**
   - Material property database
   - Phase transition manager
   - Cohesive bond tracker

4. **Boundary Conditions**
   - Ghost particles for walls
   - Periodic boundaries option
   - Free surface detection

### Algorithm Outline

```python
def sph_step(dt):
    # 1. Update neighbor lists
    build_spatial_hash()
    find_neighbors()
    
    # 2. Compute density
    for particle in particles:
        particle.density = compute_sph_density(particle)
    
    # 3. Compute pressure
    for particle in particles:
        particle.pressure = equation_of_state(particle)
    
    # 4. Compute forces
    for particle in particles:
        f_pressure = compute_pressure_force(particle)
        f_viscous = compute_viscous_force(particle)
        f_cohesive = compute_cohesive_force(particle)
        f_gravity = compute_gravity(particle)
        particle.force = f_pressure + f_viscous + f_cohesive + f_gravity
    
    # 5. Update positions and velocities
    for particle in particles:
        particle.velocity += particle.force / particle.mass * dt
        particle.position += particle.velocity * dt
    
    # 6. Update temperature
    for particle in particles:
        particle.temperature += compute_heat_transfer(particle) * dt
    
    # 7. Handle phase transitions
    check_phase_transitions()
    
    # 8. Apply boundary conditions
    enforce_boundaries()
```

### Performance Optimizations

1. **Spatial Hashing**
   - Grid size = 2 * smoothing_length
   - Hash table with collision lists
   - Only check 27 neighboring cells (3D) or 9 (2D)

2. **Particle Sorting**
   - Sort by spatial hash for cache coherence
   - Use Z-order curve for better locality

3. **Vectorization**
   - Process particle interactions in batches
   - Use SIMD for kernel evaluations

4. **Adaptive Time Stepping**
   - CFL condition: dt < 0.25 * h / c_sound
   - Viscous condition: dt < 0.125 * h² / ν
   - Use smallest required dt

## Advantages Over Grid-Based Methods

1. **Natural Free Surfaces**: No special treatment needed for material interfaces

2. **Extreme Density Ratios**: Handle space (0.001 kg/m³) to uranium (19,000 kg/m³) without issues

3. **Mass Conservation**: Exact by construction (particles have fixed mass)

4. **Intuitive**: Particles directly represent physical materials

5. **Cohesion**: Natural framework for sticky/solid behaviors

6. **No Pressure Projection**: Avoid ill-conditioned linear systems

## Example Scenarios

### 1. Planet Formation
- Start with cloud of particles
- Self-gravity causes collapse
- Heating from compression
- Differentiation by density

### 2. Asteroid Impact
- High-velocity rock particles
- Shock waves through SPH viscosity
- Fracture and ejecta
- Crater formation

### 3. Volcanic Eruption
- Magma particles with low viscosity
- Pressure buildup
- Phase transition to gas
- Explosive dynamics

### 4. Water Cycle
- Evaporation at surface
- Low-density vapor rises
- Condensation at altitude
- Rain/snow falls

## Testing Strategy

1. **Unit Tests**
   - Kernel functions
   - Neighbor finding
   - Force calculations
   - Conservation checks

2. **Integration Tests**
   - Dam break (classic SPH test)
   - Elastic collision
   - Heat diffusion
   - Phase transitions

3. **Scenario Tests**
   - Planet stability
   - Material sorting by density
   - Impact dynamics
   - Thermal evolution

## Next Steps

1. Implement basic SPH solver with water
2. Add solid mechanics for rocks
3. Implement cohesive forces
4. Add thermal physics
5. Implement self-gravity
6. Create material database
7. Build visualization system
8. Optimize performance

## Future Work

### Solar Heating and Radiative Cooling

The flux implementation uses grid-based ray tracing for solar heating and Stefan-Boltzmann radiation for cooling. For SPH, we need to adapt these concepts to particle-based methods:

#### Solar Heating Approaches

1. **Ray Casting Through Particles**
   - Cast rays from sun direction through particle field
   - Use SPH interpolation to compute optical depth along rays
   - Deposit absorbed energy into particles based on absorption coefficients
   - Challenges: Expensive for many particles, need efficient ray-particle intersection

2. **SPH-Based Radiation Transport**
   - Treat radiation as a field quantity interpolated between particles
   - Use SPH gradients to compute radiation flux divergence
   - Similar to heat conduction but with directional component
   - More consistent with SPH framework but less physically intuitive

3. **Hybrid Approach**
   - Use coarse grid for radiation transport
   - Interpolate radiation field to particles
   - Benefits from both approaches but adds complexity

#### Radiative Cooling Implementation

Stefan-Boltzmann cooling is more straightforward:
```
Q_rad = -ε σ A (T^4 - T_space^4)
```

For particles:
- Detect surface particles using neighbor count/density
- Compute effective surface area per particle
- Apply cooling based on temperature difference
- Consider atmospheric absorption (greenhouse effect)

#### Surface Detection for Radiation

Key challenge: Identifying which particles can radiate to space
- Low neighbor count indicates surface
- Density gradient points outward
- Material type boundaries (e.g., rock-air interface)
- Use SPH approximation: `surface_factor = 1 - ρ/ρ_bulk`

#### Implementation Considerations

1. **Performance**: Radiation calculations can be expensive
   - Use spatial acceleration for ray tracing
   - Update radiation less frequently than dynamics
   - Consider simplified models for real-time simulation

2. **Energy Conservation**: Ensure heating/cooling preserves total energy
   - Track energy added by solar heating
   - Track energy removed by radiation
   - Monitor total system energy

3. **Material Properties**: Need additional parameters
   - Solar absorption coefficient (0 for space, ~0.7 for rock)
   - Thermal emissivity (for Stefan-Boltzmann)
   - Optical properties for atmospheric absorption

#### Proposed Module Structure

```python
# sph/physics/solar_heating.py
def apply_solar_heating(particles, sun_angle, solar_constant):
    # Detect surface particles
    # Apply heating based on angle and absorption
    
# sph/physics/radiative_cooling.py  
def apply_radiative_cooling(particles, space_temperature):
    # Detect radiating particles
    # Apply Stefan-Boltzmann cooling
```