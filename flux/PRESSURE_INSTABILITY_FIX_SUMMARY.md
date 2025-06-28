# Pressure Instability Fix Summary

## Problem
The flux-based simulation experiences severe numerical instability when simulating extreme density ratios (uranium: 19,000 kg/m³ vs space: 0.1 kg/m³ = 190,000:1 ratio). This causes:
- NaN/overflow errors in pressure solver
- Exponential velocity growth at material/space interfaces
- Complete simulation failure for grids larger than 32x32

## Root Cause
The standard incompressible projection method assumes uniform or slowly-varying density. With extreme ratios:
1. β = 1/ρ coefficients vary by 5-6 orders of magnitude
2. Small numerical errors get amplified by these extreme coefficients
3. The multigrid solver cannot maintain stability across such variations
4. Gravity ramping during initialization compounds the problem

## Fixes Applied

### 1. Disabled Gravity Ramping
- Changed default `init_gravity_ramp` from True to False in `simulation.py`
- Gravity ramping amplifies errors exponentially with each iteration

### 2. Increased Space Density
- Changed space density from 0.001 to 0.1 kg/m³ in `materials.py`
- Reduced density ratio from 19M:1 to 190K:1

### 3. Selective Incompressibility Enforcement
Implemented in `pressure_solver.py`:
```python
# Smooth transition between incompressible/compressible
rho_normalized = (density - 1.0) / (100.0 - 1.0)
incompressibility_weight = 0.5 * (1 + tanh(3 * (rho_normalized - 0.5)))
rhs *= incompressibility_weight
```

### 4. Beta Value Limiting
Added to `state.py` face coefficient calculation:
```python
# Limit beta in space regions for stability
space_mask = density < 1.0
rho_effective[space_mask] = 0.1  # Gives beta = 10 max
```

### 5. Artificial Viscosity
Enhanced `physics.py` with aggressive damping in space:
- 50% velocity damping per timestep in space regions
- Smooth transition over 0-100 kg/m³ density range
- Additional velocity limiting at interfaces (max 1000 m/s)
- Global safety limit of 5000 m/s

## Results

### Working
- 32x32 grid runs stably with reasonable velocities (~60 m/s)
- No NaN errors
- Materials can fall through space

### Partially Working
- 64x64 and 128x128 grids run but with severe velocity clamping
- All velocities hit the 5000 m/s limit
- Pressure values grow exponentially (10^30 Pa)

### Not Working
- Fundamental instability remains unsolved
- Velocity limiting is a band-aid, not a solution
- Larger grids are effectively unusable

## All-Speed Method Implementation

### Approach
Implemented in `pressure_solver_allspeed.py`:
- Modified divergence constraint to allow compressibility in low-density regions
- Quadratic weighting function for smooth transition (ρ² scaling)
- Added damping term in divergence for space regions
- Enhanced viscous damping with exponential decay (0.1s time constant)
- Uses standard Poisson solver with modified RHS (simpler than full Helmholtz)

### Results
- Small grids (16x16, 24x24): Stable with reasonable velocities (~75 m/s)
- 32x32 grid: Initially stable but becomes unstable after 2-3 steps
- Larger grids (64x64+): Immediate instability

The all-speed method provides better results than selective incompressibility but still doesn't fully resolve the instability.

## Root Cause Analysis

The fundamental issue appears to be:
1. Extreme β = 1/ρ ratios (10^5:1) create numerical stiffness
2. Gravity source term ∇·(βg) becomes huge at material/space interfaces
3. Small errors get amplified exponentially by the extreme coefficients
4. Standard discretization schemes aren't designed for such extreme variations

## Next Steps

The current methods provide partial solutions but don't fully resolve the instability. Options:

1. **Low-Mach Preconditioning**: Industry standard for extreme density ratios, modifies the time derivative terms
2. **Cut-cell or Immersed Boundary Methods**: Treat space as a separate domain rather than extreme density
3. **Alternative Algorithms**: LBM or SPH which don't suffer from pressure-projection instabilities

See PHYSICS_FLUX.md Section 3.1 for detailed descriptions of these methods.

## Low-Mach Preconditioning Implementation

### Approach
Implemented in `pressure_solver_lowmach.py`:
- Computes local Mach number and preconditioning parameter θ = min(1, M²)
- Scales divergence constraint by √θ to relax incompressibility in low-Mach regions
- Adds pressure diffusion in space regions for stabilization
- Uses standard Poisson solver with modified RHS

### Results
- All grid sizes complete 1 stable timestep before becoming unstable
- Shows marginal improvement over standard method
- Velocities and pressures still grow exponentially
- Not sufficient to solve the fundamental instability

## Summary of All Approaches

1. **Selective Incompressibility**: Prevents crashes but requires velocity clamping
2. **All-Speed Method**: Works for small grids (≤24x24) but unstable for larger grids
3. **Low-Mach Preconditioning**: Completes 1 timestep but then becomes unstable

None of these methods fully resolve the extreme density ratio problem.

## Final Recommendations

The MAC staggered grid with finite differences appears fundamentally unsuitable for density ratios of 190,000:1. The extreme β = 1/ρ variations create numerical stiffness that standard discretization schemes cannot handle.

Recommended alternatives:

1. **Cut-cell or Immersed Boundary Methods**: Treat space as a separate domain with explicit interfaces
2. **Lattice Boltzmann Method (LBM)**: Works in velocity space, naturally handles large density ratios
3. **Smoothed Particle Hydrodynamics (SPH)**: Meshless method with no grid-based instabilities
4. **Adaptive Mesh Refinement (AMR)**: Use fine grids only at material/space interfaces

The current projection-based approach has reached its limits for this problem.