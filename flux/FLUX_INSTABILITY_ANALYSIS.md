# Flux Simulation Instability Analysis

## Executive Summary

The flux-based geological simulation exhibits numerical instabilities when materials with different densities are present. The instability is NOT caused by:
- Poor multigrid solver convergence (tested up to 1e-6 tolerance)
- The specific planet scenario initialization
- Lack of face coefficient updates

Instead, it appears to be a fundamental limitation of the incompressible flow solver when handling density variations.

## Test Results

### Stability vs Density Ratio

| Materials | Density Ratio | Stability | Velocity Growth | Test Script |
|-----------|--------------|-----------|-----------------|-------------|
| Uniform Water | 1:1 | ✅ STABLE | ~0% | `test_clean_water.py` |
| Water/Ice | 1.09:1 | ✅ STABLE | 38% over 20 steps | `test_water_ice.py` |
| Water/Rock | 2.7:1 | ❌ UNSTABLE | 276% over 10 steps | `test_water_rock.py` |
| Planet (Uranium/Space) | 19,000,000:1 | ❌ VERY UNSTABLE | Immediate NaN | `test_planet_debug.py` |

### Key Findings

1. **Threshold**: The stability threshold appears to be between density ratios of 1.09:1 (stable) and 2.7:1 (unstable).

2. **Grid Size Dependency**: Larger grids fail more dramatically:
   - 16x16: Marginally stable (~6 m/s velocities)
   - 32x32: Unstable (velocities grow to 4000+ m/s)
   - 64x64+: Immediate NaN during initialization

3. **Solver Convergence Not the Issue**: Testing with tolerances from 1e-1 to 1e-6 showed identical instability (2.76x growth), indicating the multigrid solver is well-converged.

4. **Initialization Problems**: 
   - Even with gravity disabled, uniform water shows 650 km/s velocities after initialization
   - Pressure gradients are ~7x higher than expected (112 MPa vs 15 MPa theoretical)

## Root Cause Analysis

The instability stems from the incompressible flow formulation with variable density:

1. **Pressure-Density Coupling**: In the projection method, pressure gradients must balance gravity forces: ∇P = ρg. With extreme density variations, small errors in pressure lead to huge velocity errors in low-density regions.

2. **CFL Limitations**: The timestep must satisfy CFL conditions for both high and low density regions. Low-density regions can have very high velocities for the same momentum, requiring tiny timesteps.

3. **Numerical Stiffness**: The ratio of densities creates a stiff system where the condition number of the pressure Poisson equation becomes very large.

## Recommendations

### Short-term Solutions

1. **Increase Minimum Density**
   ```python
   # In materials.py
   props[MaterialType.SPACE] = MaterialProperties(
       density=100.0,  # Increase from 0.001 to 100 kg/m³
       # ... other properties
   )
   ```
   - Reduces density ratio to 190:1
   - May still be insufficient based on water/rock results

2. **Density-Dependent Timestep**
   ```python
   # Add to CFL calculation
   density_ratio = np.max(state.density) / np.min(state.density[state.density > 0])
   dt_density = dx / (gmax * np.sqrt(density_ratio))
   dt = min(dt_advection, dt_gravity, dt_diffusion, dt_density)
   ```

3. **Interface Smoothing**
   - Apply Gaussian smoothing to density field at sharp interfaces
   - Reduces numerical stiffness but may affect physics accuracy

### Long-term Solutions

1. **Compressible Flow Solver**
   - Properly handles large density variations
   - More complex but fundamentally correct for this problem
   - Examples: Low-Mach number approximation, fully compressible Navier-Stokes

2. **Implicit Time Integration**
   - Treat gravity and pressure implicitly
   - Allows larger timesteps with better stability
   - Requires solving coupled system each timestep

3. **Adaptive Mesh Refinement (AMR)**
   - Focus resolution at density interfaces
   - Reduces computational cost while maintaining accuracy
   - Well-suited for planetary simulations

4. **Regularized Incompressible Formulation**
   - Add artificial compressibility: ∂ρ/∂t + ∇·(ρu) = -ε∇²P
   - Smooths pressure field and improves stability
   - Parameter ε controls trade-off between accuracy and stability

## Test Scripts

All test scripts are located in `/home/jcdoll/github/personal/geo/`:

- `test_clean_water.py` - Uniform water test (baseline)
- `test_water_ice.py` - Small density ratio test
- `test_water_rock.py` - Moderate density ratio test
- `test_solver_convergence.py` - Multigrid tolerance analysis
- `test_planet_debug.py` - Planet scenario analysis
- `test_uniform_density.py` - Uniform density with default initialization
- `test_uniform_density_no_gravity.py` - Tests without gravity

## Conclusion

The current incompressible flow solver is fundamentally limited to small density ratios (< 2:1). For geological simulations with realistic material properties, a different numerical approach is needed. The most practical short-term solution is to artificially limit density ratios, but this compromises physical accuracy. A proper long-term solution requires either a compressible flow solver or significant modifications to the current scheme.