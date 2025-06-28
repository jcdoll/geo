# All-Speed Method Explanation

## Overview

The all-speed method is a numerical technique that allows a single algorithm to handle both incompressible (low Mach number) and compressible (high Mach number) flows. This is exactly what we need for simulating materials (incompressible) coexisting with space/vacuum (compressible).

## The Problem It Solves

Traditional incompressible solvers enforce ∇·v = 0 everywhere, which becomes ill-conditioned when density approaches zero. The all-speed method instead allows controlled compressibility based on local conditions.

## Mathematical Formulation

### Standard Incompressible Projection
```
∇·(β∇φ) = ∇·v*/Δt
```
where β = 1/ρ and we enforce ∇·v = 0 strictly.

### All-Speed Modification
```
∇·(β∇φ) - ε∂φ/∂t = ∇·v*/Δt
```
where ε is a compressibility parameter.

### Key Innovation: Variable Compressibility
```
ε = ε(ρ) = {
    0           for ρ > ρ_incomp  (materials - fully incompressible)
    1/c²        for ρ < ρ_comp   (space - fully compressible)
    smooth transition between
}
```

where c is the local sound speed.

## Physical Interpretation

1. **In Dense Materials (ρ > 100 kg/m³)**
   - ε → 0, so we get standard incompressible flow
   - Pressure waves propagate infinitely fast (incompressible assumption)
   - ∇·v = 0 is enforced

2. **In Space (ρ < 1 kg/m³)**
   - ε → 1/c², allowing acoustic waves
   - Pressure waves propagate at finite speed c
   - ∇·v ≠ 0 is allowed (compression/expansion)

3. **Transition Region (1 < ρ < 100 kg/m³)**
   - Smooth blending prevents numerical shock
   - Gradually transitions from incompressible to compressible

## Implementation Approach

### Step 1: Modify the Poisson Equation
Instead of solving:
```python
∇·(β∇φ) = rhs
```

We solve the Helmholtz equation:
```python
∇·(β∇φ) - ε(ρ)φ = rhs
```

### Step 2: Compute Compressibility Parameter
```python
def compute_epsilon(density, c_sound=340.0):  # m/s, approx speed of sound
    # Define transition thresholds
    rho_incomp = 100.0  # kg/m³ - above this, incompressible
    rho_comp = 1.0      # kg/m³ - below this, compressible
    
    # Normalized density for smooth transition
    rho_norm = (density - rho_comp) / (rho_incomp - rho_comp)
    rho_norm = np.clip(rho_norm, 0, 1)
    
    # Smooth transition using tanh
    compress_factor = 0.5 * (1 - np.tanh(3 * (rho_norm - 0.5)))
    
    # Epsilon varies from 0 (incompressible) to 1/c² (compressible)
    epsilon = compress_factor / (c_sound * c_sound)
    
    return epsilon
```

### Step 3: Solve Modified System
The multigrid solver needs modification to handle the Helmholtz operator:
```python
# In the smoother
residual = (∇·(β∇φ) - ε*φ) - rhs
```

### Step 4: Time-Dependent Pressure
In compressible regions, pressure can change over time:
```python
# After solving for φ
if using_all_speed:
    # In compressible regions, φ represents pressure change rate
    pressure += dt * φ  # Time integration
else:
    # Standard incompressible update
    pressure += φ
```

## Advantages Over Current Approach

1. **Physically Motivated**: Based on actual compressibility physics, not ad-hoc fixes
2. **Smooth Transition**: No discontinuities at material/space interfaces
3. **Stability**: Well-conditioned even with extreme density ratios
4. **Conservation**: Maintains mass/momentum conservation

## Challenges

1. **Helmholtz Solver**: More complex than Poisson (requires modified multigrid)
2. **Sound Speed**: Need reasonable estimates for c in different materials
3. **Time Stepping**: May need implicit time integration in transition regions

## Comparison with Current Selective Incompressibility

### Current Method (Hack)
```python
# Just scale down the divergence constraint
rhs *= incompressibility_weight
```
- Breaks physics in transition regions
- No theoretical foundation
- Still unstable at large scales

### All-Speed Method (Proper)
```python
# Modify the operator itself
∇·(β∇φ) - ε(ρ)φ = rhs
```
- Maintains correct physics
- Smooth, stable transition
- Based on compressible flow theory

## Why Not Just Use Density Clamping?

Your observation is correct - the current β limiting (clamping density to 0.1) is equivalent to just increasing space density. This fails because:

1. **Changes Physics**: Makes space artificially heavy
2. **Moves Problem**: Just shifts instability to different density ratio
3. **Not Scalable**: Breaks down as grid resolution increases

The all-speed method doesn't change the density - it changes how we treat divergence based on the actual density.

## Next Steps for Implementation

1. Modify multigrid solver to handle Helmholtz operator
2. Implement smooth ε(ρ) computation
3. Add time-dependent pressure update in compressible regions
4. Test on planet scenario with original space density (0.001 kg/m³)

This is a significant change but provides a proper solution rather than numerical band-aids.