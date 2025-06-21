# AIDEV-TODO: Fix Pressure Gradient at Material Interfaces

## Problem
The current pressure gradient calculation creates spurious forces at material interfaces, preventing hydrostatic equilibrium. Water cells at boundaries experience forces of ~13,000 N/m³ instead of the expected 10,000 N/m³.

## Root Cause
The finite difference scheme for pressure gradients doesn't account for density jumps at interfaces. It computes:
```
fy = ρg - ∇P
```

But at interfaces, ∇P should jump to maintain force balance.

## Solution: Ghost Fluid Method
At material interfaces, we need to modify the pressure gradient calculation:

1. For each cell, identify if neighbors have different materials
2. At interfaces, use one-sided differences or interpolate pressure accounting for density jumps
3. Ensure force balance: the pressure gradient should satisfy hydrostatic equilibrium locally

## Implementation Sketch
```python
# Pseudo-code for y-direction
if material[i,j] != material[i+1,j]:  # Interface
    # Use ghost fluid method
    # Pressure is continuous but gradient jumps
    # ∇P should equal ρg on each side
    if in_water:
        fy[i,j] = rho_water * g - one_sided_gradient_water
    else:
        fy[i,j] = rho_air * g - one_sided_gradient_air
else:
    # Normal centered difference
    fy[i,j] = rho * g - centered_gradient
```

## References
- Fedkiw et al. "A Non-oscillatory Eulerian Approach to Interfaces in Multimaterial Flows (The Ghost Fluid Method)"
- Simple approach: Use one-sided differences at interfaces

## Priority
HIGH - This is why fluids aren't in hydrostatic equilibrium. Fixing this will make the simulation much more stable and accurate.