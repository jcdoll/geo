# Planet Scenario Instability Analysis

## Summary

The planet scenario fails at grid sizes >= 64x64 due to numerical instabilities arising from extreme density ratios (19,000,000:1 between uranium and space).

## Issues Found and Fixed

1. **Face coefficients not updated after scenario setup**
   - Fixed in simulation.py by adding `self.state.update_face_coefficients()` after scenario setup
   
2. **CFL timestep too large**
   - Old code used gravity wave speed formula inappropriate for our case
   - Fixed by adding gravity acceleration limit: dt < 0.1 * dx / max(gravity)
   - This prevents gravity from causing excessive velocity changes per timestep

3. **Fixed timestep during initialization ramping**
   - Old code used dt=0.001s regardless of gravity magnitude
   - Fixed to use adaptive timestep based on ramped gravity

## Why It Still Fails

Despite these fixes, the simulation remains unstable at large grid sizes because:

1. **Extreme density ratios**: Space (0.001 kg/m³) vs Uranium (19,000 kg/m³) creates a ratio of 19,000,000:1
2. **Numerical precision**: Such extreme ratios push the limits of floating-point arithmetic
3. **Grid size dependency**: Larger grids have stronger gravity fields, exacerbating the instability

## Recommended Solutions

1. **Increase minimum density**: Change space density from 0.001 to 1.0 or higher
2. **Use a compressible solver**: Better suited for extreme density variations
3. **Implement adaptive mesh refinement**: Focus resolution where needed
4. **Use implicit time integration**: More stable for stiff problems

## Test Results

- 16x16: Stable (~6 m/s max velocity)
- 32x32: Unstable (velocities grow to 4000+ m/s)
- 48x48: Very unstable (velocities exceed 10,000 m/s)
- 128x128: Immediate NaN during initialization

The instability manifests as exponentially growing velocities in low-density (space) regions where gravity causes huge accelerations due to the low inertia.