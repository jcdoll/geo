# Planet Scenario Mass Conservation Failure - Detailed Analysis

## Summary
The planet scenario fails the mass conservation test because the simulation becomes numerically unstable and produces NaN (Not a Number) values, making it impossible to calculate mass change. The test technically "passes" the 5% threshold because NaN < 0.05 evaluates to False in the comparison, but the simulation is completely broken.

## Visual Analysis

### Initial State (planet_initial.png)
- **Structure**: Circular planet with concentric layers
  - Blue outer region: Space (density 0.001 kg/m³)
  - Orange ring: Air/atmosphere (density ~1.2 kg/m³)
  - Pink layer: Rock crust (density ~2500 kg/m³)  
  - Green patches: Water pools (density 1000 kg/m³)
  - Yellow core: Uranium (density 19,050 kg/m³)
  - Small red spots: Magma chambers (density ~2800 kg/m³)
- **Temperature**: Realistic gradient from hot uranium core (2288K) to cool space (3K)
- **Total mass**: 1.28×10¹⁰ kg
- **Initial velocities**: Already showing NaN!

### After 1 Step (planet_step_001.png)
- **Dominant material**: Only shows space (all other materials vanished!)
- **Density**: Shows NaN everywhere (min=nan, max=nan)
- **Mass**: Changed from 1.28×10¹⁰ kg to "nan kg"
- **All materials disappeared**: The material inventory section is empty

### After 2 Steps (planet_crashed.png)
- Simulation crashed completely
- All fields show NaN
- No recovery possible

## Root Cause Analysis

The failure occurs during initialization due to:

1. **Extreme Density Contrast**: 
   - Space: 0.001 kg/m³
   - Uranium core: 19,050 kg/m³
   - Ratio: 19,050,000:1

2. **Incompressible Flow Assumption**:
   - The pressure projection method assumes incompressible flow
   - With such extreme density differences, the β coefficients (1/ρ) vary by 7 orders of magnitude
   - This creates numerical instability in the Poisson solver

3. **Initialization Shock**:
   - During init step 1: velocity reaches 2,149 m/s
   - During init step 2: velocity explodes to 1.9×10¹⁰ m/s
   - During init step 3: velocity reaches 1.3×10²⁰ m/s (faster than light!)
   - Step 4: Complete NaN propagation

## Why This Happens

The incompressible flow solver tries to enforce ∇·v = 0 everywhere, but:
- Dense uranium wants to fall through ultra-light space
- The solver must create extreme pressure gradients to prevent compression
- These pressure gradients create velocities proportional to β = 1/ρ
- In space regions, β = 1/0.001 = 1000, amplifying any numerical errors
- The multigrid solver experiences overflow in the boundary conditions
- Once NaN appears, it propagates everywhere

## Possible Solutions

1. **Use Compressible Flow**: Allow density to change, avoiding the strict incompressibility constraint
2. **Limit Density Ratios**: Cap the maximum density ratio (e.g., 1000:1)
3. **Different Space Representation**: Use vacuum boundary conditions instead of ultra-low density fluid
4. **Adaptive Timestepping**: Start with extremely small timesteps (< 1e-6 s)
5. **Regularization**: Add numerical damping or viscosity to stabilize the system

## Test Implications

The test checks for < 5% mass change over 100 steps, but:
- The simulation fails catastrophically after 1-2 steps
- Mass becomes NaN (not a number)
- The comparison `abs(nan - initial) / initial < 0.05` returns False
- So the test "passes" even though the simulation is completely broken

This is why the planet scenario is the only failing test - it's the only scenario with such extreme density contrasts in direct contact.