# Deprecated: Rigid Body Mechanics

As of 2025-01-21, all rigid body mechanics have been removed from the codebase in favor of a simplified viscosity-based approach where all materials flow at different rates.

## Why Removed?

1. **Performance**: Rigid body mechanics were the main performance bottleneck (376ms per step)
2. **Complexity**: Group identification, displacement logic, and enclosed fluid handling added significant complexity
3. **Scale Issues**: At 50m cell size, rigid bodies don't make physical sense for geological simulation

## New Approach

All materials now flow based on viscosity:
- Space: 0.0 (no resistance)
- Air/vapor: 0.01 (very low resistance)
- Water: 0.05 (flows easily)
- Magma: 0.3 (flows slowly)
- Ice: 0.7 (very slow flow)
- Rocks: 0.85-0.9 (barely flows)

## Performance Improvement

- Before: ~233ms/step (4.3 FPS)
- After: ~77ms/step (12.9 FPS)
- 3x speedup!

## Deprecated Files

The following files contain obsolete rigid body design discussions:
- `FLUID_SOLID_COUPLING_PERFORMANCE.md`
- `FLUID_SOLID_COUPLING_RETHINK.md`
- `fluid_dynamics_original.py` (backup of old implementation)

## Code Changes

1. Removed from `fluid_dynamics.py`:
   - `identify_rigid_groups()`
   - `apply_group_dynamics()`
   - `apply_rigid_body_movements()`
   - `detect_enclosed_fluids()`
   - `attempt_rigid_body_displacement()`
   - All binding force calculations

2. Simplified to:
   - `update_velocities_with_viscosity()`
   - `apply_velocity_movement()`
   - Simple collision handling with momentum conservation

## Test Changes

Removed rigid body tests:
- `test_rigid_body_fall.py`
- `test_rigid_body_fluid_container.py`
- `rigid_body.py` scenarios

All other tests continue to work with the simplified physics.