# AIDEV-TODO: CRITICAL BUG - Add Second Term to Self-Gravity Pressure Calculation

## Current State
The pressure solver in `fluid_dynamics.py::calculate_planetary_pressure()` currently omits the second term g_self·∇ρ when computing pressure for self-gravity scenarios. This is a CRITICAL physics bug.

## The Bug
With density variations from granite (2700 kg/m³) to air (1 kg/m³), we have density ratios of 2700:1. The missing g·∇ρ term is enormous at interfaces!

## Missing Implementation
```python
# Current BUGGY code only includes first term for self-gravity:
if self.sim.enable_self_gravity:
    rhs -= (self.sim.density * div_g) / 1e6   # Only ρ∇·g term - MISSING g·∇ρ!

# CORRECT implementation (similar to external gravity case):
if self.sim.enable_self_gravity:
    # First term: ρ∇·g
    rhs -= (self.sim.density * div_g) / 1e6
    
    # CRITICAL MISSING Second term: g·∇ρ
    # This is huge at material interfaces!
    grad_rho_x = (self.sim.density[1:-1, 2:] - self.sim.density[1:-1, :-2]) / (2 * dx)
    grad_rho_y = (self.sim.density[2:, 1:-1] - self.sim.density[:-2, 1:-1]) / (2 * dx)
    rhs -= (gx_total * grad_rho_x + gy_total * grad_rho_y) / 1e6
```

## Physical Impact
- **MASSIVE ERROR**: At granite/air interface, ∇ρ ≈ 2700 kg/m³ per cell!
- Missing pressure jumps of order g·2700 at every rock/air boundary
- Incorrect forces at all material interfaces
- Potentially unstable material configurations

## Implementation Notes
1. Copy the gradient calculation from external gravity section (lines 87-98)
2. Apply to self-gravity case using gx_total, gy_total
3. Test immediately with rock/air interfaces
4. May need smoothing near interfaces to prevent instability

## Priority
**CRITICAL** - This is not an optimization, it's a fundamental physics error affecting all simulations with self-gravity and material interfaces.