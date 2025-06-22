# Performance Analysis: Fluid-Solid Coupling

## Current Performance Baseline
- **Current**: 8 FPS for 100×100 grid (after vectorization optimizations)
- **Target**: 30-60 FPS
- **Gap**: Need 4-8x speedup, not slowdown!

## Full System Performance Impact

### Added Computational Costs

1. **Enhanced Pressure Solve**
   - Current: `∇²P = ∇·(ρg)` - Simple Poisson
   - Proposed: `∇·(1/ρ ∇P) = ∇·v*/Δt + ∇·(v·∇v)` - Variable coefficient with advection
   - **Cost**: ~2x slower (more complex RHS, variable coefficients)

2. **Contact Detection**
   - Need to check all rigid body pairs
   - For N rigid bodies: O(N²) checks
   - **Cost**: Potentially very expensive if many fragments

3. **Displacement Chains**
   - Must trace paths through material grid
   - Worst case: O(grid_size) per movement
   - **Cost**: Could be 10x slower than simple swaps

4. **Interface Tracking**
   - Need to identify all fluid-solid boundaries
   - Update boundary conditions each step
   - **Cost**: ~2x overhead on pressure solve

5. **Coupled System**
   - Can't parallelize sequential steps
   - More iterations for convergence
   - **Cost**: ~1.5x slower than decoupled

### Total Estimated Impact
- **Optimistic**: 5x slower → 1.6 FPS
- **Realistic**: 10x slower → 0.8 FPS
- **Worst case**: 20x slower → 0.4 FPS

**This is clearly unacceptable!**

## Pragmatic Minimal Solution

### Core Problems to Fix
1. Cell deletion (creating SPACE)
2. Unrealistic velocities (1000+ m/s)
3. Lack of volume conservation

### Minimal Fixes (High Impact, Low Cost)

#### 1. Fix Cell Deletion (CRITICAL)
```python
# Instead of:
self.sim.material_types[src_y, src_x] = MaterialType.SPACE  # BAD!

# Use proper swapping:
def swap_cells(src, dst):
    # Always exchange materials, never delete
    temp = material[dst]
    material[dst] = material[src]
    material[src] = temp
```
**Performance Impact**: Neutral (same number of operations)

#### 2. Fix Pressure Unit Bug (DONE)
- Already fixed the 1e6 multiplication error
- **Performance Impact**: None

#### 3. Simple Volume Conservation
```python
def move_rigid_body(body, direction):
    # Find what's in the way
    displaced = find_cells_in_path(body, direction)
    
    # Can only move if we can displace everything
    if can_displace_all(displaced):
        # Execute cyclic permutation
        execute_cyclic_swap(body, displaced)
```
**Performance Impact**: ~2x slower than current (need to check displacement)

#### 4. Velocity Limiting
```python
# Hard clamp to physical limits
max_velocity = min(50.0, 0.1 * cell_size / dt)  # 50 m/s or CFL limit
velocity = np.clip(velocity, -max_velocity, max_velocity)
```
**Performance Impact**: Negligible

### Expected Performance: 4-6 FPS

## Phased Implementation Approach

### Phase 1: Stop the Bleeding (1 week)
- Fix cell deletion bug
- Add velocity clamping
- Simple volume conservation
- **Target**: 4-6 FPS, but correct physics

### Phase 2: Optimize Critical Path (2 weeks)
- Profile and optimize swap operations
- Use spatial hashing for collision detection
- Parallel pressure solve
- **Target**: 10-15 FPS

### Phase 3: Algorithmic Improvements (1 month)
- Adaptive time stepping
- Multi-resolution grids
- GPU acceleration for pressure solve
- **Target**: 20-30 FPS

### Phase 4: Advanced Features (Optional)
- Contact mechanics
- Full incompressible projection
- **Only if** we've hit 30+ FPS baseline

## Alternative: Simplified Physics Mode

For interactive use, offer two modes:

### "Fast" Mode (Default)
- Simple swapping rules
- Basic pressure (hydrostatic only)
- No contact forces
- **Target**: 30-60 FPS

### "Accurate" Mode
- Full fluid-solid coupling
- Contact mechanics
- Incompressible flow
- **Target**: 5-10 FPS (for research/final renders)

## Recommendations

1. **Fix critical bugs first** (cell deletion, unit errors)
2. **Measure after each change** - don't assume impact
3. **Keep the current fast path** for interactive use
4. **Consider GPU acceleration** for pressure solve (biggest bottleneck)
5. **Simplify where possible**:
   - Do we really need contact forces?
   - Can we approximate incompressibility?
   - Is full coupling necessary?

## Conclusion

The full academically-correct system is too slow for interactive use. We should:
1. Fix the critical conservation bugs (minimal performance impact)
2. Keep the current simple system as the default
3. Add accuracy features only as optional modes
4. Focus optimization efforts on the pressure solver (current bottleneck)