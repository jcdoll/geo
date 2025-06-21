# Fluid-Solid Coupling Rethink

## Mathematical Framework for Unified Fluid-Solid Dynamics

### Current Equations (from PHYSICS.md)

Our current system uses:

1. **Force Balance (Hydrostatic)**:
   ```
   F = 0 = -∇P + ρg
   ```

2. **Pressure Equation**:
   ```
   ∇²P = ∇·(ρg) = ρ(∇·g) + g·(∇ρ)
   ```

3. **Cell Swapping Rules**:
   - Force test: `|F_net| > F_threshold`
   - Velocity test: `|v_A - v_B| > Δv_threshold`

4. **Incompressible Flow Projection** (recently added):
   ```
   ∇·v = 0
   ∇²P_dynamic = ρ∇·(v·∇v) - ∂(∇·v)/∂t
   ```

### What's Missing

The current approach lacks:
1. **Momentum conservation** during material movement
2. **Volume conservation** constraints
3. **Proper coupling** between fluid and solid phases
4. **Contact mechanics** for rigid body collisions

### Complete Coupled System

#### 1. Fundamental Conservation Laws

**Mass Conservation**:
```
∂ρ/∂t + ∇·(ρv) = 0
```

For incompressible materials (ρ = constant for each material type):
```
∇·v = 0
```

**Momentum Conservation** (Navier-Stokes):
```
ρ(∂v/∂t + v·∇v) = -∇P + ρg + ∇·τ + F_contact
```

Where:
- `τ` = viscous stress tensor
- `F_contact` = contact forces between rigid bodies

**Energy Conservation**:
```
ρc_p(∂T/∂t + v·∇T) = ∇·(k∇T) + Q
```

#### 2. Unified Pressure-Velocity System

Combining incompressibility with momentum:

```
∂v/∂t + v·∇v = -1/ρ ∇P + g + ν∇²v + F_contact/ρ
∇·v = 0
```

Taking divergence of momentum equation:
```
∇·(∂v/∂t) + ∇·(v·∇v) = -∇·(1/ρ ∇P) + ∇·g + ∇·(F_contact/ρ)
```

Since ∇·v = 0 and ∂/∂t(∇·v) = 0:
```
∇·(v·∇v) = -∇·(1/ρ ∇P) + ∇·g + ∇·(F_contact/ρ)
```

This gives our pressure Poisson equation:
```
∇·(1/ρ ∇P) = ∇·(v·∇v) - ∇·g - ∇·(F_contact/ρ)
```

#### 3. Fluid-Solid Interface Conditions

At interfaces between fluid and solid regions:

**No-penetration condition**:
```
v_fluid·n = v_solid·n
```

**Stress continuity** (for viscous fluids):
```
(-P_fluid I + τ_fluid)·n = (-P_solid I + σ_solid)·n
```

Where:
- `I` = identity tensor
- `σ_solid` = solid stress tensor
- `n` = interface normal

#### 4. Rigid Body Dynamics

For rigid bodies, we need:

**Linear momentum**:
```
M dV/dt = ∫∫ (-P·n + τ·n) dS + Mg + F_contact
```

**Angular momentum**:
```
I dω/dt = ∫∫ r × (-P·n + τ·n) dS + T_contact
```

Where:
- `M` = total mass of rigid body
- `V` = velocity of center of mass
- `I` = moment of inertia tensor
- `ω` = angular velocity
- `T_contact` = contact torques

#### 5. Contact Mechanics

When rigid bodies are in contact:

**Non-penetration constraint**:
```
φ(x_A, x_B) ≥ 0
```

Where φ is the signed distance function.

**Contact force**:
```
F_contact = -k_n φ n - c_n (v_rel·n) n - k_t v_rel_tangent
```

Where:
- `k_n` = normal stiffness
- `c_n` = normal damping
- `k_t` = tangential friction coefficient
- `v_rel` = relative velocity at contact point

#### 6. Volume-Conserving Displacement Algorithm

For each timestep:

1. **Predict velocities** from forces:
   ```
   v* = v^n + Δt(g - 1/ρ ∇P^n + F_contact/ρ)
   ```

2. **Identify rigid body movements**:
   ```
   V_rigid = 1/M ∫∫∫_rigid ρv* dV
   ```

3. **Solve coupled pressure system**:
   ```
   ∇·(1/ρ ∇P^{n+1}) = 1/Δt ∇·v* + ∇·(v*·∇v*)
   ```
   
   With boundary conditions:
   - Fluid regions: `v·n = v_rigid·n` at interfaces
   - Solid regions: `v = V_rigid + ω × r`

4. **Correct velocities**:
   ```
   v^{n+1} = v* - Δt/ρ ∇P^{n+1}
   ```

5. **Execute volume-conserving swaps**:
   ```
   For each cell pair (i,j) where swap is needed:
     if can_swap(i,j):
       swap_materials(i,j)
       swap_velocities(i,j)
       swap_temperatures(i,j)
   ```

### Key Differences from Current Approach

1. **Coupled System**: Pressure and velocity are solved together, not sequentially
2. **Material Displacement**: Instead of deleting/creating cells, we track displacement chains
3. **Contact Forces**: Explicit handling of rigid body collisions
4. **Interface Conditions**: Proper boundary conditions at fluid-solid interfaces
5. **Conservation**: Guaranteed conservation of mass, momentum, and energy

### Implementation Requirements

1. **Unified Solver**: Replace separate force/velocity/pressure steps with coupled solve
2. **Displacement Tracking**: Implement displacement chains for volume conservation
3. **Contact Detection**: Add rigid body collision detection and response
4. **Interface Handling**: Track fluid-solid interfaces explicitly
5. **Conservation Checks**: Add validation for mass/momentum/energy conservation

## Current Problems

1. **Cell Deletion**: Rigid body movements create empty space (SPACE material) where there should be displaced fluids/solids
2. **Unrealistic Velocities**: Velocities reaching 1000+ m/s indicate numerical instability
3. **Loss of Conservation**: Total cell count is not conserved - materials are being deleted
4. **Fragmentation**: Solid structures break apart with voids appearing inside them

## Root Causes

1. **Sequential Processing**: Rigid bodies move independently without coordinating with fluid displacement
2. **No Volume Conservation**: When a rigid body moves, it doesn't properly push fluids out of the way
3. **Incompatible Algorithms**: Force-based swapping and rigid body movements operate on different principles
4. **Missing Constraints**: No enforcement that every cell must contain material (no spontaneous void creation)

## Proposed Solution: Unified Volume-Conserving Framework

### Core Principles

1. **Single-Pass Algorithm**: Process all movements in one coordinated pass
2. **Volume Conservation**: Every cell swap must be reciprocal - no cell creation/deletion
3. **Displacement Chains**: Moving a rigid body creates displacement chains that propagate through fluids
4. **Pressure-Driven Flow**: Use pressure gradients to determine fluid displacement directions

### Implementation Approach

#### Phase 1: Unified Movement System

Replace the current multi-step approach with a single unified system:

```python
def apply_unified_dynamics(dt):
    # 1. Calculate all forces (gravity, pressure, etc.)
    forces = calculate_forces()
    
    # 2. Identify movement intentions
    movements = []
    for each cell:
        if is_rigid_body(cell):
            movements.append(rigid_body_movement(cell, forces))
        else:
            movements.append(fluid_movement(cell, forces))
    
    # 3. Resolve conflicts and create displacement chains
    chains = resolve_movement_conflicts(movements)
    
    # 4. Execute all movements atomically
    execute_displacement_chains(chains)
```

#### Phase 2: Displacement Chain Resolution

When a rigid body wants to move into occupied space:

```python
def create_displacement_chain(rigid_body, target_cells):
    chain = [rigid_body]
    
    # Find what needs to be displaced
    for cell in target_cells:
        if is_fluid(cell):
            # Find where fluid can go
            escape_path = find_pressure_gradient_path(cell)
            chain.extend(escape_path)
        elif is_rigid(cell):
            # Check if other rigid body can move
            if can_move(cell):
                sub_chain = create_displacement_chain(cell, ...)
                chain.extend(sub_chain)
            else:
                return None  # Movement blocked
    
    return chain if is_valid_chain(chain) else None
```

#### Phase 3: Pressure-Based Fluid Displacement

Use the pressure field to determine where displaced fluids should go:

```python
def find_pressure_gradient_path(fluid_cell):
    # Follow negative pressure gradient
    neighbors = get_neighbors(fluid_cell)
    
    # Sort by pressure (low to high)
    sorted_neighbors = sort_by_pressure(neighbors)
    
    for neighbor in sorted_neighbors:
        if can_accept_fluid(neighbor):
            return create_path(fluid_cell, neighbor)
    
    # If no direct path, create pressure wave
    return create_pressure_wave_displacement(fluid_cell)
```

### Key Algorithms

#### 1. Atomic Swap Execution

All swaps must maintain conservation:

```python
def execute_swap_chain(chain):
    # Create temporary storage
    temp_materials = []
    temp_properties = []
    
    # Extract all materials in reverse order
    for i in reversed(range(len(chain))):
        temp_materials.append(extract_material(chain[i]))
        temp_properties.append(extract_properties(chain[i]))
    
    # Place materials in forward order
    for i in range(len(chain)):
        place_material(chain[i], temp_materials[i])
        place_properties(chain[i], temp_properties[i])
```

#### 2. Rigid Body Coherence

Ensure rigid bodies move as units:

```python
def move_rigid_body_group(group_id, displacement):
    # Get all cells in group
    cells = get_rigid_group_cells(group_id)
    
    # Check if entire group can move
    displacement_chains = []
    for cell in cells:
        target = cell + displacement
        if not is_empty(target):
            chain = create_displacement_chain(cell, target)
            if chain is None:
                return False  # Movement blocked
            displacement_chains.append(chain)
    
    # Execute all chains atomically
    for chain in displacement_chains:
        execute_swap_chain(chain)
    
    return True
```

#### 3. Incompressible Flow Enforcement

Ensure fluids maintain constant density:

```python
def enforce_incompressibility():
    # Calculate velocity divergence
    div_v = calculate_divergence(velocity_field)
    
    # Solve for pressure correction
    pressure_correction = solve_poisson(div_v)
    
    # Apply velocity correction
    velocity_field -= gradient(pressure_correction)
```

### Discrete Grid Implementation

#### Grid-Based Considerations

On our fixed Eulerian grid (typically 100×100 cells at 50m resolution):

1. **Material Transport**:
   ```
   ∂ρ/∂t + ∇·(ρv) = 0
   ```
   
   In discrete form with upwind differencing:
   ```
   ρ[i,j]^{n+1} = ρ[i,j]^n - Δt/Δx * (F_{i+1/2,j} - F_{i-1/2,j} + F_{i,j+1/2} - F_{i,j-1/2})
   ```
   
   Where fluxes F use upwind values based on velocity direction.

2. **Pressure Gradient** (staggered grid):
   ```
   (∇P)_{i,j} = [(P_{i+1,j} - P_{i-1,j})/(2Δx), (P_{i,j+1} - P_{i,j-1})/(2Δx)]
   ```

3. **Divergence Operator**:
   ```
   (∇·v)_{i,j} = (v_x_{i+1,j} - v_x_{i-1,j})/(2Δx) + (v_y_{i,j+1} - v_y_{i,j-1})/(2Δx)
   ```

#### Hybrid Eulerian-Lagrangian Approach

Since rigid bodies are better tracked in Lagrangian coordinates while fluids work well in Eulerian:

1. **Fluid cells**: Remain on fixed grid, use Eulerian update
2. **Rigid bodies**: Track as connected components with:
   - Center of mass position: `X_cm`
   - Velocity: `V_cm`
   - Angular velocity: `ω`
   - Constituent cells: `{(i,j) | material[i,j] ∈ rigid_body}`

3. **Interface coupling**: 
   ```
   At each fluid cell adjacent to rigid body:
   v_fluid = v_rigid + ω × (x_fluid - X_cm)  (no-slip condition)
   ```

#### Volume-Conserving Cell Exchange Algorithm

Instead of the current approach that creates voids:

```python
def volume_conserving_exchange(src_cells, dst_cells):
    """
    Execute material exchange preserving total volume.
    No SPACE material is created or destroyed.
    """
    # Build exchange graph
    exchanges = []
    
    # For each source cell that wants to move
    for src in src_cells:
        dst = compute_destination(src)
        
        # If destination is occupied, create exchange chain
        if is_occupied(dst):
            chain = find_exchange_chain(src, dst)
            if chain:
                exchanges.append(chain)
        else:
            # Direct move only if truly empty (SPACE)
            if material[dst] == SPACE:
                exchanges.append([src, dst])
    
    # Execute all exchanges atomically
    execute_atomic_exchanges(exchanges)
```

#### Pressure Solver with Moving Boundaries

The pressure equation with moving rigid boundaries:

```
∇·(1/ρ ∇P) = S
```

Where source term S includes:
1. Velocity divergence: `∇·v*/Δt`
2. Advection: `∇·(v*·∇v*)`  
3. Moving boundary contribution: `∇·v_boundary/Δt`

Boundary conditions at rigid body surfaces:
```
∂P/∂n = ρ(a_rigid·n - g·n)
```

Where `a_rigid` is rigid body acceleration.

### Implementation Steps

1. **Create new unified dynamics module** (`unified_dynamics.py`)
2. **Implement volume-conserving exchange algorithm**
3. **Add rigid body tracker with Lagrangian update**
4. **Implement coupled pressure solver with moving boundaries**
5. **Add contact detection and response system**
6. **Extensive testing with conservation validation**

### Expected Benefits

1. **Conservation**: No cells created or destroyed
2. **Stability**: Realistic velocities and pressures
3. **Coherence**: Rigid bodies maintain structure
4. **Physical Realism**: Proper fluid displacement around moving solids

### Testing Strategy

1. **Conservation Tests**: 
   ```python
   assert count_non_space_cells(before) == count_non_space_cells(after)
   ```

2. **Velocity Tests**: 
   ```python
   assert np.max(np.abs(velocity)) < speed_of_sound
   ```

3. **Structure Tests**: Verify rigid body connectivity using flood-fill

4. **Displacement Tests**: Track fluid parcels around moving obstacles

### Performance Considerations

1. **Sparse pressure solver**: Only solve in non-SPACE regions
2. **Hierarchical collision detection**: Spatial hashing for rigid bodies
3. **Adaptive timestepping**: CFL condition based on max velocity
4. **Parallelization**: Domain decomposition for large grids

### Concrete Example: Rock Falling Through Water

To illustrate how this unified system works, consider a granite boulder falling through water:

#### Initial State
- Rock at position (50, 20) with downward velocity
- Water fills region below
- Air above water surface

#### Step 1: Force Calculation
```
Rock cells: F = ρ_rock * g - ∇P ≈ 2700 * 9.81 - ∇P
Water cells: F = ρ_water * g - ∇P ≈ 1000 * 9.81 - ∇P
```

#### Step 2: Velocity Prediction
```
v*_rock = v_rock + Δt * F_rock / ρ_rock
v*_water = v_water + Δt * F_water / ρ_water
```

#### Step 3: Incompressibility Enforcement
The rock moving down requires water to move aside:
```
∇·v = 0 everywhere
```

This creates a pressure field that:
- Is high below the rock (resisting motion)
- Is low beside the rock (allowing water to flow around)

#### Step 4: Pressure Solve
```
∇·(1/ρ ∇P) = ∇·v*/Δt
```

With boundary conditions:
- At rock surface: `v_water·n = v_rock·n` (no penetration)
- At water surface: `P = P_atmospheric`

#### Step 5: Velocity Correction
```
v_rock = v* - Δt/ρ_rock * ∇P
v_water = v* - Δt/ρ_water * ∇P
```

#### Step 6: Volume-Conserving Material Exchange
Instead of creating voids:
```
For each rock cell moving down:
  - Identify water cell being displaced
  - Create exchange chain: rock → water → (sideways/up)
  - Execute all exchanges atomically
```

This ensures:
- No SPACE cells are created
- Total water volume is conserved
- Rock maintains its shape
- Water flows realistically around the rock