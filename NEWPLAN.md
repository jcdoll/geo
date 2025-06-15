Q: For my cellular automata simulation of a planet in 2d I need to figure out a simple rule for when cells should swap. My thought is that I can have a binding force matrix, and if the net force on a cell exceeds that force, it moves in the direction of the force. What this misses is any inertia though. What are your thoughts? I have been trying to compute displacement from velocity and acceleration but the problem is if I treat everything as incompressible for simplicity then nothing moves.

Your force-based approach is on the right track, but you've identified the key issues. Here are some strategies to add inertia while allowing movement:

## Allow Slight Compressibility
Instead of strict incompressibility, allow density to vary by small amounts (say ±5-10%). This creates pressure gradients that can drive flow while keeping the simulation stable. You can then have:
- Forces update velocity: `v_new = v_old + (F/mass) * dt`
- Cells swap when velocity difference with neighbors exceeds a threshold

## Momentum Exchange Model
Give each cell a velocity vector and use a two-step process:
1. **Force step**: Update velocities based on pressure gradients, gravity, binding forces
2. **Transport step**: Cells exchange material/swap based on velocity differences

For example, if cell A has velocity pointing toward cell B and `|v_A - v_B| > threshold`, they partially or fully exchange contents.

## Pressure-Velocity Coupling
Calculate local pressure from density variations, then:
```
pressure_gradient = (P_neighbor - P_cell) / distance
acceleration = -pressure_gradient / density + other_forces
velocity += acceleration * dt
```

Cells swap when the velocity-driven "flux" between them exceeds binding strength.

## Spring-Damper Bonds
Treat adjacent cells as connected by springs with:
- Rest length (preferred spacing)
- Spring constant (binding strength) 
- Damping coefficient (to prevent oscillations)

This naturally includes both elastic restoring forces and inertia while allowing gradual rearrangement.

What kind of planetary phenomena are you trying to simulate? That might help narrow down the best approach for your specific case.

================

Q: How would I compute binding force. Would it be a calculated force or feed into the velocity threshold

Great question! I'd recommend treating binding force as a **threshold** rather than a calculated force, for both physical intuition and computational simplicity.

## Binding Force as Threshold

Each cell type has a binding strength value:
```
binding_strength = {
  'rock': 100,
  'liquid': 10, 
  'gas': 1,
  'ice': 50
}
```

Then your swap logic becomes:
```
net_force = calculate_forces(cell)  // gravity, pressure, etc.
if |net_force| > binding_strength[cell.type]:
    attempt_swap_in_direction(force_direction)
```

## Why This Works Better

**Physical intuition**: Binding strength represents how much force is needed to "break" a material free from its current position - like yield strength or cohesion.

**Computational benefits**: 
- Clean separation between force calculation and movement decision
- Easy to tune different material behaviors
- Avoids feedback loops that can cause instability

## Enhanced Version

You could make binding strength depend on local conditions:
```python
effective_binding = base_binding * pressure_factor * temperature_factor
# Ice becomes weaker when hot, rock stronger under pressure
```

Or make it directional - easier to move vertically (gravity) than horizontally.

## Alternative: Viscosity Model
If you want more realistic fluid behavior, you could also use binding strength to determine **how much** material swaps rather than just whether it swaps:

```python
swap_fraction = min(1.0, (net_force - binding_strength) / binding_strength)
```

This gives you gradual flow rather than binary swapping, which might look more natural for liquids and gases.

What types of materials are you planning to simulate?

==================

Q: The binding force needs to be an array and not a list, correct?

You're absolutely right to think about the data structure! It depends on your simulation design:

## 2D Array Approach (Recommended)
```python
# Match your simulation grid dimensions
binding_force = np.zeros((height, width))

# Initialize based on material types
for y in range(height):
    for x in range(width):
        material = simulation_grid[y][x].type
        binding_force[y][x] = material_properties[material]['binding']
```

This is better because:
- Direct spatial correspondence with your simulation grid
- Allows position-dependent binding (weathered rock, compressed materials)
- Fast lookups during simulation: `binding_force[y][x]`

## When to Update the Array
```python
# When materials change
if cell_swapped:
    binding_force[y1][x1] = get_binding_strength(new_material_1)
    binding_force[y2][x2] = get_binding_strength(new_material_2)

# Or for dynamic effects
binding_force[y][x] *= pressure_factor * temperature_factor
```

## Alternative: Computed on the Fly
```python
def get_binding_at(x, y):
    base = material_binding[grid[y][x].type]
    return base * local_pressure_factor(x,y) * local_temp_factor(x,y)
```

**Use the array approach** if binding changes frequently or you want spatial gradients. **Use computed approach** if binding is mostly static and depends only on material type.

For planetary simulation, I'd go with the 2D array since you'll likely want effects like "rock becomes weaker near the surface" or "ice binding varies with temperature layers."



=======


Guidance: @PHYSICS.md We need to rework our plan for how to handle cell swapping. I have a new proposal here in @NEWPLAN.md . I'd you to review it and then to incorporate it into our PHYSICS.md doc.

I want the start of the PHYSICS doc to lay out the fundamental equations and assumptions / rules that we are using. Then in subsequent sections we'll go into calculation details and methods for each of the topics.

This NEWPLAN is related to how we determine when cells should swap. The basic idea is we will use their velocity difference with a threshold.

For binding strenght fluids will have zero binding force (water-water = 0, water=solid = 0).  Whereas solid-solid = X N force threshold. Then if the net force on an object doesn't exceed that threshold then we don't do the relative velocity checks.

We will want the threshold force to vary as a function of temperature and other parameters, for example to account for freeze-thaw cycles or other erosion processes.