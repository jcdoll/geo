# SPH Implementation Plan - Phase 2: Physics & Materials

## Overview

With the vectorized foundation complete, we now need to add the physics required for geological simulation. This includes materials, thermal physics, self-gravity, and phase transitions.

## Phase 2.1: Material System (Priority: High)

### Goals
- Port material properties from flux implementation
- Support 9 material types with distinct properties
- Enable material-dependent physics

### Implementation

#### 2.1.1 Material Properties Database
```python
# sph/physics/materials.py
from dataclasses import dataclass
import numpy as np

@dataclass
class MaterialProperties:
    """Properties for a single material type."""
    name: str
    density_ref: float      # Reference density (kg/m³)
    bulk_modulus: float     # For Tait EOS (Pa)
    viscosity: float        # Dynamic viscosity (Pa·s)
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float    # J/(kg·K)
    melting_point: float    # K
    boiling_point: float    # K
    latent_heat_fusion: float    # J/kg
    latent_heat_vaporization: float  # J/kg
    cohesion_strength: float  # Pa (for solids)
    
class MaterialDatabase:
    """Database of material properties matching flux implementation."""
    
    materials = {
        0: MaterialProperties(
            name="water",
            density_ref=1000.0,
            bulk_modulus=2.2e9,
            viscosity=0.001,
            thermal_conductivity=0.6,
            specific_heat=4186,
            melting_point=273.15,
            boiling_point=373.15,
            latent_heat_fusion=334000,
            latent_heat_vaporization=2260000,
            cohesion_strength=0.0  # Liquid
        ),
        1: MaterialProperties(
            name="granite",
            density_ref=2700.0,
            bulk_modulus=50e9,
            viscosity=1e20,  # Effectively solid
            thermal_conductivity=2.5,
            specific_heat=790,
            melting_point=1500,
            boiling_point=3000,
            latent_heat_fusion=400000,
            latent_heat_vaporization=0,
            cohesion_strength=20e6  # 20 MPa
        ),
        # ... other materials from flux
    }
```

#### 2.1.2 Material-Aware EOS
```python
def compute_pressure_material_aware(particles, material_db, n_active):
    """Compute pressure using material-specific properties."""
    for mat_id in np.unique(particles.material_id[:n_active]):
        mask = particles.material_id[:n_active] == mat_id
        mat = material_db.materials[mat_id]
        
        # Tait EOS with material properties
        particles.pressure[mask] = tait_equation_of_state(
            particles.density[mask],
            mat.density_ref,
            mat.bulk_modulus
        )
```

## Phase 2.2: Thermal Physics (Priority: High)

### Goals
- Heat conduction between particles
- Temperature-dependent properties
- Phase transitions with latent heat

### Implementation

#### 2.2.1 SPH Heat Conduction
```python
# sph/physics/thermal_vectorized.py
def compute_heat_transfer_sph(particles, kernel, material_db, n_active, dt):
    """SPH formulation of heat conduction with variable conductivity."""
    
    dT_dt = np.zeros(n_active, dtype=np.float32)
    
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
            
        # Material properties
        mat_i = material_db.materials[particles.material_id[i]]
        k_i = mat_i.thermal_conductivity
        cp_i = mat_i.specific_heat
        
        for j_idx in range(n_neighbors):
            j = particles.neighbor_ids[i, j_idx]
            
            # Neighbor properties
            mat_j = material_db.materials[particles.material_id[j]]
            k_j = mat_j.thermal_conductivity
            cp_j = mat_j.specific_heat
            
            # Temperature difference
            dT = particles.temperature[i] - particles.temperature[j]
            
            # SPH heat flux (Cleary & Monaghan 1999)
            r_ij = particles.neighbor_distances[i, j_idx]
            
            # Harmonic mean conductivity
            k_ij = 2 * k_i * k_j / (k_i + k_j)
            
            # Heat transfer rate
            heat_flux = (4 * particles.mass[j] * k_ij * dT) / (
                particles.density[j] * (cp_i + cp_j) * r_ij
            )
            
            # Kernel gradient dot position
            dx = particles.position_x[i] - particles.position_x[j]
            dy = particles.position_y[i] - particles.position_y[j]
            grad_W = kernel.gradW(np.array([dx, dy]), particles.smoothing_h[i])
            r_dot_grad = (dx * grad_W[0] + dy * grad_W[1]) / r_ij
            
            dT_dt[i] -= heat_flux * r_dot_grad / cp_i
    
    # Update temperature
    particles.temperature[:n_active] += dT_dt * dt
```

#### 2.2.2 Phase Transitions
```python
def handle_phase_transitions(particles, material_db, n_active):
    """Handle melting, freezing, boiling with latent heat."""
    
    for i in range(n_active):
        mat = material_db.materials[particles.material_id[i]]
        T = particles.temperature[i]
        
        # Check for transitions
        if mat.name == "water" and T > mat.boiling_point:
            # Water -> Steam
            particles.material_id[i] = STEAM_ID
            # Remove latent heat
            particles.temperature[i] -= mat.latent_heat_vaporization / mat.specific_heat
            
        elif mat.name == "ice" and T > mat.melting_point:
            # Ice -> Water
            particles.material_id[i] = WATER_ID
            particles.temperature[i] -= mat.latent_heat_fusion / mat.specific_heat
            
        # ... other transitions
```

## Phase 2.3: Self-Gravity (Priority: High)

### Goals
- N-body gravity for planet formation
- Efficient algorithms for large N
- Realistic planetary structure

### Implementation

#### 2.3.1 Direct N-Body (for small systems)
```python
# sph/physics/gravity_vectorized.py
def compute_gravity_direct(particles, n_active, G=6.67430e-11):
    """Direct O(N²) gravity calculation."""
    
    # Reset gravity forces
    gravity_x = np.zeros(n_active, dtype=np.float32)
    gravity_y = np.zeros(n_active, dtype=np.float32)
    
    # Vectorized pairwise forces
    for i in range(n_active):
        # Position differences to all other particles
        dx = particles.position_x[i] - particles.position_x[:n_active]
        dy = particles.position_y[i] - particles.position_y[:n_active]
        
        # Distances (with softening to avoid singularities)
        r2 = dx*dx + dy*dy + 1e-6
        r = np.sqrt(r2)
        
        # Gravitational acceleration: a = -G * m / r²
        accel = -G * particles.mass[:n_active] / r2
        
        # Components (exclude self)
        mask = np.arange(n_active) != i
        gravity_x[i] = np.sum(accel[mask] * dx[mask] / r[mask])
        gravity_y[i] = np.sum(accel[mask] * dy[mask] / r[mask])
    
    # Add to forces
    particles.force_x[:n_active] += particles.mass[:n_active] * gravity_x
    particles.force_y[:n_active] += particles.mass[:n_active] * gravity_y
```

#### 2.3.2 Barnes-Hut Tree (for large systems)
```python
class QuadTree:
    """Quadtree for O(N log N) gravity."""
    
    def __init__(self, bounds, theta=0.5):
        self.bounds = bounds
        self.theta = theta  # Opening angle
        self.total_mass = 0.0
        self.center_of_mass = np.zeros(2)
        self.children = None
        self.particle_idx = -1
        
    def insert(self, particles, idx):
        """Insert particle into tree."""
        # ... implementation
        
    def compute_force(self, particles, i):
        """Compute gravitational force on particle i."""
        # ... Barnes-Hut algorithm
```

## Phase 2.4: Planet Initialization (Priority: High)

### Goals
- Create realistic planetary structure
- Layered composition (core, mantle, crust)
- Proper density/temperature profiles

### Implementation

```python
# sph/scenarios/planet.py
def create_planet(radius_km=6371, n_particles=100000):
    """Create Earth-like planet with layers."""
    
    particles = ParticleArrays.allocate(n_particles)
    
    # Hexagonal close packing
    spacing = radius_km * 1000 / np.sqrt(n_particles / np.pi)
    
    idx = 0
    for layer in generate_hexagonal_layers(radius_km * 1000, spacing):
        for (x, y) in layer:
            r = np.sqrt(x*x + y*y)
            depth = radius_km * 1000 - r
            
            # Determine material based on depth
            if depth > 0.8 * radius_km * 1000:  # Inner core
                material = IRON_ID
                temperature = 5700  # K
                density = 13000  # kg/m³
            elif depth > 0.55 * radius_km * 1000:  # Outer core
                material = IRON_LIQUID_ID
                temperature = 4500
                density = 11000
            elif depth > 0.05 * radius_km * 1000:  # Mantle
                material = PERIDOTITE_ID
                temperature = 1500 + depth/1000 * 2
                density = 4500
            else:  # Crust
                material = GRANITE_ID
                temperature = 300
                density = 2700
            
            particles.position_x[idx] = x
            particles.position_y[idx] = y
            particles.material_id[idx] = material
            particles.temperature[idx] = temperature
            particles.mass[idx] = density * spacing**2
            
            idx += 1
    
    return particles, idx
```

## Phase 2.5: Additional Physics

### 2.5.1 Cohesive Forces (for solid behavior)
```python
def compute_cohesive_forces(particles, bonds, material_db, n_active):
    """Cohesive forces between bonded particles."""
    
    for bond in bonds:
        i, j = bond.particle_i, bond.particle_j
        
        # Check if bond should break
        dx = particles.position_x[i] - particles.position_x[j]
        dy = particles.position_y[i] - particles.position_y[j]
        distance = np.sqrt(dx*dx + dy*dy)
        strain = (distance - bond.rest_length) / bond.rest_length
        
        mat = material_db.materials[particles.material_id[i]]
        if strain > mat.yield_strain:
            bond.active = False
            continue
        
        # Spring force
        force = bond.stiffness * strain
        fx = force * dx / distance
        fy = force * dy / distance
        
        particles.force_x[i] -= fx
        particles.force_x[j] += fx
        particles.force_y[i] -= fy
        particles.force_y[j] += fy
```

### 2.5.2 Stress Tensor (for rock mechanics)
```python
def compute_stress_tensor(particles, kernel, material_db, n_active):
    """Compute deviatoric stress for solid materials."""
    
    # Velocity gradient tensor
    for i in range(n_active):
        mat = material_db.materials[particles.material_id[i]]
        
        if mat.cohesion_strength > 0:  # Solid material
            # Compute velocity gradient ∇v
            grad_vx_x, grad_vx_y = 0.0, 0.0
            grad_vy_x, grad_vy_y = 0.0, 0.0
            
            # SPH approximation of gradient
            # ... implementation
            
            # Update stress using Jaumann rate
            # ... implementation
```

## Implementation Timeline

### Week 1: Material System
- [x] Basic material properties
- [ ] Material-aware pressure
- [ ] Material-aware viscosity
- [ ] Test multi-material scenarios

### Week 2: Thermal Physics  
- [ ] SPH heat conduction
- [ ] Phase transitions
- [ ] Latent heat handling
- [ ] Test thermal scenarios

### Week 3: Self-Gravity
- [ ] Direct N-body
- [ ] Barnes-Hut tree (optional)
- [ ] Gravitational stability tests
- [ ] Planet formation test

### Week 4: Planet Setup & Integration
- [ ] Planet initialization
- [ ] Layered structure
- [ ] Cohesive forces
- [ ] Comprehensive tests

## Test Scenarios

1. **Multi-Material Flow**: Water over rock bed
2. **Heat Transfer**: Hot magma cooling in water
3. **Phase Transition**: Ice melting, water boiling
4. **Planet Formation**: Dust cloud collapse
5. **Geological**: Magma chamber with convection
6. **Impact**: Asteroid hitting planet surface

## Performance Targets

- 10,000 particles: 30+ FPS with all physics
- 50,000 particles: 10+ FPS
- 100,000 particles: 5+ FPS

## Success Criteria

1. **Materials**: All 9 material types working correctly
2. **Thermal**: Realistic heat transfer and phase transitions
3. **Gravity**: Stable planet formation
4. **Performance**: Meet FPS targets
5. **Accuracy**: Conservation of mass, momentum, energy