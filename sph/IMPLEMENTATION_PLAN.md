# SPH Implementation Plan

## Overview

This document outlines the step-by-step implementation plan for the SPH-based geological simulation, designed to address the issues encountered with grid-based approaches while enabling new capabilities like pseudo-rigid body physics.

## Phase 1: Foundation (Week 1)

### 1.1 Particle Data Structure
```python
# sph/core/particle.py
@dataclass
class Particle:
    # Primary state
    position: np.ndarray      # [x, y]
    velocity: np.ndarray      # [vx, vy]
    mass: float
    
    # Computed quantities
    density: float
    pressure: float
    
    # Material properties
    material_id: int
    temperature: float
    
    # SPH specific
    smoothing_length: float
    neighbors: List[int]      # Indices of neighboring particles
```

### 1.2 Kernel Functions
```python
# sph/core/kernel.py
class CubicSplineKernel:
    """Standard cubic spline kernel for SPH"""
    def W(self, r: float, h: float) -> float
    def gradW(self, r_vec: np.ndarray, h: float) -> np.ndarray
```

### 1.3 Spatial Hashing
```python
# sph/core/spatial.py
class SpatialHash:
    """O(N) neighbor finding using spatial hashing"""
    def insert(self, particle_id: int, position: np.ndarray)
    def find_neighbors(self, position: np.ndarray, radius: float) -> List[int]
    def update(self, particles: List[Particle])
```

### 1.4 Basic Integration Test
- Create 100 particles in a box
- Verify neighbor finding works correctly
- Test kernel function properties (normalization, symmetry)

## Phase 2: Basic Fluid Dynamics (Week 2)

### 2.1 Density Calculation
```python
# sph/physics/density.py
def compute_density_summation(particles: List[Particle], kernel: Kernel):
    """Direct density summation: ρᵢ = Σⱼ mⱼ W(rᵢ - rⱼ, h)"""
    
def compute_density_continuity(particles: List[Particle], kernel: Kernel, dt: float):
    """Continuity equation: dρ/dt = -ρ∇·v"""
```

### 2.2 Pressure Force
```python
# sph/physics/forces.py
def compute_pressure_forces(particles: List[Particle], kernel: Kernel):
    """SPH pressure gradient: -∇P/ρ"""
    
def tait_equation_of_state(density: float, material: Material) -> float:
    """P = B[(ρ/ρ₀)^γ - 1]"""
```

### 2.3 Artificial Viscosity
```python
def compute_artificial_viscosity(particles: List[Particle], kernel: Kernel):
    """Monaghan viscosity for shock handling"""
```

### 2.4 Dam Break Test
- Classic SPH validation scenario
- 2D column of water particles
- Verify correct flow behavior
- Check conservation of mass/momentum

## Phase 3: Solid Mechanics (Week 3)

### 3.1 Stress Tensor
```python
# sph/physics/solid.py
@dataclass
class SolidParticle(Particle):
    stress: np.ndarray        # 2x2 stress tensor
    strain_rate: np.ndarray   # 2x2 strain rate
    rotation_rate: np.ndarray # Anti-symmetric part
```

### 3.2 Elastic Forces
```python
def compute_elastic_forces(particles: List[SolidParticle], kernel: Kernel):
    """Hooke's law stress-strain relationship"""
    
def update_stress_tensor(particle: SolidParticle, dt: float):
    """Jaumann stress rate for rotation handling"""
```

### 3.3 Cohesive Forces
```python
# sph/physics/cohesion.py
class CohesiveBond:
    particle_i: int
    particle_j: int
    rest_length: float
    strength: float
    
def compute_cohesive_forces(particles: List[Particle], bonds: List[CohesiveBond]):
    """Spring-like cohesive forces between bonded particles"""
```

### 3.4 Fracture
```python
def check_fracture(bonds: List[CohesiveBond], particles: List[Particle]):
    """Remove bonds when stress exceeds yield strength"""
```

### 3.5 Elastic Collision Test
- Drop elastic ball on rigid surface
- Verify correct rebound behavior
- Test stress wave propagation

## Phase 4: Thermal Physics (Week 4)

### 4.1 Heat Conduction
```python
# sph/physics/thermal.py
def compute_heat_conduction(particles: List[Particle], kernel: Kernel):
    """SPH formulation of Fourier's law"""
```

### 4.2 Phase Transitions
```python
def check_phase_transitions(particles: List[Particle]):
    """Handle melting, freezing, evaporation, condensation"""
    
def apply_latent_heat(particle: Particle, transition: PhaseTransition):
    """Adjust temperature for latent heat"""
```

### 4.3 Thermal Expansion
```python
def thermal_expansion_pressure(particle: Particle) -> float:
    """Additional pressure from thermal expansion"""
```

### 4.4 Heat Diffusion Test
- Hot particles in center, cold outside
- Verify correct diffusion rate
- Test with different materials

## Phase 5: Self-Gravity (Week 5)

### 5.1 Direct N-Body
```python
# sph/physics/gravity.py
def compute_gravity_direct(particles: List[Particle]):
    """O(N²) direct gravity calculation for small N"""
```

### 5.2 Barnes-Hut Tree
```python
class QuadTree:
    """Spatial tree for O(N log N) gravity"""
    
def compute_gravity_tree(particles: List[Particle], theta: float = 0.5):
    """Tree code gravity with opening angle theta"""
```

### 5.3 Planet Formation Test
- Cloud of particles with self-gravity
- Verify collapse and heating
- Test conservation of angular momentum

## Phase 6: Integration & Optimization (Week 6)

### 6.1 Symplectic Integrator
```python
# sph/core/integrator.py
def leapfrog_integrate(particles: List[Particle], forces: Forces, dt: float):
    """Energy-conserving time integration"""
```

### 6.2 Adaptive Timestepping
```python
def compute_timestep(particles: List[Particle]) -> float:
    """CFL and viscous stability constraints"""
```

### 6.3 Performance Optimization
- Particle sorting for cache efficiency
- Vectorized kernel evaluations
- OpenMP parallelization
- Consider GPU implementation

### 6.4 Comprehensive Tests
- Multi-material simulation
- Complex geometry (e.g., asteroid impact)
- Long-term stability tests
- Conservation verification

## Phase 7: Visualization & UI (Week 7)

### 7.1 Particle Renderer
```python
# sph/visualization.py
class SPHRenderer:
    def render_particles(self, particles: List[Particle], screen: Surface)
    def render_bonds(self, bonds: List[CohesiveBond], particles: List[Particle])
    def render_field(self, field: str)  # temperature, pressure, etc.
```

### 7.2 Interactive Tools
- Add/remove particles
- Apply forces
- Heat sources/sinks
- Material painting

### 7.3 Analysis Tools
- Graphs (energy, momentum, mass)
- Material composition
- Temperature distribution
- Stress analysis

## Success Metrics

1. **Performance**: 30+ FPS with 10,000 particles
2. **Stability**: Handle 10^6:1 density ratios
3. **Conservation**: Mass, momentum, energy < 0.1% drift
4. **Physics**: Correct behavior in standard tests
5. **Usability**: Interactive and intuitive

## Risk Mitigation

1. **Performance Issues**
   - Start with 2D (easier than 3D)
   - Implement spatial hashing early
   - Profile and optimize incrementally

2. **Numerical Instability**
   - Use conservative timesteps initially
   - Implement artificial viscosity
   - Add damping if needed

3. **Complex Physics**
   - Start with fluids only
   - Add solids incrementally
   - Validate each feature independently

## Deliverables

Each phase produces:
1. Working code with tests
2. Validation scenarios
3. Performance benchmarks
4. Documentation updates

## Timeline

- Week 1: Foundation
- Week 2: Fluid dynamics
- Week 3: Solid mechanics
- Week 4: Thermal physics
- Week 5: Self-gravity
- Week 6: Integration & optimization
- Week 7: Visualization & polish

Total: 7 weeks to working SPH geological simulator