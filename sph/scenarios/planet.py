"""
Planet initialization scenarios for SPH simulation.

Creates realistic planetary structures with:
- Layered composition (core, mantle, crust)
- Proper density and temperature profiles
- Material-based properties
"""

import numpy as np
from typing import Tuple, Optional
from ..core.particles import ParticleArrays
from ..physics.materials import MaterialType, MaterialDatabase


def generate_hexagonal_packing(center: Tuple[float, float], radius: float, 
                              spacing: float) -> np.ndarray:
    """Generate hexagonal close-packed positions within a circle.
    
    Args:
        center: (x, y) center of circle
        radius: Circle radius
        spacing: Particle spacing
        
    Returns:
        Array of (x, y) positions
    """
    positions = []
    
    # Hexagonal lattice parameters
    dy = spacing * np.sqrt(3) / 2  # Vertical spacing
    
    # Generate rows
    n_rows = int(2 * radius / dy) + 1
    
    for row in range(-n_rows, n_rows + 1):
        y = center[1] + row * dy
        
        # Row offset for hexagonal packing
        if row % 2 == 0:
            x_offset = 0
        else:
            x_offset = spacing / 2
        
        # Calculate row width at this y
        y_dist = abs(y - center[1])
        if y_dist > radius:
            continue
            
        row_width = 2 * np.sqrt(max(0, radius**2 - y_dist**2))
        n_in_row = int(row_width / spacing) + 1
        
        for col in range(n_in_row):
            x = center[0] - row_width/2 + col * spacing + x_offset
            
            # Check if inside circle
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist <= radius:
                positions.append([x, y])
    
    return np.array(positions)


def create_planet_earth_like(radius_km: float = 6371, 
                           particle_spacing_km: float = 50,
                           center: Optional[Tuple[float, float]] = None) -> Tuple[ParticleArrays, int]:
    """Create an Earth-like planet with realistic structure.
    
    Structure based on Earth:
    - Inner core (solid iron): 0-1220 km radius, 5700 K
    - Outer core (liquid iron): 1220-3480 km radius, 4500 K
    - Lower mantle: 3480-5701 km radius, 3000 K
    - Upper mantle: 5701-6336 km radius, 1500 K
    - Crust: 6336-6371 km radius, 300 K
    
    Args:
        radius_km: Planet radius in kilometers
        particle_spacing_km: Spacing between particles in km
        center: (x, y) center position, defaults to domain center
        
    Returns:
        (particles, n_active)
    """
    # Convert to meters
    radius = radius_km * 1000
    spacing = particle_spacing_km * 1000
    
    if center is None:
        # Default to center of a 20,000 km domain
        center = (10000 * 1000, 10000 * 1000)
    else:
        center = (center[0] * 1000, center[1] * 1000)
    
    # Scale layer boundaries to planet size
    scale = radius_km / 6371
    inner_core_r = 1220 * scale * 1000
    outer_core_r = 3480 * scale * 1000
    lower_mantle_r = 5701 * scale * 1000
    upper_mantle_r = 6336 * scale * 1000
    
    # Generate positions
    positions = generate_hexagonal_packing(center, radius, spacing)
    n_particles = len(positions)
    
    # Allocate particle arrays
    particles = ParticleArrays.allocate(max(n_particles * 2, n_particles + 1000), include_solids=True)
    material_db = MaterialDatabase()
    
    # Fill particle data
    for i, (x, y) in enumerate(positions):
        particles.position_x[i] = x
        particles.position_y[i] = y
        particles.velocity_x[i] = 0.0
        particles.velocity_y[i] = 0.0
        particles.smoothing_h[i] = 1.3 * spacing
        
        # Distance from center
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        depth = radius - r
        
        # Determine layer and properties
        if r <= inner_core_r:
            # Inner core - solid iron
            material = MaterialType.URANIUM  # Using as proxy for solid iron
            temperature = 5700 - (r / inner_core_r) * 1000  # 5700-4700 K
            density_factor = 1.2  # Compressed
            
        elif r <= outer_core_r:
            # Outer core - liquid iron (using magma as proxy)
            material = MaterialType.MAGMA
            temperature = 4500 - (r - inner_core_r) / (outer_core_r - inner_core_r) * 500
            density_factor = 1.1
            
        elif r <= lower_mantle_r:
            # Lower mantle - hot rock
            material = MaterialType.ROCK
            temperature = 3000 - (r - outer_core_r) / (lower_mantle_r - outer_core_r) * 1000
            density_factor = 1.05
            
        elif r <= upper_mantle_r:
            # Upper mantle - cooler rock
            material = MaterialType.ROCK
            temperature = 1500 - (r - lower_mantle_r) / (upper_mantle_r - lower_mantle_r) * 1000
            density_factor = 1.0
            
        else:
            # Crust - surface rock
            material = MaterialType.ROCK
            temperature = 300 + depth / 1000 * 30  # Geothermal gradient
            density_factor = 0.95
        
        # Set material properties
        particles.material_id[i] = material
        particles.temperature[i] = temperature
        
        # Get material reference density and adjust
        mat_props = material_db.get_properties(material)
        base_density = mat_props.density_ref * density_factor
        
        # Mass based on volume element
        particle_volume = spacing**2  # 2D approximation
        particles.mass[i] = base_density * particle_volume
        particles.density[i] = base_density
    
    return particles, n_particles


def create_planet_simple(radius: float = 5000, 
                        particle_spacing: float = 100,
                        center: Optional[Tuple[float, float]] = None,
                        uniform_material: Optional[MaterialType] = None) -> Tuple[ParticleArrays, int]:
    """Create a simple planet with uranium core, rock mantle, water ocean, and atmosphere.
    
    Args:
        radius: Planet radius in meters
        particle_spacing: Spacing between particles in meters
        center: (x, y) center position
        uniform_material: If set, use single material. Otherwise use layered structure
        
    Returns:
        (particles, n_active)
    """
    if center is None:
        center = (10000, 10000)
    
    # Generate positions
    positions = generate_hexagonal_packing(center, radius, particle_spacing)
    n_particles = len(positions)
    
    # Allocate arrays with extra space for adding particles
    particles = ParticleArrays.allocate(max(n_particles * 2, n_particles + 1000))
    material_db = MaterialDatabase()
    
    # Fill data
    for i, (x, y) in enumerate(positions):
        particles.position_x[i] = x
        particles.position_y[i] = y
        particles.velocity_x[i] = 0.0
        particles.velocity_y[i] = 0.0
        particles.smoothing_h[i] = 1.3 * particle_spacing
        
        # Distance from center
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        if uniform_material is not None:
            # Single material
            material = uniform_material
            temperature = 300.0
        else:
            # Layered structure: uranium core, rock mantle, water ocean, air atmosphere
            if r < 0.2 * radius:
                # Small uranium core (20% of radius)
                material = MaterialType.URANIUM
                temperature = 2000.0
            elif r < 0.7 * radius:
                # Rock mantle (20-70% of radius)
                material = MaterialType.ROCK
                temperature = 1000.0 - (r - 0.2 * radius) / (0.5 * radius) * 700
            elif r < 0.85 * radius:
                # Water ocean (70-85% of radius)
                material = MaterialType.WATER
                temperature = 280.0
            elif r < 0.95 * radius:
                # Dense atmosphere (85-95% of radius)
                material = MaterialType.AIR
                temperature = 250.0
            else:
                # Thin outer atmosphere (95-100% of radius)
                material = MaterialType.AIR
                temperature = 200.0
        
        particles.material_id[i] = material
        particles.temperature[i] = temperature
        
        # Set mass
        mat_props = material_db.get_properties(material)
        particle_volume = particle_spacing**2
        particles.mass[i] = mat_props.density_ref * particle_volume
        particles.density[i] = mat_props.density_ref
    
    return particles, n_particles


def create_asteroid_impact_scenario(planet_radius: float = 5000,
                                   asteroid_radius: float = 500,
                                   impact_velocity: float = 20000,
                                   spacing: float = 100) -> Tuple[ParticleArrays, int]:
    """Create a planet with incoming asteroid scenario.
    
    Args:
        planet_radius: Planet radius (m)
        asteroid_radius: Asteroid radius (m)
        impact_velocity: Asteroid velocity (m/s)
        spacing: Particle spacing (m)
        
    Returns:
        (particles, n_active)
    """
    # Create planet
    planet_center = (10000, 10000)
    particles_planet, n_planet = create_planet_simple(
        planet_radius, spacing, planet_center, MaterialType.ROCK
    )
    
    # Create asteroid
    asteroid_center = (planet_center[0] + planet_radius + asteroid_radius + 1000,
                      planet_center[1] + planet_radius * 0.5)
    
    positions_asteroid = generate_hexagonal_packing(
        asteroid_center, asteroid_radius, spacing
    )
    n_asteroid = len(positions_asteroid)
    
    # Total particles
    n_total = n_planet + n_asteroid
    
    # Reallocate if needed
    if n_total > particles_planet.position_x.shape[0]:
        particles = ParticleArrays.allocate(n_total * 2)
        # Copy planet data
        for attr in ['position_x', 'position_y', 'velocity_x', 'velocity_y',
                    'mass', 'density', 'material_id', 'temperature', 'smoothing_h']:
            getattr(particles, attr)[:n_planet] = getattr(particles_planet, attr)[:n_planet]
    else:
        particles = particles_planet
    
    # Add asteroid particles
    material_db = MaterialDatabase()
    asteroid_props = material_db.get_properties(MaterialType.ROCK)
    
    for i, (x, y) in enumerate(positions_asteroid):
        idx = n_planet + i
        particles.position_x[idx] = x
        particles.position_y[idx] = y
        
        # Velocity toward planet center
        dx = planet_center[0] - x
        dy = planet_center[1] - y
        dist = np.sqrt(dx*dx + dy*dy)
        
        particles.velocity_x[idx] = impact_velocity * dx / dist
        particles.velocity_y[idx] = impact_velocity * dy / dist
        
        particles.material_id[idx] = MaterialType.ROCK
        particles.temperature[idx] = 300.0
        particles.mass[idx] = asteroid_props.density_ref * spacing**2
        particles.density[idx] = asteroid_props.density_ref
        particles.smoothing_h[idx] = 1.3 * spacing
    
    return particles, n_total