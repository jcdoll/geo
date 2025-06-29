"""
Enhanced planet scenario with atmosphere and oceans.
"""

import numpy as np
from typing import Tuple, Optional
from ..core.particles import ParticleArrays
from ..physics.materials import MaterialType, MaterialDatabase


def create_planet_with_atmosphere(
    radius: float = 50,  # Planet radius in meters (small scale)
    particle_spacing: float = 0.5,  # Finer particle spacing
    ocean_depth: float = 3,  # Ocean depth
    atmosphere_height: float = 10,  # Atmosphere height
    center: Optional[Tuple[float, float]] = None) -> Tuple[ParticleArrays, int]:
    """
    Create a planet with atmosphere and oceans.
    
    Structure:
    - Core: uranium (radioactive heating)
    - Mantle: rock
    - Ocean: water layer on surface
    - Atmosphere: air layer above surface
    
    Args:
        radius: Planet radius (meters)
        particle_spacing: Space between particles
        ocean_depth: Depth of water layer
        atmosphere_height: Height of atmosphere
        center: Planet center position
        
    Returns:
        (particles, n_active)
    """
    if center is None:
        center = (radius * 2, radius * 2)
    
    # Layer boundaries
    core_radius = radius * 0.3
    mantle_radius = radius - ocean_depth
    ocean_radius = radius
    atmosphere_radius = radius + atmosphere_height
    
    # Generate positions for all layers
    positions = []
    materials = []
    temperatures = []
    
    # Use larger spacing for atmosphere
    atmosphere_spacing = particle_spacing * 2.5
    
    # Generate atmosphere particles using hexagonal packing for efficiency
    print(f"Generating atmosphere up to r={atmosphere_radius:.1f}m...")
    from .planet import generate_hexagonal_packing
    
    # Get all positions in atmosphere region
    atmo_positions = generate_hexagonal_packing(center, atmosphere_radius, atmosphere_spacing)
    
    # Filter to only atmosphere layer
    for x, y in atmo_positions:
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        if ocean_radius < r <= atmosphere_radius:
            positions.append([x, y])
            materials.append(MaterialType.AIR)
            # Temperature decreases with altitude
            altitude = r - ocean_radius
            temp = 288 - altitude * 6.5  # ~6.5K/km lapse rate scaled
            temperatures.append(max(220, temp))
    
    # Generate ocean and planet particles using hexagonal packing
    print(f"Generating planet and ocean up to r={ocean_radius:.1f}m...")
    planet_positions = generate_hexagonal_packing(center, ocean_radius, particle_spacing)
    
    for x, y in planet_positions:
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        if r <= ocean_radius:
            if r <= core_radius:
                # Core - hot uranium
                positions.append([x, y])
                materials.append(MaterialType.URANIUM)
                # Temperature decreases from center
                temp = 1500 - (r / core_radius) * 500
                temperatures.append(temp)
                
            elif r <= mantle_radius:
                # Mantle - rock
                positions.append([x, y])
                materials.append(MaterialType.ROCK)
                # Temperature gradient
                depth_fraction = (r - core_radius) / (mantle_radius - core_radius)
                temp = 1000 - depth_fraction * 700
                temperatures.append(temp)
                
            else:
                # Ocean layer - but only on upper hemisphere for variety
                # This creates land masses
                angle = np.arctan2(y - center[1], x - center[0])
                if np.sin(angle) > -0.3 or np.random.random() < 0.3:  # 70% ocean coverage
                    positions.append([x, y])
                    materials.append(MaterialType.WATER)
                    temperatures.append(288)  # 15°C
                else:
                    # Land (exposed rock)
                    positions.append([x, y])
                    materials.append(MaterialType.ROCK)
                    temperatures.append(288)
    
    # Convert to arrays
    n_particles = len(positions)
    print(f"Total particles: {n_particles}")
    
    # Allocate with extra space
    particles = ParticleArrays.allocate(max(n_particles * 2, n_particles + 2000))
    material_db = MaterialDatabase()
    
    # Fill particle data
    for i, ((x, y), mat, temp) in enumerate(zip(positions, materials, temperatures)):
        particles.position_x[i] = x
        particles.position_y[i] = y
        particles.velocity_x[i] = 0.0
        particles.velocity_y[i] = 0.0
        particles.material_id[i] = mat.value
        particles.temperature[i] = temp
        particles.smoothing_h[i] = particle_spacing * 1.3
        
        # Set mass based on material
        props = material_db.get_properties(mat)
        volume = np.pi * particles.smoothing_h[i]**2
        particles.mass[i] = props.density_ref * volume
        particles.density[i] = props.density_ref
    
    return particles, n_particles


def create_simple_ocean_world(
    radius: float = 30,
    particle_spacing: float = 0.8,
    ocean_fraction: float = 0.5,
    center: Optional[Tuple[float, float]] = None) -> Tuple[ParticleArrays, int]:
    """
    Create a simple ocean world - rock core with water covering.
    
    Args:
        radius: Total radius
        particle_spacing: Particle spacing  
        ocean_fraction: Fraction of radius that is ocean
        center: Center position
        
    Returns:
        (particles, n_active)
    """
    if center is None:
        center = (radius * 2, radius * 2)
        
    rock_radius = radius * (1 - ocean_fraction)
    
    positions = []
    materials = []
    
    # Generate particles
    for y in np.arange(center[1] - radius, center[1] + radius + particle_spacing, particle_spacing):
        for x in np.arange(center[0] - radius, center[0] + radius + particle_spacing, particle_spacing):
            r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            
            if r <= radius:
                positions.append([x, y])
                if r <= rock_radius:
                    materials.append(MaterialType.ROCK)
                else:
                    materials.append(MaterialType.WATER)
    
    n_particles = len(positions)
    particles = ParticleArrays.allocate(max(n_particles * 2, n_particles + 1000))
    material_db = MaterialDatabase()
    
    # Fill data
    for i, ((x, y), mat) in enumerate(zip(positions, materials)):
        particles.position_x[i] = x
        particles.position_y[i] = y
        particles.velocity_x[i] = 0.0
        particles.velocity_y[i] = 0.0
        particles.material_id[i] = mat.value
        particles.temperature[i] = 288
        particles.smoothing_h[i] = particle_spacing * 1.3
        
        props = material_db.get_properties(mat)
        volume = np.pi * particles.smoothing_h[i]**2
        particles.mass[i] = props.density_ref * volume
        particles.density[i] = props.density_ref
    
    return particles, n_particles