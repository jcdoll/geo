"""
Scenario definitions for flux-based geological simulation.

This module contains all predefined scenarios that can be loaded into
the simulation. Each scenario is a function that configures the initial
state of the simulation.
"""

import numpy as np
from typing import TYPE_CHECKING

from materials import MaterialType, MaterialDatabase

if TYPE_CHECKING:
    from state import FluxState


def setup_empty_world(state: 'FluxState', material_db: MaterialDatabase):
    """Create an empty world with just space."""
    state.vol_frac.fill(0.0)
    state.vol_frac[MaterialType.SPACE.value] = 1.0
    state.temperature.fill(273.0)  # 0°C
    state.normalize_volume_fractions()
    state.update_mixture_properties(material_db)


def setup_planet(state: 'FluxState', material_db: MaterialDatabase):
    """Create a simple circular planet with atmosphere."""
    nx, ny = state.nx, state.ny
    
    # Initialize everything as space first
    state.vol_frac.fill(0.0)
    state.vol_frac[MaterialType.SPACE.value] = 1.0
    
    # Create circular planet
    cx, cy = nx // 2, ny // 2
    radius = min(nx, ny) // 3
    
    y_grid, x_grid = np.ogrid[:ny, :nx]
    dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    
    # Rock core
    rock_mask = dist_from_center < radius
    state.vol_frac[MaterialType.SPACE.value][rock_mask] = 0.0
    state.vol_frac[MaterialType.ROCK.value][rock_mask] = 1.0
    
    # Thin atmosphere  
    atmos_mask = (dist_from_center >= radius) & (dist_from_center < radius + 5)
    state.vol_frac[MaterialType.SPACE.value][atmos_mask] = 0.0
    state.vol_frac[MaterialType.AIR.value][atmos_mask] = 1.0
    
    # Add some water on surface (pools)
    water_mask = (dist_from_center >= radius - 8) & (dist_from_center < radius - 2) & (y_grid > cy)
    state.vol_frac[MaterialType.ROCK.value][water_mask] = 0.0
    state.vol_frac[MaterialType.WATER.value][water_mask] = 1.0
    
    # Set initial temperature
    state.temperature.fill(288.0)  # 15°C
    
    # Temperature gradient - hotter at depth
    for j in range(ny):
        for i in range(nx):
            if rock_mask[j, i]:
                depth = dist_from_center[j, i] / radius
                state.temperature[j, i] = 288 + (1 - depth) * 500  # Hotter toward center
    
    # Normalize and update properties
    state.normalize_volume_fractions()
    state.update_mixture_properties(material_db)
    
    # Initialize pressure to zero (projection method will establish equilibrium)
    state.pressure.fill(0.0)


def setup_layered_planet(state: 'FluxState', material_db: MaterialDatabase):
    """Create a planet with flat atmosphere, ocean, and crust layers."""
    nx, ny = state.nx, state.ny
    
    # Clear everything
    state.vol_frac.fill(0.0)
    
    # Define layers from bottom up
    crust_height = int(ny * 0.4)
    ocean_height = int(ny * 0.2)
    atmos_height = ny - crust_height - ocean_height
    
    # Rock crust with some uranium deposits
    state.vol_frac[MaterialType.ROCK.value, -crust_height:, :] = 1.0
    
    # Add uranium veins
    for i in range(3):
        x = int((i + 1) * nx / 4)
        y = ny - int(crust_height * 0.5)
        size = 3
        state.vol_frac[MaterialType.ROCK.value, y-size:y+size, x-size:x+size] = 0.0
        state.vol_frac[MaterialType.URANIUM.value, y-size:y+size, x-size:x+size] = 1.0
    
    # Ocean layer
    ocean_top = ny - crust_height
    ocean_bottom = ocean_top - ocean_height
    state.vol_frac[MaterialType.WATER.value, ocean_bottom:ocean_top, :] = 1.0
    
    # Atmosphere
    state.vol_frac[MaterialType.AIR.value, :ocean_bottom, :] = 1.0
    
    # Temperature gradient - hot core, cool surface
    for j in range(ny):
        depth_fraction = j / ny
        # Temperature from 250K at top to 350K at bottom
        state.temperature[j, :] = 250 + 100 * depth_fraction
        
    # Update properties
    state.normalize_volume_fractions()
    state.update_mixture_properties(material_db)


def setup_volcanic_island(state: 'FluxState', material_db: MaterialDatabase):
    """Create a volcanic island scenario."""
    nx, ny = state.nx, state.ny
    
    # Clear and fill with air
    state.vol_frac.fill(0.0)
    state.vol_frac[MaterialType.AIR.value] = 1.0
    
    # Ocean
    ocean_level = int(ny * 0.6)
    state.vol_frac[MaterialType.AIR.value, ocean_level:, :] = 0.0
    state.vol_frac[MaterialType.WATER.value, ocean_level:, :] = 1.0
    
    # Island - triangular shape
    island_center = nx // 2
    island_base_width = nx // 3
    island_height = int(ny * 0.5)
    
    for j in range(ny):
        if j > ny - island_height:
            # Calculate island width at this height
            height_from_base = ny - j
            width_fraction = height_from_base / island_height
            width = int(island_base_width * width_fraction)
            
            if width > 0:
                x_start = max(0, island_center - width)
                x_end = min(nx, island_center + width)
                
                # Clear water and add rock
                state.vol_frac[MaterialType.WATER.value, j, x_start:x_end] = 0.0
                state.vol_frac[MaterialType.ROCK.value, j, x_start:x_end] = 1.0
    
    # Magma chamber
    chamber_y = ny - island_height // 3
    chamber_size = 5
    state.vol_frac[MaterialType.ROCK.value, 
                   chamber_y-chamber_size:chamber_y+chamber_size,
                   island_center-chamber_size:island_center+chamber_size] = 0.0
    state.vol_frac[MaterialType.MAGMA.value, 
                   chamber_y-chamber_size:chamber_y+chamber_size,
                   island_center-chamber_size:island_center+chamber_size] = 1.0
    
    # Temperature - hot magma, cool elsewhere
    state.temperature.fill(290.0)  # 17°C
    magma_mask = state.vol_frac[MaterialType.MAGMA.value] > 0.5
    state.temperature[magma_mask] = 1500.0  # Hot magma
    
    # Update properties
    state.normalize_volume_fractions()
    state.update_mixture_properties(material_db)


def setup_ice_world(state: 'FluxState', material_db: MaterialDatabase):
    """Create an ice world with subsurface ocean."""
    nx, ny = state.nx, state.ny
    
    # Clear
    state.vol_frac.fill(0.0)
    
    # Rock core
    core_height = int(ny * 0.3)
    state.vol_frac[MaterialType.ROCK.value, -core_height:, :] = 1.0
    
    # Subsurface ocean
    ocean_height = int(ny * 0.3)
    ocean_bottom = ny - core_height
    ocean_top = ocean_bottom - ocean_height
    state.vol_frac[MaterialType.WATER.value, ocean_top:ocean_bottom, :] = 1.0
    
    # Ice shell
    state.vol_frac[MaterialType.ICE.value, :ocean_top, :] = 1.0
    
    # Cold surface, warm interior
    for j in range(ny):
        if j < ocean_top:
            # Ice layer - very cold at surface
            state.temperature[j, :] = 100 + 150 * (j / ocean_top)  # 100K to 250K
        elif j < ocean_bottom:
            # Ocean layer - just above freezing
            state.temperature[j, :] = 275.0  # 2°C
        else:
            # Rock core - warmer
            state.temperature[j, :] = 280 + 20 * ((j - ocean_bottom) / core_height)
            
    # Update properties
    state.normalize_volume_fractions()
    state.update_mixture_properties(material_db)


# Scenario registry
SCENARIOS = {
    'empty': setup_empty_world,
    'planet': setup_planet,
    'layered': setup_layered_planet,
    'volcanic': setup_volcanic_island,
    'ice': setup_ice_world,
}


def get_scenario_names():
    """Return list of available scenario names."""
    return list(SCENARIOS.keys())


def setup_scenario(name: str, state: 'FluxState', material_db: MaterialDatabase):
    """
    Setup a named scenario.
    
    Args:
        name: Scenario name (or None for default)
        state: FluxState to configure
        material_db: MaterialDatabase for property updates
    """
    if name is None or name not in SCENARIOS:
        name = 'planet'  # Default scenario
        
    scenario_func = SCENARIOS[name]
    scenario_func(state, material_db)