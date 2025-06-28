"""
Fluid behavior test scenarios for flux-based simulation.
"""

import numpy as np
from typing import Dict, Any

from .base import FluxTestScenario
from simulation import FluxSimulation
from materials import MaterialType


class HydrostaticEquilibriumScenario(FluxTestScenario):
    """Test that a water column reaches hydrostatic equilibrium."""
    
    def get_name(self) -> str:
        return "hydrostatic_equilibrium"
        
    def get_description(self) -> str:
        return "Water column should reach hydrostatic equilibrium with correct pressure gradient"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create a column of water."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.SPACE] = 1.0
        
        # Create water column in center
        x_center = nx // 2
        x_width = nx // 4
        y_bottom = int(ny * 0.8)
        
        # Fill with water
        sim.state.vol_frac[MaterialType.SPACE, :y_bottom, x_center-x_width:x_center+x_width] = 0.0
        sim.state.vol_frac[MaterialType.WATER, :y_bottom, x_center-x_width:x_center+x_width] = 1.0
        
        # Set uniform temperature
        sim.state.temperature.fill(293.0)  # 20°C
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if hydrostatic equilibrium is achieved."""
        # Find water region
        water_mask = sim.state.vol_frac[MaterialType.WATER] > 0.5
        
        if not np.any(water_mask):
            return {
                'success': False,
                'metrics': {},
                'message': "No water found in simulation"
            }
            
        # Get average velocities in water
        avg_vx = np.mean(np.abs(sim.state.velocity_x[water_mask]))
        avg_vy = np.mean(np.abs(sim.state.velocity_y[water_mask]))
        max_v = np.max(np.sqrt(
            sim.state.velocity_x[water_mask]**2 + 
            sim.state.velocity_y[water_mask]**2
        ))
        
        # Check pressure gradient
        # Should be approximately dP/dy = rho * g
        y_indices = np.where(water_mask)[0]
        if len(np.unique(y_indices)) > 1:
            # Simple pressure gradient check
            y_min, y_max = y_indices.min(), y_indices.max()
            if y_max > y_min:
                p_bottom = np.mean(sim.state.pressure[y_max, water_mask[y_max, :]])
                p_top = np.mean(sim.state.pressure[y_min, water_mask[y_min, :]])
                
                height = (y_max - y_min) * sim.state.dx
                expected_dp = 1000.0 * 9.81 * height  # rho_water * g * h
                actual_dp = p_bottom - p_top
                
                pressure_error = abs(actual_dp - expected_dp) / (expected_dp + 1e-10)
            else:
                pressure_error = 1.0
        else:
            pressure_error = 1.0
            
        # Success criteria
        velocity_threshold = 0.01  # m/s
        pressure_threshold = 0.1   # 10% error
        
        success = (avg_vx < velocity_threshold and 
                  avg_vy < velocity_threshold and
                  pressure_error < pressure_threshold)
                  
        return {
            'success': success,
            'metrics': {
                'avg_vx': avg_vx,
                'avg_vy': avg_vy,
                'max_velocity': max_v,
                'pressure_error': pressure_error,
                'time': sim.state.time,
            },
            'message': f"v_avg=({avg_vx:.3f}, {avg_vy:.3f}) m/s, P_err={pressure_error:.2%}"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'pressure',
            'highlight_materials': [MaterialType.WATER],
        }


class WaterDropFallScenario(FluxTestScenario):
    """Test water drop falling under gravity."""
    
    def get_name(self) -> str:
        return "water_drop_fall"
        
    def get_description(self) -> str:
        return "Water drop should fall under gravity and spread on impact"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create a water drop above ground."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.SPACE] = 1.0
        
        # Create ground
        ground_height = int(ny * 0.8)
        sim.state.vol_frac[MaterialType.SPACE, ground_height:, :] = 0.0
        sim.state.vol_frac[MaterialType.ROCK, ground_height:, :] = 1.0
        
        # Create water drop
        cx, cy = nx // 2, ny // 4
        radius = 5
        
        y_grid, x_grid = np.ogrid[:ny, :nx]
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        drop_mask = dist < radius
        
        sim.state.vol_frac[MaterialType.SPACE][drop_mask] = 0.0
        sim.state.vol_frac[MaterialType.WATER][drop_mask] = 1.0
        
        # Set temperature
        sim.state.temperature.fill(293.0)
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if water has fallen and spread."""
        # Get water center of mass
        water_phi = sim.state.vol_frac[MaterialType.WATER]
        total_water = np.sum(water_phi)
        
        if total_water < 0.1:
            return {
                'success': False,
                'metrics': {},
                'message': "Water disappeared!"
            }
            
        y_grid, x_grid = np.mgrid[:sim.state.ny, :sim.state.nx]
        water_y = np.sum(water_phi * y_grid) / total_water
        
        # Check if water has fallen
        initial_y = sim.state.ny // 4
        distance_fallen = (water_y - initial_y) * sim.state.dx
        
        # Check spreading
        water_cells = np.sum(water_phi > 0.1)
        initial_cells = np.pi * 5**2  # Initial drop area
        spread_ratio = water_cells / initial_cells
        
        # Success: water has fallen at least 10m and spread
        success = distance_fallen > 10.0 and spread_ratio > 1.5
        
        return {
            'success': success,
            'metrics': {
                'distance_fallen': distance_fallen,
                'spread_ratio': spread_ratio,
                'water_volume': total_water * sim.state.dx**2,
                'time': sim.state.time,
            },
            'message': f"Fallen {distance_fallen:.1f}m, spread {spread_ratio:.1f}x"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'material_composite',
            'highlight_materials': [MaterialType.WATER],
        }


class BuoyancyScenario(FluxTestScenario):
    """Test buoyancy - less dense materials should rise."""
    
    def get_name(self) -> str:
        return "buoyancy"
        
    def get_description(self) -> str:
        return "Ice should float on water due to lower density"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create ice block underwater."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.SPACE] = 1.0
        
        # Fill bottom half with water
        water_level = ny // 2
        sim.state.vol_frac[MaterialType.SPACE, water_level:, :] = 0.0
        sim.state.vol_frac[MaterialType.WATER, water_level:, :] = 1.0
        
        # Place ice block at bottom
        ice_cx = nx // 2
        ice_cy = int(ny * 0.75)
        ice_radius = 8
        
        y_grid, x_grid = np.ogrid[:ny, :nx]
        dist = np.sqrt((x_grid - ice_cx)**2 + (y_grid - ice_cy)**2)
        ice_mask = dist < ice_radius
        
        # Replace water with ice
        sim.state.vol_frac[MaterialType.WATER][ice_mask] = 0.0
        sim.state.vol_frac[MaterialType.ICE][ice_mask] = 1.0
        
        # Set temperature (below freezing to prevent melting)
        sim.state.temperature.fill(250.0)
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if ice has risen."""
        # Get ice center of mass
        ice_phi = sim.state.vol_frac[MaterialType.ICE]
        total_ice = np.sum(ice_phi)
        
        if total_ice < 0.1:
            return {
                'success': False,
                'metrics': {},
                'message': "Ice disappeared (melted?)"
            }
            
        y_grid, _ = np.mgrid[:sim.state.ny, :sim.state.nx]
        ice_y = np.sum(ice_phi * y_grid) / total_ice
        
        # Check if ice has risen
        initial_y = int(sim.state.ny * 0.75)
        distance_risen = (initial_y - ice_y) * sim.state.dx
        
        # Check if ice is at surface
        water_rows = np.where(np.any(sim.state.vol_frac[MaterialType.WATER] > 0.1, axis=1))[0]
        if len(water_rows) > 0:
            water_surface_y = np.min(water_rows)
            at_surface = ice_y < water_surface_y + 5
        else:
            water_surface_y = sim.state.ny
            at_surface = False
        
        # Success: ice has risen at least 5m or reached surface
        success = distance_risen > 5.0 or at_surface
        
        return {
            'success': success,
            'metrics': {
                'distance_risen': distance_risen,
                'ice_y': ice_y,
                'water_surface_y': water_surface_y,
                'at_surface': at_surface,
                'time': sim.state.time,
            },
            'message': f"Ice risen {distance_risen:.1f}m, at_surface={at_surface}"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'material_dominant',
            'highlight_materials': [MaterialType.ICE, MaterialType.WATER],
        }


class RockSinkingScenario(FluxTestScenario):
    """Test that denser materials sink in water."""
    
    def get_name(self) -> str:
        return "rock_sinking"
        
    def get_description(self) -> str:
        return "Rock should sink in water due to higher density"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create rock block above water."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.AIR] = 1.0
        
        # Fill bottom 2/3 with water
        water_level = 2 * ny // 3
        sim.state.vol_frac[MaterialType.AIR, water_level:, :] = 0.0
        sim.state.vol_frac[MaterialType.WATER, water_level:, :] = 1.0
        
        # Place rock block above water
        rock_cx = nx // 2
        rock_cy = water_level - 10  # Start above water
        rock_radius = 6
        
        y_grid, x_grid = np.ogrid[:ny, :nx]
        dist = np.sqrt((x_grid - rock_cx)**2 + (y_grid - rock_cy)**2)
        rock_mask = dist < rock_radius
        
        # Replace air with rock
        sim.state.vol_frac[MaterialType.AIR][rock_mask] = 0.0
        sim.state.vol_frac[MaterialType.ROCK][rock_mask] = 1.0
        
        # Set temperature
        sim.state.temperature.fill(293.0)  # Room temperature
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if rock has sunk."""
        # Get rock center of mass
        rock_phi = sim.state.vol_frac[MaterialType.ROCK]
        total_rock = np.sum(rock_phi)
        
        if total_rock < 0.1:
            return {
                'success': False,
                'metrics': {},
                'message': "Rock disappeared!"
            }
            
        y_grid, _ = np.mgrid[:sim.state.ny, :sim.state.nx]
        rock_y = np.sum(rock_phi * y_grid) / total_rock
        
        # Check if rock has sunk
        water_level = 2 * sim.state.ny // 3
        initial_y = water_level - 10
        distance_sunk = (rock_y - initial_y) * sim.state.dx
        
        # Check if rock is in water
        water_at_rock = np.mean(sim.state.vol_frac[MaterialType.WATER][rock_phi > 0.5])
        in_water = water_at_rock > 0.5
        
        # Success: rock has sunk at least 10m and is in water
        success = distance_sunk > 10.0 and in_water
        
        return {
            'success': success,
            'metrics': {
                'distance_sunk': distance_sunk,
                'rock_y': rock_y,
                'in_water': in_water,
                'water_at_rock': water_at_rock,
                'time': sim.state.time,
            },
            'message': f"Rock sunk {distance_sunk:.1f}m, in_water={in_water}"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'material_dominant',
            'highlight_materials': [MaterialType.ROCK, MaterialType.WATER],
        }