"""
Thermal behavior test scenarios for flux-based simulation.
"""

import numpy as np
from typing import Dict, Any

from .base import FluxTestScenario
from simulation import FluxSimulation
from materials import MaterialType


class HeatDiffusionScenario(FluxTestScenario):
    """Test heat diffusion through different materials."""
    
    def get_name(self) -> str:
        return "heat_diffusion"
        
    def get_description(self) -> str:
        return "Heat should diffuse from hot to cold regions following thermal conductivity"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create layers with different thermal properties."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.SPACE] = 1.0
        
        # Create three vertical bands: rock, water, air
        band_width = nx // 3
        
        # Rock on left (low conductivity)
        sim.state.vol_frac[MaterialType.SPACE, :, :band_width] = 0.0
        sim.state.vol_frac[MaterialType.ROCK, :, :band_width] = 1.0
        
        # Water in middle (medium conductivity)
        sim.state.vol_frac[MaterialType.SPACE, :, band_width:2*band_width] = 0.0
        sim.state.vol_frac[MaterialType.WATER, :, band_width:2*band_width] = 1.0
        
        # Air on right (very low conductivity)
        sim.state.vol_frac[MaterialType.SPACE, :, 2*band_width:] = 0.0
        sim.state.vol_frac[MaterialType.AIR, :, 2*band_width:] = 1.0
        
        # Set temperature gradient
        sim.state.temperature.fill(273.0)  # 0°C baseline
        
        # Hot stripe in the middle
        hot_y = ny // 2
        hot_width = 5
        sim.state.temperature[hot_y-hot_width:hot_y+hot_width, :] = 373.0  # 100°C
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check heat diffusion patterns."""
        nx, ny = sim.state.nx, sim.state.ny
        band_width = nx // 3
        
        # Calculate temperature spreads in each material
        rock_temps = sim.state.temperature[:, :band_width]
        water_temps = sim.state.temperature[:, band_width:2*band_width]
        air_temps = sim.state.temperature[:, 2*band_width:]
        
        # Calculate vertical temperature standard deviations
        rock_spread = np.std(rock_temps)
        water_spread = np.std(water_temps)
        air_spread = np.std(air_temps)
        
        # Get average temperatures
        rock_avg = np.mean(rock_temps)
        water_avg = np.mean(water_temps)
        air_avg = np.mean(air_temps)
        
        # Check if heat has diffused (reduced peak temperature)
        max_temp = np.max(sim.state.temperature)
        initial_max = 373.0
        diffusion_occurred = max_temp < initial_max - 5.0
        
        # Thermal conductivities: water > rock > air
        # So we expect: water_spread > rock_spread > air_spread
        conductivity_order = water_spread > rock_spread > air_spread
        
        # Success: heat has diffused and follows material properties
        success = diffusion_occurred and conductivity_order
        
        return {
            'success': success,
            'metrics': {
                'rock_spread': rock_spread,
                'water_spread': water_spread,
                'air_spread': air_spread,
                'rock_avg': rock_avg,
                'water_avg': water_avg,
                'air_avg': air_avg,
                'max_temp': max_temp,
                'conductivity_order': conductivity_order,
                'time': sim.state.time,
            },
            'message': f"Spreads - Rock: {rock_spread:.1f}, Water: {water_spread:.1f}, Air: {air_spread:.1f}"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'temperature',
            'highlight_materials': [MaterialType.ROCK, MaterialType.WATER, MaterialType.AIR],
        }


class UraniumHeatingScenario(FluxTestScenario):
    """Test radioactive heating from uranium."""
    
    def get_name(self) -> str:
        return "uranium_heating"
        
    def get_description(self) -> str:
        return "Uranium should generate heat and warm surrounding materials"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create uranium deposits in rock."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid with rock
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.ROCK] = 1.0
        
        # Create uranium deposits
        deposit_size = 5
        deposits = [
            (nx // 4, ny // 4),
            (3 * nx // 4, ny // 4),
            (nx // 2, 3 * ny // 4),
        ]
        
        for cx, cy in deposits:
            x_min = max(0, cx - deposit_size // 2)
            x_max = min(nx, cx + deposit_size // 2)
            y_min = max(0, cy - deposit_size // 2)
            y_max = min(ny, cy + deposit_size // 2)
            
            sim.state.vol_frac[MaterialType.ROCK, y_min:y_max, x_min:x_max] = 0.0
            sim.state.vol_frac[MaterialType.URANIUM, y_min:y_max, x_min:x_max] = 1.0
        
        # Set uniform cool temperature
        sim.state.temperature.fill(273.0)  # 0°C
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if uranium has heated surroundings."""
        # Get temperature near uranium
        uranium_mask = sim.state.vol_frac[MaterialType.URANIUM] > 0.5
        
        if not np.any(uranium_mask):
            return {
                'success': False,
                'metrics': {},
                'message': "No uranium found!"
            }
            
        # Calculate average temperatures
        uranium_temp = np.mean(sim.state.temperature[uranium_mask])
        rock_temp = np.mean(sim.state.temperature[sim.state.vol_frac[MaterialType.ROCK] > 0.5])
        overall_temp = np.mean(sim.state.temperature)
        
        # Check temperature gradients near uranium
        y_idx, x_idx = np.where(uranium_mask)
        nearby_temps = []
        
        for y, x in zip(y_idx, x_idx):
            # Check cells within 3 grid points
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny_idx, nx_idx = y + dy, x + dx
                    if (0 <= ny_idx < sim.state.ny and 0 <= nx_idx < sim.state.nx and
                        not uranium_mask[ny_idx, nx_idx]):
                        nearby_temps.append(sim.state.temperature[ny_idx, nx_idx])
                        
        avg_nearby_temp = np.mean(nearby_temps) if nearby_temps else 273.0
        
        # Calculate total heat generated
        initial_temp = 273.0
        temp_increase = overall_temp - initial_temp
        
        # Success: uranium is hot and has heated surroundings
        success = (uranium_temp > initial_temp + 10.0 and 
                  avg_nearby_temp > initial_temp + 5.0)
        
        return {
            'success': success,
            'metrics': {
                'uranium_temp': uranium_temp,
                'rock_temp': rock_temp,
                'overall_temp': overall_temp,
                'avg_nearby_temp': avg_nearby_temp,
                'temp_increase': temp_increase,
                'time': sim.state.time,
            },
            'message': f"Uranium: {uranium_temp:.1f}K, Nearby: {avg_nearby_temp:.1f}K, Increase: {temp_increase:.1f}K"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'temperature',
            'highlight_materials': [MaterialType.URANIUM],
        }


class SolarHeatingScenario(FluxTestScenario):
    """Test solar heating at surface."""
    
    def get_name(self) -> str:
        return "solar_heating"
        
    def get_description(self) -> str:
        return "Surface materials should heat up from solar radiation"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create surface with different materials."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid with space
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.SPACE] = 1.0
        
        # Create ground level at middle
        ground_y = ny // 2
        
        # Different materials across surface
        section_width = nx // 4
        
        # Ice (high albedo)
        sim.state.vol_frac[MaterialType.SPACE, ground_y:, :section_width] = 0.0
        sim.state.vol_frac[MaterialType.ICE, ground_y:, :section_width] = 1.0
        
        # Water (medium albedo)
        sim.state.vol_frac[MaterialType.SPACE, ground_y:, section_width:2*section_width] = 0.0
        sim.state.vol_frac[MaterialType.WATER, ground_y:, section_width:2*section_width] = 1.0
        
        # Rock (low albedo)
        sim.state.vol_frac[MaterialType.SPACE, ground_y:, 2*section_width:3*section_width] = 0.0
        sim.state.vol_frac[MaterialType.ROCK, ground_y:, 2*section_width:3*section_width] = 1.0
        
        # Sand (medium-low albedo)
        sim.state.vol_frac[MaterialType.SPACE, ground_y:, 3*section_width:] = 0.0
        sim.state.vol_frac[MaterialType.SAND, ground_y:, 3*section_width:] = 1.0
        
        # Set uniform cool temperature
        sim.state.temperature.fill(250.0)  # -23°C
        
        # Enable solar heating in physics
        sim.physics.solar_constant = 1361.0  # W/m²
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check differential heating based on albedo."""
        nx = sim.state.nx
        section_width = nx // 4
        ground_y = sim.state.ny // 2
        
        # Get surface temperatures for each material
        ice_surf_temp = np.mean(sim.state.temperature[ground_y, :section_width])
        water_surf_temp = np.mean(sim.state.temperature[ground_y, section_width:2*section_width])
        rock_surf_temp = np.mean(sim.state.temperature[ground_y, 2*section_width:3*section_width])
        sand_surf_temp = np.mean(sim.state.temperature[ground_y, 3*section_width:])
        
        # Calculate temperature increases
        initial_temp = 250.0
        ice_increase = ice_surf_temp - initial_temp
        water_increase = water_surf_temp - initial_temp
        rock_increase = rock_surf_temp - initial_temp
        sand_increase = sand_surf_temp - initial_temp
        
        # Expected order based on albedo (low albedo = more heating)
        # Rock (0.3) > Sand (0.4) > Water (0.5) > Ice (0.9)
        heating_order = rock_increase > sand_increase > water_increase > ice_increase
        
        # Check if any heating occurred
        max_increase = max(ice_increase, water_increase, rock_increase, sand_increase)
        heating_occurred = max_increase > 5.0
        
        # Success: differential heating based on albedo
        success = heating_occurred and heating_order
        
        return {
            'success': success,
            'metrics': {
                'ice_temp': ice_surf_temp,
                'water_temp': water_surf_temp,
                'rock_temp': rock_surf_temp,
                'sand_temp': sand_surf_temp,
                'ice_increase': ice_increase,
                'water_increase': water_increase,
                'rock_increase': rock_increase,
                'sand_increase': sand_increase,
                'heating_order_correct': heating_order,
                'time': sim.state.time,
            },
            'message': f"Increases - Rock: {rock_increase:.1f}K, Sand: {sand_increase:.1f}K, Water: {water_increase:.1f}K, Ice: {ice_increase:.1f}K"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'temperature',
            'highlight_materials': [MaterialType.ICE, MaterialType.WATER, MaterialType.ROCK, MaterialType.SAND],
        }


class RadiativeCoolingScenario(FluxTestScenario):
    """Test radiative cooling to space."""
    
    def get_name(self) -> str:
        return "radiative_cooling"
        
    def get_description(self) -> str:
        return "Hot surfaces should cool by radiating to space"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create hot surface exposed to space."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid with space
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.SPACE] = 1.0
        
        # Create hot rock layer at bottom
        rock_height = ny // 4
        sim.state.vol_frac[MaterialType.SPACE, -rock_height:, :] = 0.0
        sim.state.vol_frac[MaterialType.ROCK, -rock_height:, :] = 1.0
        
        # Set high temperature
        sim.state.temperature.fill(100.0)  # Cold space
        sim.state.temperature[-rock_height:, :] = 400.0  # Hot rock (127°C)
        
        # Disable solar heating to isolate cooling
        sim.physics.solar_constant = 0.0
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if surface has cooled."""
        ny = sim.state.ny
        rock_height = ny // 4
        
        # Get rock temperatures
        rock_mask = sim.state.vol_frac[MaterialType.ROCK] > 0.5
        if not np.any(rock_mask):
            return {
                'success': False,
                'metrics': {},
                'message': "No rock found!"
            }
            
        # Surface temperature (top of rock)
        surface_y = ny - rock_height
        surface_temp = np.mean(sim.state.temperature[surface_y, :])
        
        # Average rock temperature
        rock_temp = np.mean(sim.state.temperature[rock_mask])
        
        # Deep rock temperature (bottom)
        deep_temp = np.mean(sim.state.temperature[-1, :])
        
        # Calculate cooling
        initial_temp = 400.0
        surface_cooling = initial_temp - surface_temp
        avg_cooling = initial_temp - rock_temp
        
        # Check temperature gradient (surface should be cooler)
        gradient_correct = surface_temp < deep_temp
        
        # Success: significant cooling at surface
        success = surface_cooling > 20.0 and gradient_correct
        
        return {
            'success': success,
            'metrics': {
                'surface_temp': surface_temp,
                'rock_temp': rock_temp,
                'deep_temp': deep_temp,
                'surface_cooling': surface_cooling,
                'avg_cooling': avg_cooling,
                'gradient_correct': gradient_correct,
                'time': sim.state.time,
            },
            'message': f"Surface: {surface_temp:.1f}K (cooled {surface_cooling:.1f}K), Deep: {deep_temp:.1f}K"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'temperature',
            'highlight_materials': [MaterialType.ROCK],
        }