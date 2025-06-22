"""
Material phase transition test scenarios for flux-based simulation.
"""

import numpy as np
from typing import Dict, Any

from .base import FluxTestScenario
from simulation import FluxSimulation
from materials import MaterialType


class WaterFreezingScenario(FluxTestScenario):
    """Test water freezing into ice at low temperature."""
    
    def get_name(self) -> str:
        return "water_freezing"
        
    def get_description(self) -> str:
        return "Water should freeze into ice when cooled below 273K"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create a container of water and cool it."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.SPACE] = 1.0
        
        # Create water region in center
        x_center = nx // 2
        x_width = nx // 4
        y_top = ny // 4
        y_bottom = 3 * ny // 4
        
        sim.state.vol_frac[MaterialType.SPACE, y_top:y_bottom, x_center-x_width:x_center+x_width] = 0.0
        sim.state.vol_frac[MaterialType.WATER, y_top:y_bottom, x_center-x_width:x_center+x_width] = 1.0
        
        # Set initial temperature just above freezing
        sim.state.temperature.fill(280.0)  # 7°C
        
        # Add cold regions to trigger freezing
        sim.state.temperature[:, :x_width] = 250.0  # -23°C on left
        sim.state.temperature[:, -x_width:] = 250.0  # -23°C on right
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if water has frozen where cold."""
        # Get material volumes
        water_volume = np.sum(sim.state.vol_frac[MaterialType.WATER])
        ice_volume = np.sum(sim.state.vol_frac[MaterialType.ICE])
        total_h2o = water_volume + ice_volume
        
        if total_h2o < 0.1:
            return {
                'success': False,
                'metrics': {},
                'message': "Water/ice disappeared!"
            }
            
        # Check ice formation in cold regions
        cold_mask = sim.state.temperature < 273.0
        ice_in_cold = np.sum(sim.state.vol_frac[MaterialType.ICE][cold_mask])
        water_in_cold = np.sum(sim.state.vol_frac[MaterialType.WATER][cold_mask])
        
        # Calculate freezing fraction in cold regions
        if water_in_cold + ice_in_cold > 0:
            freeze_fraction = ice_in_cold / (water_in_cold + ice_in_cold)
        else:
            freeze_fraction = 0.0
            
        # Check average temperature of ice regions
        ice_mask = sim.state.vol_frac[MaterialType.ICE] > 0.5
        if np.any(ice_mask):
            avg_ice_temp = np.mean(sim.state.temperature[ice_mask])
        else:
            avg_ice_temp = 999.0
            
        # Success: significant freezing has occurred in cold regions
        success = freeze_fraction > 0.5 and avg_ice_temp < 273.0
        
        return {
            'success': success,
            'metrics': {
                'water_volume': water_volume,
                'ice_volume': ice_volume,
                'freeze_fraction': freeze_fraction,
                'avg_ice_temp': avg_ice_temp,
                'time': sim.state.time,
            },
            'message': f"Freeze fraction: {freeze_fraction:.1%}, Ice temp: {avg_ice_temp:.1f}K"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'temperature',
            'highlight_materials': [MaterialType.WATER, MaterialType.ICE],
        }


class IceMeltingScenario(FluxTestScenario):
    """Test ice melting into water at high temperature."""
    
    def get_name(self) -> str:
        return "ice_melting"
        
    def get_description(self) -> str:
        return "Ice should melt into water when heated above 273K"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create ice blocks and heat them."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.SPACE] = 1.0
        
        # Create ice blocks
        block_size = 10
        spacing = 20
        
        for i in range(3):
            x_center = (i + 1) * nx // 4
            y_center = ny // 2
            
            x_min = max(0, x_center - block_size // 2)
            x_max = min(nx, x_center + block_size // 2)
            y_min = max(0, y_center - block_size // 2)
            y_max = min(ny, y_center + block_size // 2)
            
            sim.state.vol_frac[MaterialType.SPACE, y_min:y_max, x_min:x_max] = 0.0
            sim.state.vol_frac[MaterialType.ICE, y_min:y_max, x_min:x_max] = 1.0
        
        # Set initial temperature below freezing
        sim.state.temperature.fill(250.0)  # -23°C
        
        # Add heat sources
        sim.state.temperature[ny//2-5:ny//2+5, :] = 300.0  # 27°C horizontal band
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if ice has melted where warm."""
        # Get material volumes
        water_volume = np.sum(sim.state.vol_frac[MaterialType.WATER])
        ice_volume = np.sum(sim.state.vol_frac[MaterialType.ICE])
        total_h2o = water_volume + ice_volume
        
        if total_h2o < 0.1:
            return {
                'success': False,
                'metrics': {},
                'message': "Water/ice disappeared!"
            }
            
        # Check melting in warm regions
        warm_mask = sim.state.temperature > 273.0
        ice_in_warm = np.sum(sim.state.vol_frac[MaterialType.ICE][warm_mask])
        water_in_warm = np.sum(sim.state.vol_frac[MaterialType.WATER][warm_mask])
        
        # Calculate melt fraction in warm regions
        if water_in_warm + ice_in_warm > 0:
            melt_fraction = water_in_warm / (water_in_warm + ice_in_warm)
        else:
            melt_fraction = 0.0
            
        # Check if water has formed
        water_formed = water_volume > 0.1
        
        # Success: significant melting has occurred
        success = melt_fraction > 0.5 and water_formed
        
        return {
            'success': success,
            'metrics': {
                'water_volume': water_volume,
                'ice_volume': ice_volume,
                'melt_fraction': melt_fraction,
                'time': sim.state.time,
            },
            'message': f"Melt fraction: {melt_fraction:.1%}, Water: {water_volume:.1f}"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'temperature',
            'highlight_materials': [MaterialType.WATER, MaterialType.ICE],
        }


class WaterEvaporationScenario(FluxTestScenario):
    """Test water evaporation at high temperature."""
    
    def get_name(self) -> str:
        return "water_evaporation"
        
    def get_description(self) -> str:
        return "Water should evaporate into water vapor when heated above 373K"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create water and heat it to boiling."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid with air
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.AIR] = 1.0
        
        # Create shallow water layer at bottom
        water_height = ny // 8
        sim.state.vol_frac[MaterialType.AIR, -water_height:, :] = 0.0
        sim.state.vol_frac[MaterialType.WATER, -water_height:, :] = 1.0
        
        # Set temperature gradient - hot at bottom
        for j in range(ny):
            temp = 280.0 + (j / ny) * 120.0  # 280K to 400K
            sim.state.temperature[j, :] = temp
            
        # Make bottom very hot to ensure boiling
        sim.state.temperature[-5:, :] = 400.0  # 127°C
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if water has evaporated."""
        # Get material volumes
        water_volume = np.sum(sim.state.vol_frac[MaterialType.WATER])
        vapor_volume = np.sum(sim.state.vol_frac[MaterialType.WATER_VAPOR])
        total_h2o = water_volume + vapor_volume
        
        initial_water = sim.state.nx * (sim.state.ny // 8)  # Approximate initial water cells
        
        if total_h2o < 0.1:
            return {
                'success': False,
                'metrics': {},
                'message': "Water/vapor disappeared!"
            }
            
        # Check vapor formation
        vapor_formed = vapor_volume > 0.1
        evaporation_fraction = vapor_volume / total_h2o if total_h2o > 0 else 0.0
        
        # Check vapor location (should rise)
        if vapor_volume > 0:
            vapor_mask = sim.state.vol_frac[MaterialType.WATER_VAPOR] > 0.1
            y_indices = np.where(np.any(vapor_mask, axis=1))[0]
            if len(y_indices) > 0:
                avg_vapor_y = np.mean(y_indices)
                vapor_risen = avg_vapor_y < sim.state.ny * 0.75  # Vapor in upper 3/4
            else:
                vapor_risen = False
        else:
            vapor_risen = False
            avg_vapor_y = sim.state.ny
            
        # Success: significant evaporation and vapor has risen
        success = evaporation_fraction > 0.1 and vapor_risen
        
        return {
            'success': success,
            'metrics': {
                'water_volume': water_volume,
                'vapor_volume': vapor_volume,
                'evaporation_fraction': evaporation_fraction,
                'avg_vapor_y': avg_vapor_y,
                'vapor_risen': vapor_risen,
                'time': sim.state.time,
            },
            'message': f"Evaporation: {evaporation_fraction:.1%}, Vapor risen: {vapor_risen}"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'temperature',
            'highlight_materials': [MaterialType.WATER, MaterialType.WATER_VAPOR],
        }


class RockMeltingScenario(FluxTestScenario):
    """Test rock melting into magma at high temperature."""
    
    def get_name(self) -> str:
        return "rock_melting"
        
    def get_description(self) -> str:
        return "Rock should melt into magma when heated above 1473K"
        
    def setup(self, sim: FluxSimulation) -> None:
        """Create rock layers with extreme heat source."""
        nx, ny = sim.state.nx, sim.state.ny
        
        # Clear the grid
        sim.state.vol_frac.fill(0.0)
        sim.state.vol_frac[MaterialType.SPACE] = 1.0
        
        # Create rock layers
        rock_height = ny // 2
        sim.state.vol_frac[MaterialType.SPACE, -rock_height:, :] = 0.0
        sim.state.vol_frac[MaterialType.ROCK, -rock_height:, :] = 1.0
        
        # Add uranium heat source at bottom center
        uranium_x = nx // 2
        uranium_width = 5
        uranium_height = 5
        
        sim.state.vol_frac[MaterialType.ROCK, -uranium_height:, uranium_x-uranium_width:uranium_x+uranium_width] = 0.0
        sim.state.vol_frac[MaterialType.URANIUM, -uranium_height:, uranium_x-uranium_width:uranium_x+uranium_width] = 1.0
        
        # Set initial temperature
        sim.state.temperature.fill(300.0)  # Room temperature
        
        # Update properties
        sim.state.normalize_volume_fractions()
        sim.state.update_mixture_properties(sim.material_db)
        
    def evaluate(self, sim: FluxSimulation) -> Dict[str, Any]:
        """Check if rock has melted near heat source."""
        # Get material volumes
        rock_volume = np.sum(sim.state.vol_frac[MaterialType.ROCK])
        magma_volume = np.sum(sim.state.vol_frac[MaterialType.MAGMA])
        uranium_volume = np.sum(sim.state.vol_frac[MaterialType.URANIUM])
        
        # Check for melting
        magma_formed = magma_volume > 0.1
        
        # Check temperature near uranium
        uranium_mask = sim.state.vol_frac[MaterialType.URANIUM] > 0.5
        if np.any(uranium_mask):
            # Get neighboring cells
            y_idx, x_idx = np.where(uranium_mask)
            neighbor_temps = []
            for y, x in zip(y_idx, x_idx):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < sim.state.ny and 0 <= nx < sim.state.nx:
                            neighbor_temps.append(sim.state.temperature[ny, nx])
            max_temp = np.max(neighbor_temps) if neighbor_temps else 0.0
        else:
            max_temp = 0.0
            
        # Check if magma is hot
        magma_mask = sim.state.vol_frac[MaterialType.MAGMA] > 0.5
        if np.any(magma_mask):
            avg_magma_temp = np.mean(sim.state.temperature[magma_mask])
        else:
            avg_magma_temp = 0.0
            
        # Success: magma has formed and is hot
        success = magma_formed and avg_magma_temp > 1200.0
        
        return {
            'success': success,
            'metrics': {
                'rock_volume': rock_volume,
                'magma_volume': magma_volume,
                'uranium_volume': uranium_volume,
                'max_temp': max_temp,
                'avg_magma_temp': avg_magma_temp,
                'time': sim.state.time,
            },
            'message': f"Magma: {magma_volume:.1f}, Max T: {max_temp:.0f}K, Magma T: {avg_magma_temp:.0f}K"
        }
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        return {
            'preferred_display_mode': 'temperature',
            'highlight_materials': [MaterialType.ROCK, MaterialType.MAGMA, MaterialType.URANIUM],
        }