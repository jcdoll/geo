"""Fluid dynamics test scenarios (water, magma, etc.)."""

import numpy as np
from typing import Dict, Any
from scipy import ndimage

from .base import TestScenario
from materials import MaterialType
from geo_game import GeoGame


class WaterConservationScenario(TestScenario):
    """Test water conservation during fluid flow."""
    
    def __init__(self, grid_size: int = 60, water_fraction: float = 0.3, 
                 tolerance_percent: float = 1.0, max_steps: int = 100, **kwargs):
        """Initialize water conservation scenario.
        
        Args:
            grid_size: Size of the simulation grid
            water_fraction: Fraction of surface to cover with water
            tolerance_percent: Allowed percentage variation in water
            max_steps: Maximum steps for timeout
        """
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.water_fraction = water_fraction
        self.tolerance_percent = tolerance_percent
        self.max_steps = max_steps
        
    def get_name(self) -> str:
        return f"water_conservation_{self.grid_size}x{self.grid_size}"
        
    def get_description(self) -> str:
        return f"Tests water conservation in {self.grid_size}x{self.grid_size} grid"
        
    def setup(self, sim: GeoGame) -> None:
        """Create a controlled environment with water."""
        # Clear to space
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 0.0
        
        # Create simple rocky ground with atmosphere
        ground_height = sim.height * 3 // 4
        sim.material_types[ground_height:, :] = MaterialType.BASALT
        sim.material_types[:ground_height, :] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Add water layer on top of ground
        water_depth = 3
        water_start = ground_height - water_depth
        sim.material_types[water_start:ground_height, :] = MaterialType.WATER
        
        # Create some terrain variation to test flow
        terrain_variation = 5
        for x in range(0, sim.width, 10):
            height_offset = np.random.RandomState(42 + x).randint(-terrain_variation, terrain_variation)
            new_ground = ground_height + height_offset
            if 0 < new_ground < sim.height:
                if height_offset > 0:  # Mountain
                    sim.material_types[ground_height:new_ground, x:x+5] = MaterialType.BASALT
                else:  # Valley
                    sim.material_types[new_ground:ground_height, x:x+5] = MaterialType.WATER
        
        # Enable relevant physics
        sim.enable_heat_diffusion = False  # Not testing heat
        sim.enable_self_gravity = False  # Simplify test
        sim.external_gravity = (0, 9.81)  # Earth gravity
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def count_water_cells(self, sim: GeoGame) -> int:
        """Count all water-bearing cells (water, ice, vapor)."""
        mask = (
            (sim.material_types == MaterialType.WATER) |
            (sim.material_types == MaterialType.ICE) |
            (sim.material_types == MaterialType.WATER_VAPOR)
        )
        return int(np.sum(mask))
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate water conservation."""
        current_count = self.count_water_cells(sim)
        initial_count = self.initial_state.get('water_count', 0)
        
        if initial_count == 0:
            return {
                'success': False,
                'metrics': {'water_count': current_count},
                'message': 'No initial water count stored!'
            }
            
        change = current_count - initial_count
        percent_change = (change / initial_count * 100) if initial_count > 0 else 0
        
        tolerance = int(initial_count * self.tolerance_percent / 100)
        within_tolerance = abs(change) <= tolerance
        
        # Check for timeout
        if self.check_timeout():
            within_tolerance = False
            
        return {
            'success': within_tolerance,
            'metrics': {
                'initial_count': initial_count,
                'current_count': current_count,
                'change': change,
                'percent_change': percent_change,
            },
            'message': f"Water: {initial_count} → {current_count} (Δ={change:+d}, {percent_change:+.1f}%)"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial state including water count."""
        super().store_initial_state(sim)
        self.initial_state['water_count'] = self.count_water_cells(sim)
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [MaterialType.WATER, MaterialType.ICE, MaterialType.WATER_VAPOR],
            'preferred_display_mode': 'material',
            'show_metrics': ['water_count', 'percent_change'],
        }


class WaterDropletCoalescenceScenario(TestScenario):
    """Test surface tension causing water droplets to coalesce."""
    
    def __init__(self, num_droplets: int = 5, droplet_size: int = 3, 
                 grid_size: int = 60, **kwargs):
        """Initialize droplet coalescence scenario."""
        super().__init__(**kwargs)
        self.num_droplets = num_droplets
        self.droplet_size = droplet_size
        self.grid_size = grid_size
        
    def get_name(self) -> str:
        return f"water_coalescence_{self.num_droplets}_droplets"
        
    def get_description(self) -> str:
        return f"Tests surface tension with {self.num_droplets} water droplets"
        
    def setup(self, sim: GeoGame) -> None:
        """Create water droplets in air."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Create droplets at random positions
        rng = np.random.RandomState(42)
        margin = self.droplet_size + 5
        
        for i in range(self.num_droplets):
            cx = rng.randint(margin, sim.width - margin)
            cy = rng.randint(margin, sim.height - margin)
            
            # Create circular droplet
            for dy in range(-self.droplet_size, self.droplet_size + 1):
                for dx in range(-self.droplet_size, self.droplet_size + 1):
                    if dx*dx + dy*dy <= self.droplet_size * self.droplet_size:
                        y, x = cy + dy, cx + dx
                        if 0 <= y < sim.height and 0 <= x < sim.width:
                            sim.material_types[y, x] = MaterialType.WATER
        
        # Enable physics
        sim.enable_heat_diffusion = False
        sim.enable_self_gravity = False
        sim.external_gravity = (0, 0)  # No gravity - test pure surface tension
        sim.enable_surface_tension = True
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def count_water_features(self, sim: GeoGame) -> int:
        """Count number of disconnected water features."""
        water_mask = sim.material_types == MaterialType.WATER
        labeled, num_features = ndimage.label(water_mask)
        return num_features
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check if droplets are coalescing."""
        num_features = self.count_water_features(sim)
        initial_features = self.initial_state.get('initial_features', self.num_droplets)
        
        # Success if features have decreased (coalescence)
        success = num_features < initial_features
        
        # Calculate cohesion metric
        water_mask = sim.material_types == MaterialType.WATER
        if np.any(water_mask):
            water_coords = np.argwhere(water_mask)
            center_y = np.mean(water_coords[:, 0])
            center_x = np.mean(water_coords[:, 1])
            distances = np.sqrt(np.sum((water_coords - [center_y, center_x])**2, axis=1))
            avg_distance = np.mean(distances)
        else:
            avg_distance = 0
            
        return {
            'success': success,
            'metrics': {
                'num_features': num_features,
                'initial_features': initial_features,
                'avg_distance': avg_distance,
            },
            'message': f"Water features: {initial_features} → {num_features}, compactness: {avg_distance:.1f}"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial feature count."""
        super().store_initial_state(sim)
        self.initial_state['initial_features'] = self.count_water_features(sim)


class MagmaFlowScenario(TestScenario):
    """Test magma flow and cooling behavior."""
    
    def __init__(self, volcano_size: int = 10, grid_size: int = 80, **kwargs):
        """Initialize magma flow scenario."""
        super().__init__(**kwargs)
        self.volcano_size = volcano_size
        self.grid_size = grid_size
        
    def get_name(self) -> str:
        return f"magma_flow_volcano_{self.volcano_size}"
        
    def get_description(self) -> str:
        return "Tests magma flow from volcano and cooling to basalt"
        
    def setup(self, sim: GeoGame) -> None:
        """Create volcano with magma source."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Create ground
        ground_level = sim.height * 3 // 4
        sim.material_types[ground_level:, :] = MaterialType.BASALT
        sim.temperature[ground_level:, :] = 350.0  # Warm ground
        
        # Create volcano cone
        volcano_x = sim.width // 2
        volcano_height = self.volcano_size
        
        for y in range(volcano_height):
            width_at_height = (volcano_height - y) * 2
            y_pos = ground_level - y
            if y_pos >= 0:
                x_start = max(0, volcano_x - width_at_height)
                x_end = min(sim.width, volcano_x + width_at_height)
                sim.material_types[y_pos, x_start:x_end] = MaterialType.BASALT
                sim.temperature[y_pos, x_start:x_end] = 500.0  # Hot volcano
        
        # Add magma chamber at top
        chamber_y = ground_level - volcano_height + 2
        chamber_width = 3
        if chamber_y >= 0:
            x_start = max(0, volcano_x - chamber_width)
            x_end = min(sim.width, volcano_x + chamber_width)
            sim.material_types[chamber_y:chamber_y+3, x_start:x_end] = MaterialType.MAGMA
            sim.temperature[chamber_y:chamber_y+3, x_start:x_end] = 1500.0
        
        # Enable physics
        sim.enable_heat_diffusion = True
        sim.enable_material_melting = True
        sim.external_gravity = (0, 9.81)
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Monitor magma flow and cooling."""
        magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
        basalt_count = np.sum(sim.material_types == MaterialType.BASALT)
        
        # Check if magma has flowed (spread beyond initial chamber)
        magma_positions = np.argwhere(sim.material_types == MaterialType.MAGMA)
        if len(magma_positions) > 0:
            magma_spread = np.std(magma_positions[:, 1])  # Horizontal spread
            lowest_magma_y = np.max(magma_positions[:, 0])  # Lowest point
        else:
            magma_spread = 0
            lowest_magma_y = 0
            
        # Check temperature distribution
        magma_temps = sim.temperature[sim.material_types == MaterialType.MAGMA]
        avg_magma_temp = np.mean(magma_temps) if len(magma_temps) > 0 else 0
        
        # Success if magma has flowed and some has cooled
        initial_magma = self.initial_state.get('initial_magma', 0)
        has_flowed = magma_spread > 5
        has_cooled = magma_count < initial_magma
        
        return {
            'success': has_flowed and has_cooled,
            'metrics': {
                'magma_count': magma_count,
                'basalt_count': basalt_count,
                'magma_spread': magma_spread,
                'avg_magma_temp': avg_magma_temp,
                'lowest_magma_y': lowest_magma_y,
            },
            'message': f"Magma: {magma_count} cells, spread: {magma_spread:.1f}, temp: {avg_magma_temp:.0f}K"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial magma count."""
        super().store_initial_state(sim)
        self.initial_state['initial_magma'] = np.sum(sim.material_types == MaterialType.MAGMA)
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [MaterialType.MAGMA, MaterialType.BASALT],
            'preferred_display_mode': 'temperature',
            'show_metrics': ['magma_count', 'avg_magma_temp'],
        }


class WaterConservationStressScenario(WaterConservationScenario):
    """Aggressive water conservation test with many surface features."""
    
    def __init__(self, **kwargs):
        """Initialize stress test with more cavities."""
        super().__init__(
            cavity_count=100,
            cavity_radius_range=(2, 5),
            tolerance_percent=5.0,  # Allow more variation in stress test
            **kwargs
        )
        
    def get_name(self) -> str:
        return "water_conservation_stress"
        
    def get_description(self) -> str:
        return "Stress test with 100 surface cavities"


class WaterBlobScenario(TestScenario):
    """Test water blob behavior and surface tension."""
    
    def __init__(self, blob_width: int = 20, blob_height: int = 10,
                 grid_size: int = 60, **kwargs):
        """Initialize water blob test."""
        super().__init__(**kwargs)
        self.blob_width = blob_width
        self.blob_height = blob_height
        self.grid_size = grid_size
        
    def get_name(self) -> str:
        return f"water_blob_{self.blob_width}x{self.blob_height}"
        
    def get_description(self) -> str:
        return f"Tests water blob ({self.blob_width}x{self.blob_height}) cohesion"
        
    def setup(self, sim: GeoGame) -> None:
        """Create rectangular water blob in air."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Create water blob
        center_x = sim.width // 2
        center_y = sim.height // 2
        
        y_start = max(0, center_y - self.blob_height // 2)
        y_end = min(sim.height, center_y + self.blob_height // 2)
        x_start = max(0, center_x - self.blob_width // 2)
        x_end = min(sim.width, center_x + self.blob_width // 2)
        
        sim.material_types[y_start:y_end, x_start:x_end] = MaterialType.WATER
        
        # Enable physics
        sim.enable_heat_diffusion = False
        sim.enable_self_gravity = False
        sim.external_gravity = (0, 0)  # No gravity - test surface tension
        sim.enable_surface_tension = True
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check blob cohesion and shape evolution."""
        water_count = np.sum(sim.material_types == MaterialType.WATER)
        initial_count = self.initial_state.get('initial_water', 0)
        
        if water_count == 0:
            return {
                'success': False,
                'metrics': {'water_count': 0},
                'message': 'No water remaining!'
            }
            
        # Calculate blob compactness
        water_coords = np.argwhere(sim.material_types == MaterialType.WATER)
        center_y = np.mean(water_coords[:, 0])
        center_x = np.mean(water_coords[:, 1])
        
        # RMS distance from center
        distances = np.sqrt(np.sum((water_coords - [center_y, center_x])**2, axis=1))
        rms_distance = np.sqrt(np.mean(distances**2))
        
        # Initial RMS distance for reference
        initial_rms = self.initial_state.get('initial_rms', 0)
        
        # Count connected components
        labeled, num_components = ndimage.label(sim.material_types == MaterialType.WATER)
        
        # Success if blob stays together and becomes more compact
        water_conserved = water_count >= initial_count * 0.95
        stayed_together = num_components == 1
        became_compact = rms_distance < initial_rms * 1.1  # Allow 10% expansion
        
        success = water_conserved and stayed_together and became_compact
        
        return {
            'success': success,
            'metrics': {
                'water_count': water_count,
                'num_components': num_components,
                'rms_distance': rms_distance,
                'initial_rms': initial_rms,
                'compactness_ratio': rms_distance / initial_rms if initial_rms > 0 else 1,
            },
            'message': f"Water: {water_count}, Components: {num_components}, Compactness: {rms_distance:.1f}"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial blob metrics."""
        super().store_initial_state(sim)
        self.initial_state['initial_water'] = np.sum(sim.material_types == MaterialType.WATER)
        
        # Calculate initial RMS
        water_coords = np.argwhere(sim.material_types == MaterialType.WATER)
        if len(water_coords) > 0:
            center_y = np.mean(water_coords[:, 0])
            center_x = np.mean(water_coords[:, 1])
            distances = np.sqrt(np.sum((water_coords - [center_y, center_x])**2, axis=1))
            self.initial_state['initial_rms'] = np.sqrt(np.mean(distances**2))
        else:
            self.initial_state['initial_rms'] = 0


class WaterLineCollapseScenario(TestScenario):
    """Test water line collapse and flow behavior."""
    
    def __init__(self, line_thickness: int = 2, line_height: int = 30,
                 grid_size: int = 60, **kwargs):
        """Initialize water line test."""
        super().__init__(**kwargs)
        self.line_thickness = line_thickness
        self.line_height = line_height
        self.grid_size = grid_size
        
    def get_name(self) -> str:
        return f"water_line_collapse_h{self.line_height}"
        
    def get_description(self) -> str:
        return f"Tests collapse of {self.line_height}-cell tall water column"
        
    def setup(self, sim: GeoGame) -> None:
        """Create vertical water line."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Create ground
        ground_level = sim.height - 5
        sim.material_types[ground_level:, :] = MaterialType.BASALT
        
        # Create vertical water line
        center_x = sim.width // 2
        water_bottom = ground_level
        water_top = max(0, water_bottom - self.line_height)
        
        x_start = max(0, center_x - self.line_thickness // 2)
        x_end = min(sim.width, center_x + self.line_thickness // 2 + 1)
        
        sim.material_types[water_top:water_bottom, x_start:x_end] = MaterialType.WATER
        
        # Enable physics
        sim.enable_heat_diffusion = False
        sim.enable_self_gravity = False
        sim.external_gravity = (0, 9.81)
        sim.enable_surface_tension = True
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check collapse behavior."""
        water_count = np.sum(sim.material_types == MaterialType.WATER)
        initial_count = self.initial_state.get('initial_water', 0)
        
        if water_count == 0:
            return {
                'success': False,
                'metrics': {'water_count': 0},
                'message': 'No water remaining!'
            }
            
        # Measure water spread
        water_coords = np.argwhere(sim.material_types == MaterialType.WATER)
        min_x = np.min(water_coords[:, 1])
        max_x = np.max(water_coords[:, 1])
        water_width = max_x - min_x + 1
        
        # Measure height
        min_y = np.min(water_coords[:, 0])
        max_y = np.max(water_coords[:, 0])
        water_height = max_y - min_y + 1
        
        # Initial dimensions
        initial_width = self.initial_state.get('initial_width', self.line_thickness)
        initial_height = self.initial_state.get('initial_height', self.line_height)
        
        # Check if collapsed (wider and shorter)
        has_spread = water_width > initial_width * 2
        has_shortened = water_height < initial_height * 0.5
        
        # Conservation check
        water_conserved = water_count >= initial_count * 0.95
        
        success = has_spread and has_shortened and water_conserved
        
        return {
            'success': success,
            'metrics': {
                'water_count': water_count,
                'water_width': water_width,
                'water_height': water_height,
                'width_ratio': water_width / initial_width,
                'height_ratio': water_height / initial_height,
            },
            'message': f"Water: {water_width}x{water_height} (was {initial_width}x{initial_height})"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial dimensions."""
        super().store_initial_state(sim)
        self.initial_state['initial_water'] = np.sum(sim.material_types == MaterialType.WATER)
        
        water_coords = np.argwhere(sim.material_types == MaterialType.WATER)
        if len(water_coords) > 0:
            self.initial_state['initial_width'] = np.max(water_coords[:, 1]) - np.min(water_coords[:, 1]) + 1
            self.initial_state['initial_height'] = np.max(water_coords[:, 0]) - np.min(water_coords[:, 0]) + 1
        else:
            self.initial_state['initial_width'] = 0
            self.initial_state['initial_height'] = 0


class FluidGravityScenario(TestScenario):
    """Test fluid behavior near gravitational body."""
    
    def __init__(self, planet_radius: int = 15, fluid_amount: int = 200,
                 fluid_type: MaterialType = MaterialType.WATER,
                 grid_size: int = 80, **kwargs):
        """Initialize fluid gravity test."""
        super().__init__(**kwargs)
        self.planet_radius = planet_radius
        self.fluid_amount = fluid_amount
        self.fluid_type = fluid_type
        self.grid_size = grid_size
        
    def get_name(self) -> str:
        return f"fluid_gravity_{self.fluid_type.name.lower()}"
        
    def get_description(self) -> str:
        return f"Tests {self.fluid_type.name} behavior near gravitational body"
        
    def setup(self, sim: GeoGame) -> None:
        """Create planet with fluid."""
        # Clear to space
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 3.0
        
        # Create spherical planet
        center_x, center_y = sim.width // 2, sim.height // 2
        yy, xx = np.ogrid[:sim.height, :sim.width]
        dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        
        planet_mask = dist <= self.planet_radius
        sim.material_types[planet_mask] = MaterialType.BASALT
        sim.temperature[planet_mask] = 300.0
        
        # Add atmosphere layer
        atmo_mask = (dist > self.planet_radius) & (dist <= self.planet_radius + 3)
        sim.material_types[atmo_mask] = MaterialType.AIR
        sim.temperature[atmo_mask] = 280.0
        
        # Add fluid blob away from planet
        fluid_x = center_x + self.planet_radius + 10
        fluid_y = center_y
        
        # Create fluid blob
        fluid_placed = 0
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                y = fluid_y + dy
                x = fluid_x + dx
                if (0 <= y < sim.height and 0 <= x < sim.width and 
                    sim.material_types[y, x] == MaterialType.SPACE):
                    sim.material_types[y, x] = self.fluid_type
                    sim.temperature[y, x] = 290.0 if self.fluid_type == MaterialType.WATER else 1400.0
                    fluid_placed += 1
                    if fluid_placed >= self.fluid_amount:
                        break
            if fluid_placed >= self.fluid_amount:
                break
                
        # Enable self-gravity to pull fluid toward planet
        sim.enable_self_gravity = True
        sim.external_gravity = (0, 0)
        sim.enable_heat_diffusion = False
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check if fluid falls toward planet."""
        fluid_count = np.sum(sim.material_types == self.fluid_type)
        initial_count = self.initial_state.get('initial_fluid', 0)
        
        if fluid_count == 0:
            return {
                'success': False,
                'metrics': {'fluid_count': 0},
                'message': 'No fluid remaining!'
            }
            
        # Calculate fluid center of mass
        fluid_coords = np.argwhere(sim.material_types == self.fluid_type)
        fluid_center_y = np.mean(fluid_coords[:, 0])
        fluid_center_x = np.mean(fluid_coords[:, 1])
        
        # Planet center
        center_x, center_y = sim.width // 2, sim.height // 2
        
        # Distance from planet center
        distance_to_planet = np.sqrt((fluid_center_x - center_x)**2 + 
                                   (fluid_center_y - center_y)**2)
        
        # Initial distance
        initial_distance = self.initial_state.get('initial_distance', 0)
        
        # Check if fluid moved toward planet
        moved_closer = distance_to_planet < initial_distance - 2
        
        # Check if fluid is on planet surface
        on_surface = distance_to_planet <= self.planet_radius + 5
        
        # Conservation check
        fluid_conserved = fluid_count >= initial_count * 0.90
        
        success = moved_closer and fluid_conserved
        
        return {
            'success': success,
            'metrics': {
                'fluid_count': fluid_count,
                'distance_to_planet': distance_to_planet,
                'initial_distance': initial_distance,
                'moved_closer': moved_closer,
                'on_surface': on_surface,
            },
            'message': f"Fluid distance: {distance_to_planet:.1f} (was {initial_distance:.1f})"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial fluid position."""
        super().store_initial_state(sim)
        self.initial_state['initial_fluid'] = np.sum(sim.material_types == self.fluid_type)
        
        # Calculate initial distance
        fluid_coords = np.argwhere(sim.material_types == self.fluid_type)
        if len(fluid_coords) > 0:
            fluid_center_y = np.mean(fluid_coords[:, 0])
            fluid_center_x = np.mean(fluid_coords[:, 1])
            center_x, center_y = sim.width // 2, sim.height // 2
            self.initial_state['initial_distance'] = np.sqrt(
                (fluid_center_x - center_x)**2 + (fluid_center_y - center_y)**2
            )
        else:
            self.initial_state['initial_distance'] = 0
            
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [self.fluid_type, MaterialType.BASALT],
            'preferred_display_mode': 'material',
            'show_metrics': ['distance_to_planet', 'fluid_count'],
        }