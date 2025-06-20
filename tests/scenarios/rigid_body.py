"""Rigid body physics test scenarios."""

import numpy as np
from typing import Dict, Any, Tuple

from .base import TestScenario
from materials import MaterialType
from geo_game import GeoGame


class RigidBodyWithEnclosedFluidScenario(TestScenario):
    """Test rigid body containing fluid (donut with water)."""
    
    def __init__(self, container_size: int = 15, wall_thickness: int = 2,
                 fluid_type: MaterialType = MaterialType.WATER,
                 container_material: MaterialType = MaterialType.GRANITE, **kwargs):
        """Initialize rigid body with enclosed fluid test."""
        super().__init__(**kwargs)
        self.container_size = container_size
        self.wall_thickness = wall_thickness
        self.fluid_type = fluid_type
        self.container_material = container_material
        
    def get_name(self) -> str:
        return f"rigid_container_{self.fluid_type.name.lower()}"
        
    def get_description(self) -> str:
        return f"Tests rigid {self.container_material.name} container with {self.fluid_type.name}"
        
    def setup(self, sim: GeoGame) -> None:
        """Create rigid container with fluid inside."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Create ground
        ground_level = sim.height * 3 // 4
        sim.material_types[ground_level:, :] = MaterialType.BASALT
        
        # Create container (donut shape) above ground
        center_x = sim.width // 2
        center_y = ground_level - self.container_size - 5
        
        # Create solid square first
        y_start = max(0, center_y - self.container_size // 2)
        y_end = min(sim.height, center_y + self.container_size // 2)
        x_start = max(0, center_x - self.container_size // 2)
        x_end = min(sim.width, center_x + self.container_size // 2)
        
        sim.material_types[y_start:y_end, x_start:x_end] = self.container_material
        
        # Hollow out center, leaving walls
        hollow_start_y = y_start + self.wall_thickness
        hollow_end_y = y_end - self.wall_thickness
        hollow_start_x = x_start + self.wall_thickness
        hollow_end_x = x_end - self.wall_thickness
        
        if hollow_start_y < hollow_end_y and hollow_start_x < hollow_end_x:
            # Fill hollow with fluid
            sim.material_types[hollow_start_y:hollow_end_y, hollow_start_x:hollow_end_x] = self.fluid_type
            
        # Set appropriate temperatures
        if self.fluid_type == MaterialType.WATER:
            sim.temperature[sim.material_types == self.fluid_type] = 290.0
        elif self.fluid_type == MaterialType.MAGMA:
            sim.temperature[sim.material_types == self.fluid_type] = 1400.0
            
        # Create rigid body for container
        container_mask = sim.material_types == self.container_material
        if np.any(container_mask):
            # Find connected component
            from scipy import ndimage
            labeled, _ = ndimage.label(container_mask)
            # Use the largest component as rigid body
            if labeled.max() > 0:
                sim.add_rigid_body(container_mask, is_static=False)
                
        # Enable physics
        sim.enable_self_gravity = False
        sim.external_gravity = (0, 9.81)
        sim.enable_heat_diffusion = False
        sim.debug_rigid_bodies = True
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check container integrity and fluid containment."""
        # Count materials
        container_count = np.sum(sim.material_types == self.container_material)
        fluid_count = np.sum(sim.material_types == self.fluid_type)
        
        initial_container = self.initial_state.get('initial_container', 0)
        initial_fluid = self.initial_state.get('initial_fluid', 0)
        
        # Check container integrity
        container_intact = container_count >= initial_container * 0.95  # 95% preserved
        
        # Check fluid containment
        fluid_contained = fluid_count >= initial_fluid * 0.90  # 90% contained
        
        # Find container position
        if container_count > 0:
            container_coords = np.argwhere(sim.material_types == self.container_material)
            container_center_y = np.mean(container_coords[:, 0])
            container_center_x = np.mean(container_coords[:, 1])
            
            # Check if fallen
            initial_y = self.initial_state.get('initial_container_y', 0)
            fall_distance = container_center_y - initial_y
        else:
            container_center_y = -1
            fall_distance = 0
            
        # Check if fluid is still inside container
        if fluid_count > 0 and container_count > 0:
            fluid_coords = np.argwhere(sim.material_types == self.fluid_type)
            fluid_center_y = np.mean(fluid_coords[:, 0])
            fluid_center_x = np.mean(fluid_coords[:, 1])
            
            # Simple check: fluid center should be close to container center
            separation = np.sqrt((fluid_center_y - container_center_y)**2 + 
                               (fluid_center_x - container_center_x)**2)
            fluid_inside = separation < self.container_size / 3
        else:
            fluid_inside = False
            
        # Success criteria
        success = container_intact and fluid_contained and fluid_inside
        
        return {
            'success': success,
            'metrics': {
                'container_count': container_count,
                'fluid_count': fluid_count,
                'container_intact': container_intact,
                'fluid_contained': fluid_contained,
                'fluid_inside': fluid_inside,
                'fall_distance': fall_distance,
            },
            'message': f"Container: {container_count}/{initial_container}, Fluid: {fluid_count}/{initial_fluid}, Fell: {fall_distance:.1f}"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial counts and positions."""
        super().store_initial_state(sim)
        self.initial_state['initial_container'] = np.sum(sim.material_types == self.container_material)
        self.initial_state['initial_fluid'] = np.sum(sim.material_types == self.fluid_type)
        
        # Store container position
        container_coords = np.argwhere(sim.material_types == self.container_material)
        if len(container_coords) > 0:
            self.initial_state['initial_container_y'] = np.mean(container_coords[:, 0])


class RigidBodyFluidDisplacementScenario(TestScenario):
    """Test rigid body displacing fluid (rock pushing water)."""
    
    def __init__(self, rock_size: int = 10, rock_material: MaterialType = MaterialType.GRANITE,
                 fluid_depth: int = 20, **kwargs):
        """Initialize fluid displacement test."""
        super().__init__(**kwargs)
        self.rock_size = rock_size
        self.rock_material = rock_material
        self.fluid_depth = fluid_depth
        
    def get_name(self) -> str:
        return f"rigid_displacement_{self.rock_material.name.lower()}_in_water"
        
    def get_description(self) -> str:
        return f"Tests {self.rock_material.name} rigid body displacing water"
        
    def setup(self, sim: GeoGame) -> None:
        """Create rock above water pool."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Create water pool
        water_level = sim.height - self.fluid_depth
        sim.material_types[water_level:, :] = MaterialType.WATER
        
        # Create solid bottom
        bottom_thickness = 5
        sim.material_types[-bottom_thickness:, :] = MaterialType.BASALT
        
        # Create rock above water
        rock_x = sim.width // 2
        rock_y = water_level - self.rock_size - 10
        
        # Make rock square
        y_start = max(0, rock_y - self.rock_size // 2)
        y_end = min(sim.height, rock_y + self.rock_size // 2)
        x_start = max(0, rock_x - self.rock_size // 2)
        x_end = min(sim.width, rock_x + self.rock_size // 2)
        
        rock_mask = np.zeros_like(sim.material_types, dtype=bool)
        rock_mask[y_start:y_end, x_start:x_end] = True
        sim.material_types[rock_mask] = self.rock_material
        
        # Create rigid body
        sim.add_rigid_body(rock_mask, is_static=False)
        
        # Enable physics
        sim.enable_self_gravity = False
        sim.external_gravity = (0, 9.81)
        sim.enable_heat_diffusion = False
        sim.debug_rigid_bodies = True
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check displacement behavior."""
        # Count materials
        rock_count = np.sum(sim.material_types == self.rock_material)
        water_count = np.sum(sim.material_types == MaterialType.WATER)
        
        initial_rock = self.initial_state.get('initial_rock', 0)
        initial_water = self.initial_state.get('initial_water', 0)
        initial_water_level = self.initial_state.get('initial_water_level', 0)
        
        # Find rock position
        if rock_count > 0:
            rock_coords = np.argwhere(sim.material_types == self.rock_material)
            rock_center_y = np.mean(rock_coords[:, 0])
            rock_bottom_y = np.max(rock_coords[:, 0])
        else:
            rock_center_y = -1
            rock_bottom_y = -1
            
        # Find current water level (highest water cell in center column)
        center_x = sim.width // 2
        water_column = sim.material_types[:, center_x] == MaterialType.WATER
        if np.any(water_column):
            current_water_level = np.min(np.where(water_column)[0])
        else:
            current_water_level = sim.height
            
        # Check if rock has entered water
        rock_in_water = rock_bottom_y > initial_water_level
        
        # Check water level rise (displacement)
        water_rise = initial_water_level - current_water_level
        
        # Expected displacement volume
        if rock_in_water and rock_count > 0:
            # Count rock cells below original water level
            submerged_rock = np.sum((sim.material_types == self.rock_material) & 
                                  (np.arange(sim.height)[:, None] >= initial_water_level))
            expected_rise = submerged_rock / sim.width  # Simple approximation
        else:
            expected_rise = 0
            submerged_rock = 0
            
        # Success criteria
        rock_intact = rock_count >= initial_rock * 0.95
        water_conserved = abs(water_count - initial_water) < initial_water * 0.05
        displacement_occurred = water_rise > 0 and rock_in_water
        
        success = rock_intact and water_conserved and displacement_occurred
        
        return {
            'success': success,
            'metrics': {
                'rock_count': rock_count,
                'water_count': water_count,
                'rock_in_water': rock_in_water,
                'water_rise': water_rise,
                'expected_rise': expected_rise,
                'submerged_rock': submerged_rock,
            },
            'message': f"Rock in water: {rock_in_water}, Water rise: {water_rise:.1f} cells"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial state."""
        super().store_initial_state(sim)
        self.initial_state['initial_rock'] = np.sum(sim.material_types == self.rock_material)
        self.initial_state['initial_water'] = np.sum(sim.material_types == MaterialType.WATER)
        
        # Find initial water level
        center_x = sim.width // 2
        water_column = sim.material_types[:, center_x] == MaterialType.WATER
        if np.any(water_column):
            self.initial_state['initial_water_level'] = np.min(np.where(water_column)[0])
        else:
            self.initial_state['initial_water_level'] = sim.height
            
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [self.rock_material, MaterialType.WATER],
            'preferred_display_mode': 'material',
            'show_metrics': ['rock_in_water', 'water_rise'],
        }


class RigidBodyRotationScenario(TestScenario):
    """Test rigid body rotation and angular momentum."""
    
    def __init__(self, shape: str = "L", size: int = 10,
                 material: MaterialType = MaterialType.GRANITE, **kwargs):
        """Initialize rotation test."""
        super().__init__(**kwargs)
        self.shape = shape
        self.size = size
        self.material = material
        
    def get_name(self) -> str:
        return f"rigid_rotation_{self.shape}_shape"
        
    def get_description(self) -> str:
        return f"Tests rotation of {self.shape}-shaped rigid body"
        
    def setup(self, sim: GeoGame) -> None:
        """Create asymmetric rigid body."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Create shape in center
        center_x = sim.width // 2
        center_y = sim.height // 2
        
        shape_mask = np.zeros_like(sim.material_types, dtype=bool)
        
        if self.shape == "L":
            # Create L-shape
            # Vertical part
            y_start = center_y - self.size
            y_end = center_y + self.size // 2
            x_start = center_x - self.size // 4
            x_end = center_x + self.size // 4
            shape_mask[y_start:y_end, x_start:x_end] = True
            
            # Horizontal part
            y_start = center_y + self.size // 4
            y_end = center_y + self.size // 2
            x_start = center_x - self.size // 4
            x_end = center_x + self.size
            shape_mask[y_start:y_end, x_start:x_end] = True
            
        elif self.shape == "T":
            # Create T-shape
            # Vertical part
            y_start = center_y - self.size // 2
            y_end = center_y + self.size
            x_start = center_x - self.size // 4
            x_end = center_x + self.size // 4
            shape_mask[y_start:y_end, x_start:x_end] = True
            
            # Horizontal part (top of T)
            y_start = center_y - self.size // 2
            y_end = center_y - self.size // 4
            x_start = center_x - self.size
            x_end = center_x + self.size
            shape_mask[y_start:y_end, x_start:x_end] = True
            
        # Set material
        sim.material_types[shape_mask] = self.material
        
        # Create rigid body with initial angular velocity
        if np.any(shape_mask):
            body_id = sim.add_rigid_body(shape_mask, is_static=False)
            # Apply initial angular velocity
            if hasattr(sim, 'rigid_bodies') and body_id in sim.rigid_bodies:
                sim.rigid_bodies[body_id].angular_velocity = 0.5  # rad/s
                
        # Enable physics
        sim.enable_self_gravity = False
        sim.external_gravity = (0, 0)  # No gravity - test pure rotation
        sim.enable_heat_diffusion = False
        sim.debug_rigid_bodies = True
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check rotation behavior."""
        # Count material
        shape_count = np.sum(sim.material_types == self.material)
        initial_count = self.initial_state.get('initial_count', 0)
        
        # Calculate orientation change
        if shape_count > 0:
            shape_coords = np.argwhere(sim.material_types == self.material)
            
            # Calculate principal moments to determine orientation
            center_y = np.mean(shape_coords[:, 0])
            center_x = np.mean(shape_coords[:, 1])
            
            # Calculate second moments
            dy = shape_coords[:, 0] - center_y
            dx = shape_coords[:, 1] - center_x
            
            Ixx = np.sum(dy * dy)
            Iyy = np.sum(dx * dx)
            Ixy = np.sum(dx * dy)
            
            # Calculate orientation angle from principal axes
            angle = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
            angle_deg = np.degrees(angle)
            
            # Compare to initial
            initial_angle = self.initial_state.get('initial_angle', 0)
            rotation = angle_deg - initial_angle
            
            # Normalize to [-180, 180]
            while rotation > 180:
                rotation -= 360
            while rotation < -180:
                rotation += 360
        else:
            rotation = 0
            
        # Check if shape is intact
        shape_intact = shape_count >= initial_count * 0.95
        
        # Success if rotated significantly
        has_rotated = abs(rotation) > 10  # At least 10 degrees
        
        success = shape_intact and has_rotated
        
        return {
            'success': success,
            'metrics': {
                'shape_count': shape_count,
                'rotation_deg': rotation,
                'shape_intact': shape_intact,
            },
            'message': f"Rotation: {rotation:.1f}°, Shape intact: {shape_intact}"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial orientation."""
        super().store_initial_state(sim)
        self.initial_state['initial_count'] = np.sum(sim.material_types == self.material)
        
        # Calculate initial orientation
        shape_coords = np.argwhere(sim.material_types == self.material)
        if len(shape_coords) > 0:
            center_y = np.mean(shape_coords[:, 0])
            center_x = np.mean(shape_coords[:, 1])
            
            dy = shape_coords[:, 0] - center_y
            dx = shape_coords[:, 1] - center_x
            
            Ixx = np.sum(dy * dy)
            Iyy = np.sum(dx * dx)
            Ixy = np.sum(dx * dy)
            
            angle = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
            self.initial_state['initial_angle'] = np.degrees(angle)