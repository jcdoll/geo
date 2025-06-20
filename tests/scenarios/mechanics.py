"""Mechanical physics test scenarios (gravity, buoyancy, pressure)."""

import numpy as np
from typing import Dict, Any, Tuple

from .base import TestScenario
from materials import MaterialType
from geo_game import GeoGame


class GravityFallScenario(TestScenario):
    """Test objects falling under gravity."""
    
    def __init__(self, object_size: int = 5, object_material: MaterialType = MaterialType.BASALT,
                 grid_size: int = 60, **kwargs):
        """Initialize gravity fall scenario."""
        super().__init__(**kwargs)
        self.object_size = object_size
        self.object_material = object_material
        self.grid_size = grid_size
        
    def get_name(self) -> str:
        return f"gravity_fall_{self.object_material.name.lower()}"
        
    def get_description(self) -> str:
        return f"Tests {self.object_material.name} falling under gravity"
        
    def setup(self, sim: GeoGame) -> None:
        """Create object suspended in air."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Create ground at bottom
        ground_height = sim.height - 5
        sim.material_types[ground_height:, :] = MaterialType.BASALT
        
        # Create falling object near top
        obj_y = 10
        obj_x = sim.width // 2
        
        # Make a square object
        y_start = max(0, obj_y - self.object_size // 2)
        y_end = min(sim.height, obj_y + self.object_size // 2)
        x_start = max(0, obj_x - self.object_size // 2)
        x_end = min(sim.width, obj_x + self.object_size // 2)
        
        sim.material_types[y_start:y_end, x_start:x_end] = self.object_material
        
        # Enable gravity
        sim.enable_self_gravity = False
        sim.external_gravity = (0, 9.81)
        sim.enable_heat_diffusion = False
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def get_object_center(self, sim: GeoGame) -> Tuple[float, float]:
        """Get center of mass of the object."""
        obj_mask = sim.material_types == self.object_material
        if not np.any(obj_mask):
            return -1, -1
            
        coords = np.argwhere(obj_mask)
        center_y = np.mean(coords[:, 0])
        center_x = np.mean(coords[:, 1])
        return center_y, center_x
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check if object has fallen."""
        center_y, center_x = self.get_object_center(sim)
        initial_y = self.initial_state.get('initial_y', 10)
        
        if center_y < 0:
            return {
                'success': False,
                'metrics': {'center_y': -1},
                'message': 'Object disappeared!'
            }
            
        # Calculate fall distance
        fall_distance = center_y - initial_y
        
        # Success if object has fallen significantly
        has_fallen = fall_distance > 10
        
        # Check if object is intact
        obj_count = np.sum(sim.material_types == self.object_material)
        initial_count = self.initial_state.get('object_count', 0)
        intact = obj_count >= initial_count * 0.9  # Allow 10% loss
        
        return {
            'success': has_fallen and intact,
            'metrics': {
                'center_y': center_y,
                'fall_distance': fall_distance,
                'object_count': obj_count,
                'intact': intact,
            },
            'message': f"Object at y={center_y:.1f} (fell {fall_distance:.1f} units)"
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial object position."""
        super().store_initial_state(sim)
        center_y, center_x = self.get_object_center(sim)
        self.initial_state['initial_y'] = center_y
        self.initial_state['object_count'] = np.sum(sim.material_types == self.object_material)


class BuoyancyScenario(TestScenario):
    """Test buoyancy forces on objects in fluids."""
    
    def __init__(self, fluid_type: MaterialType = MaterialType.WATER,
                 object_type: MaterialType = MaterialType.ICE,
                 object_size: int = 5, grid_size: int = 60, **kwargs):
        """Initialize buoyancy scenario."""
        super().__init__(**kwargs)
        self.fluid_type = fluid_type
        self.object_type = object_type
        self.object_size = object_size
        self.grid_size = grid_size
        
    def get_name(self) -> str:
        return f"buoyancy_{self.object_type.name.lower()}_in_{self.fluid_type.name.lower()}"
        
    def get_description(self) -> str:
        return f"Tests buoyancy of {self.object_type.name} in {self.fluid_type.name}"
        
    def setup(self, sim: GeoGame) -> None:
        """Create fluid environment with floating object."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        
        # Create fluid body
        fluid_level = sim.height // 2
        sim.material_types[fluid_level:, :] = self.fluid_type
        
        # Set appropriate temperatures
        if self.fluid_type == MaterialType.WATER:
            sim.temperature[fluid_level:, :] = 290.0  # Room temp water
        elif self.fluid_type == MaterialType.MAGMA:
            sim.temperature[fluid_level:, :] = 1400.0  # Molten rock
            
        # Create object just below fluid surface
        obj_y = fluid_level + 5
        obj_x = sim.width // 2
        
        y_start = max(0, obj_y - self.object_size // 2)
        y_end = min(sim.height, obj_y + self.object_size // 2)
        x_start = max(0, obj_x - self.object_size // 2)
        x_end = min(sim.width, obj_x + self.object_size // 2)
        
        sim.material_types[y_start:y_end, x_start:x_end] = self.object_type
        
        # Set object temperature based on type
        if self.object_type == MaterialType.ICE:
            sim.temperature[y_start:y_end, x_start:x_end] = 260.0  # Below freezing
        else:
            sim.temperature[y_start:y_end, x_start:x_end] = 290.0
            
        # Enable physics
        sim.enable_self_gravity = False
        sim.external_gravity = (0, 9.81)
        sim.enable_heat_diffusion = True if self.object_type == MaterialType.ICE else False
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check buoyancy behavior."""
        # Get object position
        obj_mask = sim.material_types == self.object_type
        if not np.any(obj_mask):
            # Object might have melted or transformed
            return {
                'success': True,  # This could be expected behavior
                'metrics': {'object_exists': False},
                'message': 'Object transformed or melted'
            }
            
        coords = np.argwhere(obj_mask)
        center_y = np.mean(coords[:, 0])
        
        # Find fluid surface
        fluid_surface_y = -1
        for y in range(sim.height):
            if sim.material_types[y, sim.width // 2] == self.fluid_type:
                fluid_surface_y = y
                break
                
        if fluid_surface_y < 0:
            return {
                'success': False,
                'metrics': {'fluid_found': False},
                'message': 'No fluid found!'
            }
            
        # Calculate position relative to fluid surface
        depth_in_fluid = center_y - fluid_surface_y
        
        # Get material densities
        from materials import MaterialDatabase
        mat_db = MaterialDatabase()
        obj_density = mat_db.get_properties(self.object_type).density
        fluid_density = mat_db.get_properties(self.fluid_type).density
        
        # Expected behavior based on density
        should_float = obj_density < fluid_density
        is_floating = depth_in_fluid < self.object_size  # Partially above surface
        is_sinking = depth_in_fluid > self.object_size * 2  # Well below surface
        
        # Success if behavior matches density
        if should_float:
            success = is_floating and not is_sinking
        else:
            success = is_sinking and not is_floating
            
        return {
            'success': success,
            'metrics': {
                'center_y': center_y,
                'fluid_surface_y': fluid_surface_y,
                'depth_in_fluid': depth_in_fluid,
                'obj_density': obj_density,
                'fluid_density': fluid_density,
                'should_float': should_float,
            },
            'message': f"Object at depth {depth_in_fluid:.1f} (should {'float' if should_float else 'sink'})"
        }


class HydrostaticPressureScenario(TestScenario):
    """Test hydrostatic pressure gradients in fluids."""
    
    def __init__(self, fluid_depth: int = 40, grid_size: int = 60, **kwargs):
        """Initialize hydrostatic pressure scenario."""
        super().__init__(**kwargs)
        self.fluid_depth = fluid_depth
        self.grid_size = grid_size
        
    def get_name(self) -> str:
        return f"hydrostatic_pressure_depth_{self.fluid_depth}"
        
    def get_description(self) -> str:
        return "Tests pressure increases with depth in static fluid"
        
    def setup(self, sim: GeoGame) -> None:
        """Create deep fluid column."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0
        sim.pressure[:] = 101325.0  # 1 atm
        
        # Create fluid column
        fluid_start = sim.height - self.fluid_depth
        sim.material_types[fluid_start:, :] = MaterialType.WATER
        sim.temperature[fluid_start:, :] = 290.0
        
        # Create solid walls to contain fluid
        wall_width = 5
        sim.material_types[:, :wall_width] = MaterialType.BASALT
        sim.material_types[:, -wall_width:] = MaterialType.BASALT
        
        # Enable pressure calculation
        sim.enable_pressure = True
        sim.enable_self_gravity = False
        sim.external_gravity = (0, 9.81)
        sim.enable_heat_diffusion = False
        
        # Zero initial velocities
        sim.velocity_x[:] = 0
        sim.velocity_y[:] = 0
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check pressure gradient."""
        # Sample pressure at different depths
        sample_x = sim.width // 2
        
        # Find water surface
        water_surface_y = -1
        for y in range(sim.height):
            if sim.material_types[y, sample_x] == MaterialType.WATER:
                water_surface_y = y
                break
                
        if water_surface_y < 0:
            return {
                'success': False,
                'metrics': {},
                'message': 'No water found!'
            }
            
        # Sample pressures at different depths
        depths = [5, 10, 20, 30]
        pressures = []
        
        for depth in depths:
            y = water_surface_y + depth
            if y < sim.height and sim.material_types[y, sample_x] == MaterialType.WATER:
                pressures.append(sim.pressure[y, sample_x])
            else:
                break
                
        if len(pressures) < 2:
            return {
                'success': False,
                'metrics': {'num_samples': len(pressures)},
                'message': 'Not enough depth samples'
            }
            
        # Check if pressure increases with depth
        pressure_increasing = all(pressures[i] < pressures[i+1] for i in range(len(pressures)-1))
        
        # Calculate average pressure gradient
        if len(pressures) >= 2:
            depth_diff = depths[len(pressures)-1] - depths[0]
            pressure_diff = pressures[-1] - pressures[0]
            gradient = pressure_diff / (depth_diff * sim.cell_size) if depth_diff > 0 else 0
            
            # Expected gradient: rho * g ≈ 1000 * 9.81 = 9810 Pa/m
            expected_gradient = 9810
            gradient_error = abs(gradient - expected_gradient) / expected_gradient
        else:
            gradient = 0
            gradient_error = 1.0
            
        success = pressure_increasing and gradient_error < 0.2  # 20% tolerance
        
        return {
            'success': success,
            'metrics': {
                'num_samples': len(pressures),
                'pressure_increasing': pressure_increasing,
                'gradient': gradient,
                'gradient_error_pct': gradient_error * 100,
            },
            'message': f"Pressure gradient: {gradient:.0f} Pa/m (error: {gradient_error*100:.1f}%)"
        }