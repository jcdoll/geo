"""Consolidated rigid body dynamics test scenarios"""

import numpy as np
from tests.framework.test_framework import TestScenario
from materials import MaterialType


class FallingRockScenario(TestScenario):
    """Rock blocks falling as coherent rigid bodies"""
    
    def __init__(self):
        super().__init__()
        self.name = "Falling Rock"
        self.description = "Granite blocks falling and maintaining cohesion as rigid bodies"
        
    def setup(self, sim):
        """Set up falling rock test"""
        sim.external_gravity = (0, 10)
        sim.enable_self_gravity = False
        sim.enable_solid_drag = True
        sim.debug_rigid_bodies = True  # Show rigid body info
        
        # Disable thermal processes to focus on mechanics
        sim.enable_heat_diffusion = False
        sim.enable_material_processes = False
        sim.enable_atmospheric_processes = False
        
        # Air environment
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 280.0
        
        # Create two separate granite blocks
        # Block 1: 3x3 block
        block1_x, block1_y = 10, 5
        for dy in range(3):
            for dx in range(3):
                y, x = block1_y + dy, block1_x + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.GRANITE
                    sim.temperature[y, x] = 300.0
        
        # Block 2: 2x4 block (different shape)
        block2_x, block2_y = 25, 8
        for dy in range(2):
            for dx in range(4):
                y, x = block2_y + dy, block2_x + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.GRANITE
                    sim.temperature[y, x] = 300.0
        
        # Solid ground at bottom
        ground_y = sim.height - 5
        for y in range(ground_y, sim.height):
            for x in range(sim.width):
                sim.material_types[y, x] = MaterialType.BASALT
                sim.temperature[y, x] = 300.0
        
        self.initial_granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
        self.ground_y = ground_y
        print(f"Setup: Two granite blocks ({self.initial_granite_count} total cells) falling onto basalt ground")
        
    def evaluate(self, sim):
        """Monitor rigid body falling"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 5 == 0:
            granite_mask = sim.material_types == MaterialType.GRANITE
            granite_count = np.sum(granite_mask)
            
            if granite_count > 0:
                granite_coords = np.argwhere(granite_mask)
                center_y = np.mean(granite_coords[:, 0])
                
                # Check if reached ground
                max_y = np.max(granite_coords[:, 0])
                distance_to_ground = self.ground_y - max_y
                
                if distance_to_ground <= 1:
                    status = "landed"
                elif distance_to_ground < 5:
                    status = "approaching ground"
                else:
                    status = "falling"
                
                print(f"Step {step:3d}: {granite_count} granite cells at Y={center_y:.1f} ({status})")
            else:
                print(f"Step {step:3d}: No granite remaining")
                
        return {
            "success": np.sum(sim.material_types == MaterialType.GRANITE) > 0,
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


class ContainerFallScenario(TestScenario):
    """Container with fluid falling and preserving contents"""
    
    def __init__(self):
        super().__init__()
        self.name = "Container Fall"
        self.description = "Rock container with water falling as a unit"
        
    def setup(self, sim):
        """Set up container fall test"""
        sim.external_gravity = (0, 8)
        sim.enable_self_gravity = False
        sim.enable_solid_drag = False
        sim.debug_rigid_bodies = True
        
        # Disable thermal processes
        sim.enable_heat_diffusion = False
        sim.enable_material_processes = False
        sim.enable_atmospheric_processes = False
        
        # Air environment
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 280.0
        
        # Create container (U-shaped)
        container_x, container_y = sim.width // 2, 8
        
        # Container walls and bottom (5x5 with 3x3 cavity)
        for dy in range(5):
            for dx in range(5):
                y, x = container_y + dy, container_x - 2 + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    # Skip interior (1,1) to (3,3) for cavity
                    if 1 <= dy <= 3 and 1 <= dx <= 3:
                        continue
                    sim.material_types[y, x] = MaterialType.GRANITE
                    sim.temperature[y, x] = 300.0
        
        # Fill cavity with water
        for dy in range(1, 4):
            for dx in range(1, 4):
                y, x = container_y + dy, container_x - 2 + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.WATER
                    sim.temperature[y, x] = 275.0
        
        # Ground at bottom
        ground_y = sim.height - 8
        for y in range(ground_y, sim.height):
            for x in range(sim.width):
                sim.material_types[y, x] = MaterialType.BASALT
                sim.temperature[y, x] = 300.0
        
        self.initial_granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
        self.initial_water_count = np.sum(sim.material_types == MaterialType.WATER)
        self.ground_y = ground_y
        
        print(f"Setup: Container ({self.initial_granite_count} granite, {self.initial_water_count} water)")
        
    def evaluate(self, sim):
        """Monitor container integrity during fall"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 5 == 0:
            granite_mask = sim.material_types == MaterialType.GRANITE
            water_mask = sim.material_types == MaterialType.WATER
            
            granite_count = np.sum(granite_mask)
            water_count = np.sum(water_mask)
            
            status_parts = []
            
            if granite_count > 0:
                granite_center_y = np.mean(np.argwhere(granite_mask)[:, 0])
                status_parts.append(f"Container: {granite_count} cells at Y={granite_center_y:.1f}")
            else:
                status_parts.append("Container: LOST")
            
            if water_count > 0:
                water_center_y = np.mean(np.argwhere(water_mask)[:, 0])
                # Check if water is still contained
                water_loss = self.initial_water_count - water_count
                if water_loss == 0:
                    containment = "contained"
                elif water_loss < 3:
                    containment = "minor leak"
                else:
                    containment = "major leak"
                status_parts.append(f"Water: {water_count} cells at Y={water_center_y:.1f} ({containment})")
            else:
                status_parts.append("Water: LOST")
            
            print(f"Step {step:3d}: {' | '.join(status_parts)}")
        
        # Success if both container and water exist
        container_exists = np.sum(sim.material_types == MaterialType.GRANITE) > 0
        water_exists = np.sum(sim.material_types == MaterialType.WATER) > 0
        return {
            "success": container_exists and water_exists,
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


class DonutDisplacementScenario(TestScenario):
    """Donut-shaped object displacing fluid"""
    
    def __init__(self):
        super().__init__()
        self.name = "Donut Displacement"
        self.description = "Rock donut with water center falling into water ocean"
        
    def setup(self, sim):
        """Set up donut displacement test"""
        sim.external_gravity = (0, 12)
        sim.enable_self_gravity = False
        sim.enable_solid_drag = True
        sim.debug_rigid_bodies = True
        
        # Enable fluid dynamics for displacement
        sim.enable_heat_diffusion = False
        sim.enable_material_processes = False
        
        # Air above, water below
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 280.0
        
        # Water ocean in bottom third
        ocean_start = 2 * sim.height // 3
        for y in range(ocean_start, sim.height):
            for x in range(sim.width):
                sim.material_types[y, x] = MaterialType.WATER
                sim.temperature[y, x] = 275.0
        
        # Create donut shape high above water
        center_x, center_y = sim.width // 2, 15
        outer_radius = 6
        inner_radius = 3
        
        # Donut ring
        for y in range(sim.height):
            for x in range(sim.width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if inner_radius < dist <= outer_radius:
                    sim.material_types[y, x] = MaterialType.GRANITE
                    sim.temperature[y, x] = 300.0
        
        # Fill donut center with water
        for y in range(sim.height):
            for x in range(sim.width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= inner_radius:
                    sim.material_types[y, x] = MaterialType.WATER
                    sim.temperature[y, x] = 275.0
        
        self.ocean_start = ocean_start
        self.initial_granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
        self.initial_center_water = np.sum((sim.material_types == MaterialType.WATER) & 
                                          (np.arange(sim.height)[:, None] < ocean_start))
        
        print(f"Setup: Granite donut ({self.initial_granite_count} cells) with water center ({self.initial_center_water} cells)")
        print(f"Ocean starts at Y={ocean_start}")
        
    def evaluate(self, sim):
        """Monitor donut displacement in ocean"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 10 == 0:
            granite_mask = sim.material_types == MaterialType.GRANITE
            water_mask = sim.material_types == MaterialType.WATER
            
            granite_count = np.sum(granite_mask)
            total_water = np.sum(water_mask)
            
            status_parts = []
            
            if granite_count > 0:
                granite_coords = np.argwhere(granite_mask)
                granite_center_y = np.mean(granite_coords[:, 0])
                
                if granite_center_y < self.ocean_start - 3:
                    phase = "falling (air)"
                elif granite_center_y < self.ocean_start + 3:
                    phase = "entering ocean"
                else:
                    phase = "in ocean"
                
                status_parts.append(f"Donut: {granite_count} cells at Y={granite_center_y:.1f} ({phase})")
            else:
                status_parts.append("Donut: LOST")
            
            # Check water displacement
            center_water = np.sum(water_mask & (np.arange(sim.height)[:, None] < self.ocean_start))
            displaced_water = total_water - (self.initial_center_water + (sim.width * (sim.height - self.ocean_start)))
            
            status_parts.append(f"Water: {total_water} total, {center_water} in center, displaced={displaced_water}")
            
            print(f"Step {step:3d}: {' | '.join(status_parts)}")
        
        return {
            "success": np.sum(sim.material_types == MaterialType.GRANITE) > 0,
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


# Register scenarios for visual runner  
SCENARIOS = {
    'falling_rock': lambda: FallingRockScenario(),
    'container_fall': lambda: ContainerFallScenario(),
    'donut_displacement': lambda: DonutDisplacementScenario(),
}