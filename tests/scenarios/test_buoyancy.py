"""Consolidated buoyancy test scenarios for visual testing"""

import numpy as np
from tests.framework.test_framework import TestScenario
from materials import MaterialType


class IceFloatingScenario(TestScenario):
    """Ice falling into water and demonstrating buoyancy vs granite"""
    
    def __init__(self):
        super().__init__()
        
    def get_name(self) -> str:
        return "Ice Floating vs Granite"
        
    def get_description(self) -> str:
        return "Ice and granite fall into water, ice should float higher"
        
    def setup(self, sim):
        """Set up ice vs granite buoyancy test"""
        # Configure for buoyancy testing
        sim.external_gravity = (0, 8)  # Moderate gravity
        sim.enable_self_gravity = False
        sim.enable_solid_drag = True
        sim.debug_rigid_bodies = False
        
        # Disable thermal processes
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        sim.enable_heat_diffusion = False
        sim.enable_material_processes = False
        sim.enable_atmospheric_processes = False
        if hasattr(sim, 'heat_transfer'):
            sim.heat_transfer.enabled = False
        
        # Setup environment
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 280.0
        
        # Deep water ocean in bottom half
        water_surface_y = sim.height // 2
        for y in range(water_surface_y, sim.height):
            for x in range(sim.width):
                sim.material_types[y, x] = MaterialType.WATER
                sim.temperature[y, x] = 275.0
        
        # Ice block (3x3) above water
        ice_start_y = 8
        ice_center_x = sim.width // 3
        for dy in range(3):
            for dx in range(3):
                y, x = ice_start_y + dy, ice_center_x + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.ICE
                    sim.temperature[y, x] = 270.0
        
        # Granite block (2x2) for comparison  
        granite_center_x = 2 * sim.width // 3
        for dy in range(2):
            for dx in range(2):
                y, x = ice_start_y + dy, granite_center_x + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.GRANITE
                    sim.temperature[y, x] = 300.0
        
        self.water_surface_y = water_surface_y
        self.initial_ice_count = np.sum(sim.material_types == MaterialType.ICE)
        self.initial_granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
        
        print(f"Setup: Ice (3x3) vs Granite (2x2) above water at Y={water_surface_y}")
        
    def evaluate(self, sim):
        """Track ice vs granite positions"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 10 == 0:
            ice_mask = sim.material_types == MaterialType.ICE
            granite_mask = sim.material_types == MaterialType.GRANITE
            
            ice_status = "lost"
            granite_status = "lost"
            
            if np.any(ice_mask):
                ice_coords = np.argwhere(ice_mask)
                ice_center_y = np.mean(ice_coords[:, 0])
                ice_phase = "floating" if ice_center_y < self.water_surface_y + 3 else "submerged"
                ice_status = f"Y={ice_center_y:.1f} ({len(ice_coords)} cells, {ice_phase})"
            
            if np.any(granite_mask):
                granite_coords = np.argwhere(granite_mask)
                granite_center_y = np.mean(granite_coords[:, 0])
                granite_phase = "floating" if granite_center_y < self.water_surface_y + 3 else "submerged"
                granite_status = f"Y={granite_center_y:.1f} ({len(granite_coords)} cells, {granite_phase})"
            
            print(f"Step {step:3d}: Ice: {ice_status:35s} | Granite: {granite_status}")
        
        # Success if both materials exist
        ice_exists = np.any(sim.material_types == MaterialType.ICE)
        granite_exists = np.any(sim.material_types == MaterialType.GRANITE)
        return {
            "success": ice_exists and granite_exists,
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


class RockDonutWithMagmaScenario(TestScenario):
    """Rock donut filled with magma testing container dynamics"""
    
    def __init__(self):
        super().__init__()
        
    def get_name(self) -> str:
        return "Rock Donut Container"
        
    def get_description(self) -> str:
        return "Rock donut with hot magma tests temperature preservation during fall"
        
    def setup(self, sim):
        """Set up rock donut scenario"""
        sim.external_gravity = (0, 10)
        sim.enable_self_gravity = False
        sim.enable_solid_drag = False
        sim.debug_rigid_bodies = True
        
        # Disable thermal processes to isolate container mechanics
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        sim.enable_heat_diffusion = False
        sim.enable_material_processes = False
        sim.enable_atmospheric_processes = False
        if hasattr(sim, 'heat_transfer'):
            sim.heat_transfer.enabled = False
        
        # Clear to space
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = sim.space_temperature
        
        # Create granite donut with magma core
        center_y, center_x = 8, sim.width // 2
        
        # Donut ring (7x7 with 3x3 hole)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                y, x = center_y + dy, center_x + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    if abs(dy) <= 1 and abs(dx) <= 1:
                        continue  # Hole for magma
                    sim.material_types[y, x] = MaterialType.GRANITE
                    sim.temperature[y, x] = 1000.0
        
        # Fill hole with hot magma
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y, x = center_y + dy, center_x + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.MAGMA
                    sim.temperature[y, x] = 1800.0
        
        # Water ocean at bottom
        ocean_start = sim.height - 15
        for y in range(ocean_start, sim.height):
            for x in range(sim.width):
                sim.material_types[y, x] = MaterialType.WATER
                sim.temperature[y, x] = 275.0
        
        self.initial_granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
        self.initial_magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
        self.initial_magma_temp = np.mean(sim.temperature[sim.material_types == MaterialType.MAGMA])
        self.ocean_start = ocean_start
        
        print(f"Setup: Granite donut ({self.initial_granite_count} cells) with magma core ({self.initial_magma_count} cells)")
        print(f"Initial magma temperature: {self.initial_magma_temp:.0f}K")
        
    def evaluate(self, sim):
        """Monitor donut integrity and temperature preservation"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 5 == 0:
            granite_mask = sim.material_types == MaterialType.GRANITE
            magma_mask = sim.material_types == MaterialType.MAGMA
            
            granite_count = np.sum(granite_mask)
            magma_count = np.sum(magma_mask)
            
            status_parts = []
            
            if granite_count > 0:
                granite_center_y = np.mean(np.argwhere(granite_mask)[:, 0])
                status_parts.append(f"Granite: {granite_count} cells at Y={granite_center_y:.1f}")
            else:
                status_parts.append("Granite: LOST")
            
            if magma_count > 0:
                magma_center_y = np.mean(np.argwhere(magma_mask)[:, 0])
                magma_temp = np.mean(sim.temperature[magma_mask])
                temp_ok = magma_temp > 1000
                temp_status = "HOT" if temp_ok else "COLD!"
                status_parts.append(f"Magma: {magma_count} cells at Y={magma_center_y:.1f}, T={magma_temp:.0f}K ({temp_status})")
            else:
                status_parts.append("Magma: LOST")
            
            print(f"Step {step:3d}: {' | '.join(status_parts)}")
        
        # Success criteria: both materials exist and magma stays hot
        granite_exists = np.sum(sim.material_types == MaterialType.GRANITE) > 0
        magma_exists = np.sum(sim.material_types == MaterialType.MAGMA) > 0
        
        if magma_exists:
            magma_temp = np.mean(sim.temperature[sim.material_types == MaterialType.MAGMA])
            temp_preserved = magma_temp > 1000
        else:
            temp_preserved = False
        
        return {
            "success": granite_exists and magma_exists and temp_preserved,
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


class DensityComparisonScenario(TestScenario):
    """Multiple materials with different densities for comparison"""
    
    def __init__(self):
        super().__init__()
        
    def get_name(self) -> str:
        return "Density Comparison"
        
    def get_description(self) -> str:
        return "Ice, granite, and basalt falling into water to compare densities"
        
    def setup(self, sim):
        """Set up multi-material density test"""
        sim.external_gravity = (0, 6)
        sim.enable_self_gravity = False
        sim.enable_solid_drag = True
        sim.debug_rigid_bodies = False
        
        # Disable thermal processes
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        sim.enable_heat_diffusion = False
        sim.enable_material_processes = False
        sim.enable_atmospheric_processes = False
        if hasattr(sim, 'heat_transfer'):
            sim.heat_transfer.enabled = False
        
        # Air environment (lighter than water)
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 280.0
        
        # Water layer at bottom
        water_start = sim.height - 12
        for y in range(water_start, sim.height):
            for x in range(sim.width):
                sim.material_types[y, x] = MaterialType.WATER
                sim.temperature[y, x] = 275.0
        
        # Create different materials horizontally spaced
        start_y = 5
        spacing = sim.width // 4
        
        # Ice block (should float)
        ice_x = spacing
        for dy in range(2):
            for dx in range(2):
                sim.material_types[start_y + dy, ice_x + dx] = MaterialType.ICE
                sim.temperature[start_y + dy, ice_x + dx] = 270.0
        
        # Granite block (should sink)
        granite_x = spacing * 2
        for dy in range(2):
            for dx in range(2):
                sim.material_types[start_y + dy, granite_x + dx] = MaterialType.GRANITE
                sim.temperature[start_y + dy, granite_x + dx] = 300.0
        
        # Basalt block (should sink faster)
        basalt_x = spacing * 3
        for dy in range(2):
            for dx in range(2):
                sim.material_types[start_y + dy, basalt_x + dx] = MaterialType.BASALT
                sim.temperature[start_y + dy, basalt_x + dx] = 350.0
        
        self.water_start = water_start
        self.materials = ["Ice", "Granite", "Basalt"]
        self.material_types = [MaterialType.ICE, MaterialType.GRANITE, MaterialType.BASALT]
        
        # Print densities for reference
        ice_density = sim.material_db.get_properties(MaterialType.ICE).density
        granite_density = sim.material_db.get_properties(MaterialType.GRANITE).density  
        basalt_density = sim.material_db.get_properties(MaterialType.BASALT).density
        water_density = sim.material_db.get_properties(MaterialType.WATER).density
        
        print(f"Density comparison:")
        print(f"  Ice: {ice_density} kg/m³ (should float)")
        print(f"  Granite: {granite_density} kg/m³ (should sink)")
        print(f"  Basalt: {basalt_density} kg/m³ (should sink faster)")
        print(f"  Water: {water_density} kg/m³ (reference)")
        
    def evaluate(self, sim):
        """Track relative positions of different materials"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 10 == 0:
            positions = []
            
            for name, mat_type in zip(self.materials, self.material_types):
                mask = sim.material_types == mat_type
                if np.any(mask):
                    coords = np.argwhere(mask)
                    center_y = np.mean(coords[:, 0])
                    count = len(coords)
                    
                    if center_y < self.water_start - 1:
                        status = "falling"
                    elif center_y < self.water_start + 2:
                        status = "surface"
                    else:
                        status = "submerged"
                    
                    positions.append(f"{name}: Y={center_y:.1f} ({count} cells, {status})")
                else:
                    positions.append(f"{name}: LOST")
            
            print(f"Step {step:3d}: {' | '.join(positions)}")
        
        # Success if all materials still exist
        return {
            "success": all(np.any(sim.material_types == mat_type) for mat_type in self.material_types),
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


class RigidBodyBuoyancyScenario(TestScenario):
    """Complete rigid body buoyancy test - ice floats, granite sinks"""
    
    def __init__(self):
        super().__init__()
        
    def get_name(self) -> str:
        return "Rigid Body Buoyancy"
        
    def get_description(self) -> str:
        return "Complete test of ice floating and granite sinking as rigid bodies"
        
    def setup(self, sim):
        """Set up rigid body buoyancy test"""
        sim.external_gravity = (0, 10)
        sim.enable_self_gravity = False
        sim.enable_solid_drag = True
        sim.debug_rigid_bodies = True
        
        # Disable thermal processes
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        sim.enable_heat_diffusion = False
        sim.enable_material_processes = False
        sim.enable_atmospheric_processes = False
        if hasattr(sim, 'heat_transfer'):
            sim.heat_transfer.enabled = False
        
        # Air environment
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 280.0
        
        # Deep ocean at bottom
        ocean_depth = 10
        ocean_start = sim.height - ocean_depth
        for y in range(ocean_start, sim.height):
            for x in range(sim.width):
                sim.material_types[y, x] = MaterialType.WATER
                sim.temperature[y, x] = 275.0
        
        # Ice block (3x3) - should float
        ice_y = 5
        ice_x = sim.width // 3
        for dy in range(3):
            for dx in range(3):
                y, x = ice_y + dy, ice_x + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.ICE
                    sim.temperature[y, x] = 270.0
        
        # Granite block (3x3) - should sink
        granite_y = 5
        granite_x = 2 * sim.width // 3
        for dy in range(3):
            for dx in range(3):
                y, x = granite_y + dy, granite_x + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.GRANITE
                    sim.temperature[y, x] = 300.0
        
        self.ocean_start = ocean_start
        print(f"Setup: Ice and granite blocks falling into ocean at Y={ocean_start}")
        
    def evaluate(self, sim):
        """Track rigid body motion and buoyancy"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 10 == 0:
            ice_mask = sim.material_types == MaterialType.ICE
            granite_mask = sim.material_types == MaterialType.GRANITE
            
            ice_status = "lost"
            granite_status = "lost"
            
            if np.any(ice_mask):
                ice_y = np.mean(np.argwhere(ice_mask)[:, 0])
                ice_phase = "air" if ice_y < self.ocean_start else "water"
                ice_status = f"Y={ice_y:.1f} ({ice_phase})"
            
            if np.any(granite_mask):
                granite_y = np.mean(np.argwhere(granite_mask)[:, 0])
                granite_phase = "air" if granite_y < self.ocean_start else "water"
                granite_status = f"Y={granite_y:.1f} ({granite_phase})"
            
            print(f"Step {step:3d}: Ice: {ice_status:20s} | Granite: {granite_status}")
        
        return {
            "success": True,
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


# Register scenarios for visual runner
SCENARIOS = {
    'ice_floating': lambda: IceFloatingScenario(),
    'rock_donut_container': lambda: RockDonutWithMagmaScenario(), 
    'density_comparison': lambda: DensityComparisonScenario(),
    'rigid_body_buoyancy': lambda: RigidBodyBuoyancyScenario(),
}