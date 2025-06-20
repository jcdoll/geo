"""Consolidated fluid dynamics test scenarios"""

import numpy as np
from tests.framework.test_framework import TestScenario
from materials import MaterialType


class WaterConservationScenario(TestScenario):
    """Test water conservation during fluid flow"""
    
    def __init__(self):
        super().__init__()
        self.name = "Water Conservation"
        self.description = "Water droplet maintaining volume during flow"
        
    def setup(self, sim):
        """Set up water conservation test"""
        sim.external_gravity = (0, 5)
        sim.enable_self_gravity = False
        sim.enable_solid_drag = True
        sim.debug_rigid_bodies = False
        
        # Enable thermal and atmospheric processes for realistic fluid behavior
        sim.enable_heat_diffusion = True
        sim.enable_atmospheric_processes = True
        
        # Air environment
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 280.0
        
        # Create water droplet in upper portion
        center_x, center_y = sim.width // 2, sim.height // 4
        radius = 3
        
        for y in range(sim.height):
            for x in range(sim.width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= radius:
                    sim.material_types[y, x] = MaterialType.WATER
                    sim.temperature[y, x] = 275.0
        
        self.initial_water_count = np.sum(sim.material_types == MaterialType.WATER)
        print(f"Setup: Water droplet with {self.initial_water_count} cells")
        
    def evaluate(self, sim):
        """Monitor water conservation"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 10 == 0:
            current_water_count = np.sum(sim.material_types == MaterialType.WATER)
            water_coords = np.argwhere(sim.material_types == MaterialType.WATER)
            
            if len(water_coords) > 0:
                center_y = np.mean(water_coords[:, 0])
                center_x = np.mean(water_coords[:, 1])
                print(f"Step {step:3d}: {current_water_count} water cells at ({center_x:.1f}, {center_y:.1f})")
            else:
                print(f"Step {step:3d}: No water remaining!")
                
        return {
            "success": np.sum(sim.material_types == MaterialType.WATER) > 0,
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


class MagmaFlowScenario(TestScenario):
    """Test magma flow and cooling behavior"""
    
    def __init__(self):
        super().__init__()
        self.name = "Magma Flow"
        self.description = "Hot magma flowing and cooling in air"
        
    def setup(self, sim):
        """Set up magma flow test"""
        sim.external_gravity = (0, 8)
        sim.enable_self_gravity = False
        sim.enable_solid_drag = False
        
        # Enable thermal processes for cooling
        sim.enable_heat_diffusion = True
        sim.enable_radiative_cooling = True
        sim.enable_material_processes = True  # Allow solidification
        
        # Cool air environment
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 280.0
        
        # Hot magma blob at top
        blob_x, blob_y = sim.width // 2, 5
        blob_size = 4
        
        for dy in range(blob_size):
            for dx in range(blob_size):
                y, x = blob_y + dy, blob_x - blob_size//2 + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.MAGMA
                    sim.temperature[y, x] = 1800.0  # Very hot
        
        self.initial_magma_temp = 1800.0
        self.initial_magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
        print(f"Setup: Magma blob ({self.initial_magma_count} cells) at {self.initial_magma_temp}K")
        
    def evaluate(self, sim):
        """Monitor magma flow and cooling"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 10 == 0:
            magma_mask = sim.material_types == MaterialType.MAGMA
            magma_count = np.sum(magma_mask)
            
            if magma_count > 0:
                magma_coords = np.argwhere(magma_mask)
                center_y = np.mean(magma_coords[:, 0])
                avg_temp = np.mean(sim.temperature[magma_mask])
                
                # Check for solidification
                solid_rocks = np.sum((sim.material_types == MaterialType.BASALT) | 
                                   (sim.material_types == MaterialType.GRANITE))
                
                print(f"Step {step:3d}: {magma_count} magma cells at Y={center_y:.1f}, T={avg_temp:.0f}K, {solid_rocks} solidified")
            else:
                print(f"Step {step:3d}: All magma solidified or lost")
                
        return {
            "success": True,  # Always continue
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


class SurfaceTensionScenario(TestScenario):
    """Test surface tension effects on droplet formation"""
    
    def __init__(self):
        super().__init__()
        self.name = "Surface Tension"
        self.description = "Water droplet formation and surface tension effects"
        
    def setup(self, sim):
        """Set up surface tension test"""
        sim.external_gravity = (0, 3)  # Light gravity
        sim.enable_self_gravity = False
        sim.enable_solid_drag = True
        sim.enable_surface_tension = True  # Key feature
        
        # Minimal other processes to isolate surface tension
        sim.enable_heat_diffusion = False
        sim.enable_material_processes = False
        
        # Space environment (very low density)
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 280.0
        
        # Create scattered water cells that should coalesce
        water_positions = [
            (10, 15), (10, 16), (10, 17),  # Horizontal line
            (11, 15), (12, 15),            # Vertical extension
            (9, 18), (8, 19)               # Separate droplets
        ]
        
        for y, x in water_positions:
            if 0 <= y < sim.height and 0 <= x < sim.width:
                sim.material_types[y, x] = MaterialType.WATER
                sim.temperature[y, x] = 275.0
        
        self.initial_water_count = np.sum(sim.material_types == MaterialType.WATER)
        print(f"Setup: {self.initial_water_count} scattered water cells - should coalesce due to surface tension")
        
    def evaluate(self, sim):
        """Monitor droplet formation"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 10 == 0:
            water_mask = sim.material_types == MaterialType.WATER
            water_count = np.sum(water_mask)
            
            if water_count > 0:
                water_coords = np.argwhere(water_mask)
                center_y = np.mean(water_coords[:, 0])
                center_x = np.mean(water_coords[:, 1])
                
                # Calculate compactness (measure of coalescence)
                distances = np.sqrt(np.sum((water_coords - [center_y, center_x])**2, axis=1))
                avg_distance = np.mean(distances)
                
                print(f"Step {step:3d}: {water_count} water cells, center=({center_x:.1f},{center_y:.1f}), compactness={avg_distance:.2f}")
            else:
                print(f"Step {step:3d}: No water remaining")
                
        return {
            "success": np.sum(sim.material_types == MaterialType.WATER) > 0,
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


# Register scenarios for visual runner
SCENARIOS = {
    'water_conservation': lambda: WaterConservationScenario(),
    'magma_flow': lambda: MagmaFlowScenario(),
    'surface_tension': lambda: SurfaceTensionScenario(),
}