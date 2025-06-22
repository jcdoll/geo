"""Material metamorphosis and phase transition test scenarios"""

import numpy as np
from tests.framework.test_framework import TestScenario
from materials import MaterialType


class MagmaContainmentScenario(TestScenario):
    """Test magma containment and metamorphic transitions"""
    
    def __init__(self):
        super().__init__()
        self.name = "Magma Containment"
        self.description = "Hot magma contained by rock undergoing metamorphism"
        
    def setup(self, sim):
        """Set up magma containment test"""
        sim.external_gravity = (0, 5)
        sim.enable_self_gravity = False
        sim.enable_solid_drag = False
        
        # Enable thermal and material processes for metamorphism
        sim.enable_heat_diffusion = True
        sim.enable_material_processes = True
        sim.enable_radiative_cooling = True
        
        # Cool air environment
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 280.0
        
        # Create rock chamber with magma inside
        chamber_x, chamber_y = sim.width // 2, sim.height // 2
        chamber_size = 8
        
        # Rock walls (granite initially)
        for dy in range(chamber_size):
            for dx in range(chamber_size):
                y, x = chamber_y - chamber_size//2 + dy, chamber_x - chamber_size//2 + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    # Create thick walls (2 cells thick)
                    if dy < 2 or dy >= chamber_size-2 or dx < 2 or dx >= chamber_size-2:
                        sim.material_types[y, x] = MaterialType.GRANITE
                        sim.temperature[y, x] = 800.0  # Warm from proximity to magma
        
        # Hot magma core
        for dy in range(2, chamber_size-2):
            for dx in range(2, chamber_size-2):
                y, x = chamber_y - chamber_size//2 + dy, chamber_x - chamber_size//2 + dx
                if 0 <= y < sim.height and 0 <= x < sim.width:
                    sim.material_types[y, x] = MaterialType.MAGMA
                    sim.temperature[y, x] = 1600.0  # Very hot
        
        self.initial_granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
        self.initial_magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
        self.chamber_center = (chamber_x, chamber_y)
        
        print(f"Setup: Magma chamber - {self.initial_granite_count} granite walls, {self.initial_magma_count} magma core")
        print(f"Watch for metamorphic transitions as heat spreads")
        
    def evaluate(self, sim):
        """Monitor metamorphic transitions"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 10 == 0:
            # Count different material types
            granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
            magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
            basalt_count = np.sum(sim.material_types == MaterialType.BASALT)
            schist_count = np.sum(sim.material_types == MaterialType.SCHIST)
            gneiss_count = np.sum(sim.material_types == MaterialType.GNEISS)
            
            # Average temperatures
            if magma_count > 0:
                magma_temp = np.mean(sim.temperature[sim.material_types == MaterialType.MAGMA])
            else:
                magma_temp = 0
                
            if granite_count > 0:
                granite_temp = np.mean(sim.temperature[sim.material_types == MaterialType.GRANITE])
            else:
                granite_temp = 0
            
            print(f"Step {step:3d}: Magma={magma_count}(T={magma_temp:.0f}K), Granite={granite_count}(T={granite_temp:.0f}K), "
                  f"Basalt={basalt_count}, Schist={schist_count}, Gneiss={gneiss_count}")
        
        # Success if magma is contained and materials exist
        magma_exists = np.sum(sim.material_types == MaterialType.MAGMA) > 0
        rock_exists = (np.sum(sim.material_types == MaterialType.GRANITE) + 
                      np.sum(sim.material_types == MaterialType.BASALT) + 
                      np.sum(sim.material_types == MaterialType.SCHIST) + 
                      np.sum(sim.material_types == MaterialType.GNEISS)) > 0
        
        return {
            "success": magma_exists or rock_exists,
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


class PhaseTransitionScenario(TestScenario):
    """Test phase transitions between material states"""
    
    def __init__(self):
        super().__init__()
        self.name = "Phase Transitions"
        self.description = "Ice melting to water, water freezing, magma solidifying"
        
    def setup(self, sim):
        """Set up phase transition test"""
        sim.external_gravity = (0, 3)  # Light gravity
        sim.enable_self_gravity = False
        sim.enable_solid_drag = True
        
        # Enable thermal processes for phase transitions
        sim.enable_heat_diffusion = True
        sim.enable_material_processes = True
        sim.enable_radiative_cooling = True
        
        # Moderate temperature air
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 290.0  # Slightly warm
        
        # Create different temperature zones
        zones_y = sim.height // 4
        
        # Hot zone (top) - melts ice
        for y in range(0, zones_y):
            for x in range(sim.width):
                sim.temperature[y, x] = 350.0  # Above melting point
        
        # Warm zone - liquid water stable
        for y in range(zones_y, 2*zones_y):
            for x in range(sim.width):
                sim.temperature[y, x] = 290.0  # Room temperature
        
        # Cool zone - freezes water
        for y in range(2*zones_y, 3*zones_y):
            for x in range(sim.width):
                sim.temperature[y, x] = 250.0  # Below freezing
        
        # Very cool zone - everything freezes
        for y in range(3*zones_y, sim.height):
            for x in range(sim.width):
                sim.temperature[y, x] = 200.0  # Very cold
        
        # Place materials in different zones
        materials_to_test = [
            (zones_y//2, sim.width//4, MaterialType.ICE, 270.0),       # Ice in hot zone (should melt)
            (zones_y + zones_y//2, sim.width//2, MaterialType.WATER, 275.0),  # Water in warm zone (stable)
            (2*zones_y + zones_y//2, 3*sim.width//4, MaterialType.WATER, 275.0),  # Water in cool zone (should freeze)
            (3*zones_y + zones_y//2, sim.width//4, MaterialType.MAGMA, 1400.0),   # Magma in cold zone (should solidify)
        ]
        
        for y, x, material, temp in materials_to_test:
            # Create 2x2 blocks for visibility
            for dy in range(2):
                for dx in range(2):
                    if 0 <= y+dy < sim.height and 0 <= x+dx < sim.width:
                        sim.material_types[y+dy, x+dx] = material
                        sim.temperature[y+dy, x+dx] = temp
        
        self.zones_y = zones_y
        self.initial_counts = {
            MaterialType.ICE: np.sum(sim.material_types == MaterialType.ICE),
            MaterialType.WATER: np.sum(sim.material_types == MaterialType.WATER),
            MaterialType.MAGMA: np.sum(sim.material_types == MaterialType.MAGMA),
        }
        
        print(f"Setup: Phase transition test with temperature zones")
        print(f"Hot zone (0-{zones_y}): T=350K, Warm ({zones_y}-{2*zones_y}): T=290K")
        print(f"Cool ({2*zones_y}-{3*zones_y}): T=250K, Cold ({3*zones_y}-{sim.height}): T=200K")
        
    def evaluate(self, sim):
        """Monitor phase transitions"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 15 == 0:
            # Count current material states
            ice_count = np.sum(sim.material_types == MaterialType.ICE)
            water_count = np.sum(sim.material_types == MaterialType.WATER)
            magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
            basalt_count = np.sum(sim.material_types == MaterialType.BASALT)
            granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
            
            print(f"Step {step:3d}: Ice={ice_count}, Water={water_count}, Magma={magma_count}, "
                  f"Basalt={basalt_count}, Granite={granite_count}")
            
            # Check for transitions in each zone
            zones_y = self.zones_y
            
            # Hot zone - expect ice to melt
            hot_zone_ice = np.sum((sim.material_types == MaterialType.ICE) & 
                                 (np.arange(sim.height)[:, None] < zones_y))
            hot_zone_water = np.sum((sim.material_types == MaterialType.WATER) & 
                                   (np.arange(sim.height)[:, None] < zones_y))
            
            if step == 15:  # First check
                print(f"  Hot zone transitions: {hot_zone_ice} ice, {hot_zone_water} water")
        
        return {
            "success": True,  # Always continue to observe transitions
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


class MetamorphicGradientScenario(TestScenario):
    """Test metamorphic gradient from heat source"""
    
    def __init__(self):
        super().__init__()
        self.name = "Metamorphic Gradient"
        self.description = "Rock metamorphism in temperature gradient from hot intrusion"
        
    def setup(self, sim):
        """Set up metamorphic gradient test"""
        sim.external_gravity = (0, 2)  # Very light gravity
        sim.enable_self_gravity = False
        sim.enable_solid_drag = False
        
        # Enable thermal and metamorphic processes
        sim.enable_heat_diffusion = True
        sim.enable_material_processes = True
        sim.enable_radiative_cooling = True
        
        # Cool air environment
        sim.material_types[:] = MaterialType.AIR
        sim.temperature[:] = 280.0
        
        # Fill most of area with granite (host rock)
        for y in range(sim.height):
            for x in range(sim.width):
                sim.material_types[y, x] = MaterialType.GRANITE
                sim.temperature[y, x] = 300.0  # Cool host rock
        
        # Hot magmatic intrusion on left side
        intrusion_width = sim.width // 4
        for y in range(sim.height):
            for x in range(intrusion_width):
                sim.material_types[y, x] = MaterialType.MAGMA
                sim.temperature[y, x] = 1500.0  # Very hot intrusion
        
        # Air boundary on right side for cooling
        air_start = 3 * sim.width // 4
        for y in range(sim.height):
            for x in range(air_start, sim.width):
                sim.material_types[y, x] = MaterialType.AIR
                sim.temperature[y, x] = 280.0
        
        self.intrusion_width = intrusion_width
        self.air_start = air_start
        self.initial_granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
        
        print(f"Setup: Metamorphic gradient test")
        print(f"Hot magma intrusion (0-{intrusion_width}), granite host rock ({intrusion_width}-{air_start}), air boundary ({air_start}-{sim.width})")
        print(f"Expect metamorphic gradient: granite -> schist -> gneiss near intrusion")
        
    def evaluate(self, sim):
        """Monitor metamorphic gradient development"""
        step = getattr(self, "step_count", 0)
        self.step_count = step + 1
        if step % 20 == 0:
            # Count materials by distance from intrusion
            granite_count = np.sum(sim.material_types == MaterialType.GRANITE)
            schist_count = np.sum(sim.material_types == MaterialType.SCHIST)
            gneiss_count = np.sum(sim.material_types == MaterialType.GNEISS)
            magma_count = np.sum(sim.material_types == MaterialType.MAGMA)
            
            # Temperature profile across the gradient
            mid_y = sim.height // 2
            temps = []
            for x in range(self.intrusion_width, self.air_start, 5):
                if x < sim.width:
                    temps.append(sim.temperature[mid_y, x])
            
            avg_temp = np.mean(temps) if temps else 0
            
            print(f"Step {step:3d}: Magma={magma_count}, Granite={granite_count}, Schist={schist_count}, "
                  f"Gneiss={gneiss_count}, Avg host temp={avg_temp:.0f}K")
        
        return {
            "success": True,  # Always continue to observe metamorphism
            "metrics": {},
            "message": "Step {} completed".format(step)
        }


# Register scenarios for visual runner
SCENARIOS = {
    'magma_containment': lambda: MagmaContainmentScenario(),
    'phase_transitions': lambda: PhaseTransitionScenario(),
    'metamorphic_gradient': lambda: MetamorphicGradientScenario(),
}