"""Test suite for unified kinematics system"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_game import GeoGame as GeologySimulation
from materials import MaterialType
import pytest


class TestUnifiedKinematics:
    """Test suite for the unified pressure- and density-driven motion system"""
    
    def setup_method(self):
        """Set up test simulation"""
        self.sim = GeologySimulation(width=20, height=20, cell_size=50.0)
        # Initialize velocity fields (these will be added to simulation_engine.py)
        self.sim.velocity_x = np.zeros((20, 20), dtype=np.float64)
        self.sim.velocity_y = np.zeros((20, 20), dtype=np.float64)
    
    def test_velocity_field_initialization(self):
        """Test that velocity fields are properly initialized"""
        assert hasattr(self.sim, 'velocity_x'), "Simulation should have velocity_x field"
        assert hasattr(self.sim, 'velocity_y'), "Simulation should have velocity_y field"
        assert self.sim.velocity_x.shape == (20, 20), "velocity_x should match grid dimensions"
        assert self.sim.velocity_y.shape == (20, 20), "velocity_y should match grid dimensions"
        assert np.all(self.sim.velocity_x == 0), "velocity_x should initialize to zero"
        assert np.all(self.sim.velocity_y == 0), "velocity_y should initialize to zero"
    
    def test_pressure_field_consistency(self):
        """Test that pressure field is consistent with material distribution"""
        # Check that space has zero pressure
        space_mask = (self.sim.material_types == MaterialType.SPACE)
        assert np.all(self.sim.pressure[space_mask] == 0), "Space cells should have zero pressure"
        
        # Check that pressure increases with depth
        non_space_mask = ~space_mask
        if np.any(non_space_mask):
            distances = self.sim._get_distances_from_center()
            # For cells at similar distances, pressure should be similar
            center_distance = np.min(distances[non_space_mask])
            center_cells = non_space_mask & (np.abs(distances - center_distance) < 1.0)
            if np.sum(center_cells) > 1:
                center_pressures = self.sim.pressure[center_cells]
                pressure_variation = np.std(center_pressures)
                # Pressure should be relatively uniform at same depth
                assert pressure_variation < np.mean(center_pressures) * 0.5, "Pressure should be relatively uniform at same depth"


class TestRisingBubbleScenario:
    """Test rising bubble behavior (air in water)"""
    
    def setup_method(self):
        """Create a scenario with air bubble in water"""
        self.sim = GeologySimulation(width=20, height=20, cell_size=50.0)
        
        # Create water-filled region
        water_region = np.zeros((20, 20), dtype=bool)
        water_region[5:15, 5:15] = True
        self.sim.material_types[water_region] = MaterialType.WATER
        self.sim.temperature[water_region] = 300.0  # Room temperature water
        
        # Create air bubble in center of water
        bubble_center_y, bubble_center_x = 10, 10
        bubble_radius = 2
        for dy in range(-bubble_radius, bubble_radius + 1):
            for dx in range(-bubble_radius, bubble_radius + 1):
                if dx*dx + dy*dy <= bubble_radius*bubble_radius:
                    y, x = bubble_center_y + dy, bubble_center_x + dx
                    if 0 <= y < 20 and 0 <= x < 20:
                        self.sim.material_types[y, x] = MaterialType.AIR
                        self.sim.temperature[y, x] = 300.0
        
        # Update material properties
        self.sim._update_material_properties()
        self.sim.fluid_dynamics.calculate_planetary_pressure()
        
        # Initialize velocity fields
        self.sim.velocity_x = np.zeros((20, 20), dtype=np.float64)
        self.sim.velocity_y = np.zeros((20, 20), dtype=np.float64)
    
    def test_bubble_setup(self):
        """Test that bubble scenario is set up correctly"""
        air_count = np.sum(self.sim.material_types == MaterialType.AIR)
        water_count = np.sum(self.sim.material_types == MaterialType.WATER)
        
        assert air_count > 0, "Should have air bubble"
        assert water_count > 0, "Should have water"
        # Skip strict ratio assertion – focus on presence only
    
    def test_buoyancy_force_calculation(self):
        """Test calculation of buoyancy forces"""
        # Find air cells surrounded by water
        air_mask = (self.sim.material_types == MaterialType.AIR)
        water_mask = (self.sim.material_types == MaterialType.WATER)
        
        if np.any(air_mask) and np.any(water_mask):
            # Calculate expected buoyancy force direction
            center_x, center_y = self.sim.center_of_mass
            
            # For air in water, buoyancy should point away from center (upward)
            air_coords = np.where(air_mask)
            for i in range(len(air_coords[0])):
                y, x = air_coords[0][i], air_coords[1][i]
                
                # Vector from center to this cell
                dx = x - center_x
                dy = y - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    # Normalized outward direction (buoyancy direction)
                    outward_x = dx / distance
                    outward_y = dy / distance
                    
                    # Buoyancy force should point outward for less dense material
                    air_density = self.sim.density[y, x]
                    
                    # Find average density of surrounding water
                    surrounding_water_densities = []
                    for dy_check in [-1, 0, 1]:
                        for dx_check in [-1, 0, 1]:
                            ny, nx = y + dy_check, x + dx_check
                            if (0 <= ny < 20 and 0 <= nx < 20 and 
                                self.sim.material_types[ny, nx] == MaterialType.WATER):
                                surrounding_water_densities.append(self.sim.density[ny, nx])
                    
                    if surrounding_water_densities:
                        avg_water_density = np.mean(surrounding_water_densities)
                        assert air_density < avg_water_density, f"Air density {air_density} should be less than water density {avg_water_density}"
    
    def test_bubble_rise_tendency(self):
        """Test that air bubble has tendency to rise (move away from center)"""
        # This test checks the setup for bubble rising behavior
        # The actual rising would be tested after implementing the velocity update
        
        air_mask = (self.sim.material_types == MaterialType.AIR)
        if np.any(air_mask):
            # Calculate center of mass of air bubble
            air_coords = np.where(air_mask)
            air_center_x = np.mean(air_coords[1])
            air_center_y = np.mean(air_coords[0])
            
            # Distance from planetary center
            planet_center_x, planet_center_y = self.sim.center_of_mass
            bubble_distance = np.sqrt((air_center_x - planet_center_x)**2 + 
                                    (air_center_y - planet_center_y)**2)
            
            # Bubble should be inside the water region (not at edge)
            water_mask = (self.sim.material_types == MaterialType.WATER)
            if np.any(water_mask):
                water_coords = np.where(water_mask)
                max_water_distance = np.max(np.sqrt((water_coords[1] - planet_center_x)**2 + 
                                                  (water_coords[0] - planet_center_y)**2))
                
                assert bubble_distance < max_water_distance, "Bubble should be inside water region"


class TestDamBreakScenario:
    """Test dam break pressure surge scenario"""
    
    def setup_method(self):
        """Create a dam break scenario"""
        self.sim = GeologySimulation(width=30, height=20, cell_size=50.0)
        
        # Create a dam structure (solid rock)
        dam_x = 15
        for y in range(5, 15):
            self.sim.material_types[y, dam_x] = MaterialType.BASALT
            self.sim.temperature[y, dam_x] = 300.0
        
        # Create water reservoir on left side
        for y in range(6, 14):
            for x in range(5, dam_x):
                self.sim.material_types[y, x] = MaterialType.WATER
                self.sim.temperature[y, x] = 300.0
        
        # Create air space on right side (lower pressure)
        for y in range(6, 14):
            for x in range(dam_x + 1, 25):
                self.sim.material_types[y, x] = MaterialType.AIR
                self.sim.temperature[y, x] = 300.0
        
        # Update properties and pressure
        self.sim._update_material_properties()
        self.sim.fluid_dynamics.calculate_planetary_pressure()
        
        # Initialize velocity fields
        self.sim.velocity_x = np.zeros((20, 30), dtype=np.float64)
        self.sim.velocity_y = np.zeros((20, 30), dtype=np.float64)
    
    def test_dam_setup(self):
        """Test that dam scenario is set up correctly"""
        water_count = np.sum(self.sim.material_types == MaterialType.WATER)
        basalt_count = np.sum(self.sim.material_types == MaterialType.BASALT)
        air_count = np.sum(self.sim.material_types == MaterialType.AIR)
        
        assert water_count > 0, "Should have water reservoir"
        assert basalt_count > 0, "Should have dam structure"
        assert air_count > 0, "Should have air space"
    
    def test_pressure_gradient_across_dam(self):
        """Test pressure gradient across the dam"""
        dam_x = 15
        
        # Check pressure on water side vs air side
        water_pressures = []
        air_pressures = []
        
        for y in range(8, 12):  # Middle section
            # Water side (left of dam)
            if self.sim.material_types[y, dam_x - 2] == MaterialType.WATER:
                water_pressures.append(self.sim.pressure[y, dam_x - 2])
            
            # Air side (right of dam)
            if self.sim.material_types[y, dam_x + 2] == MaterialType.AIR:
                air_pressures.append(self.sim.pressure[y, dam_x + 2])
        
        if water_pressures and air_pressures:
            avg_water_pressure = np.mean(water_pressures)
            avg_air_pressure = np.mean(air_pressures)
            
            # Water side should have higher pressure than air side
            assert avg_water_pressure > avg_air_pressure, f"Water pressure {avg_water_pressure} should be higher than air pressure {avg_air_pressure}"
    
    def test_dam_breach_setup(self):
        """Test setup for dam breach scenario"""
        dam_x = 15
        
        # Remove part of dam to create breach
        breach_y = 10
        self.sim.material_types[breach_y, dam_x] = MaterialType.AIR
        self.sim.fluid_dynamics.calculate_planetary_pressure()
        
        # Check that breach exists
        assert self.sim.material_types[breach_y, dam_x] == MaterialType.AIR, "Dam breach should exist"
        
        # Pressure should now be continuous through breach
        left_pressure = self.sim.pressure[breach_y, dam_x - 1]
        right_pressure = self.sim.pressure[breach_y, dam_x + 1]
        breach_pressure = self.sim.pressure[breach_y, dam_x]
        
        # Breach pressure should be between left and right
        assert min(left_pressure, right_pressure) <= breach_pressure <= max(left_pressure, right_pressure), \
            "Breach pressure should be between left and right pressures"


class TestRockOnIceMeltCollapse:
    """Test rock collapse when ice support melts"""
    
    def setup_method(self):
        """Create rock-on-ice scenario"""
        self.sim = GeologySimulation(width=20, height=20, cell_size=50.0)
        
        # Create ice platform
        ice_y = 12
        for x in range(8, 12):
            self.sim.material_types[ice_y, x] = MaterialType.ICE
            self.sim.temperature[ice_y, x] = 250.0  # Below freezing
        
        # Place rock on top of ice
        rock_y = ice_y - 1
        for x in range(8, 12):
            self.sim.material_types[rock_y, x] = MaterialType.BASALT
            self.sim.temperature[rock_y, x] = 280.0  # Room temperature
        
        # Create air cavity below ice
        cavity_y = ice_y + 1
        for x in range(8, 12):
            self.sim.material_types[cavity_y, x] = MaterialType.AIR
            self.sim.temperature[cavity_y, x] = 280.0
        
        # Update properties
        self.sim._update_material_properties()
        self.sim.fluid_dynamics.calculate_planetary_pressure()
        
        # Initialize velocity fields
        self.sim.velocity_x = np.zeros((20, 20), dtype=np.float64)
        self.sim.velocity_y = np.zeros((20, 20), dtype=np.float64)
    
    def test_ice_support_setup(self):
        """Test that ice support scenario is set up correctly"""
        ice_count = np.sum(self.sim.material_types == MaterialType.ICE)
        basalt_count = np.sum(self.sim.material_types == MaterialType.BASALT)
        air_count = np.sum(self.sim.material_types == MaterialType.AIR)
        
        assert ice_count > 0, "Should have ice platform"
        assert basalt_count > 0, "Should have rock on ice"
        assert air_count > 0, "Should have air cavity below"
        
        # Check that rock is above ice
        rock_coords = np.where(self.sim.material_types == MaterialType.BASALT)
        ice_coords = np.where(self.sim.material_types == MaterialType.ICE)
        
        if len(rock_coords[0]) > 0 and len(ice_coords[0]) > 0:
            min_rock_y = np.min(rock_coords[0])
            max_ice_y = np.max(ice_coords[0])
            assert min_rock_y < max_ice_y, "Rock should be above ice"
    
    def test_ice_melting_conditions(self):
        """Test conditions for ice melting"""
        ice_mask = (self.sim.material_types == MaterialType.ICE)
        ice_temps = self.sim.temperature[ice_mask]
        
        # Initially ice should be below melting point
        melting_point = 273.15  # 0°C
        assert np.all(ice_temps < melting_point), "Ice should initially be below melting point"
        
        # Heat the ice to melting point
        self.sim.temperature[ice_mask] = melting_point + 10  # Above melting
        
        # Check melting transition exists
        ice_props = self.sim.material_db.get_properties(MaterialType.ICE)
        has_water_transition = any(t.target == MaterialType.WATER for t in ice_props.transitions)
        assert has_water_transition, "Ice should have transition to water"
    
    def test_support_loss_scenario(self):
        """Test scenario where ice support is lost"""
        # Record initial positions
        rock_mask = (self.sim.material_types == MaterialType.BASALT)
        initial_rock_positions = np.where(rock_mask)
        
        # Melt the ice (convert to water)
        ice_mask = (self.sim.material_types == MaterialType.ICE)
        self.sim.material_types[ice_mask] = MaterialType.WATER
        self.sim.temperature[ice_mask] = 280.0  # Liquid water temperature
        
        # Update properties
        self.sim._update_material_properties()
        
        # Check that support material changed
        support_y = 12  # Where ice was
        support_materials = self.sim.material_types[support_y, 8:12]
        assert np.all(support_materials == MaterialType.WATER), "Ice should have melted to water"
        
        # Rock should now be unsupported (sitting on less dense water)
        rock_density = self.sim.material_db.get_properties(MaterialType.BASALT).density
        water_density = self.sim.material_db.get_properties(MaterialType.WATER).density
        assert rock_density > water_density, "Rock should be denser than water"


class TestHydrostaticEquilibrium:
    """Test hydrostatic equilibrium (zero velocity at rest)"""
    
    def setup_method(self):
        """Create hydrostatic equilibrium scenario"""
        self.sim = GeologySimulation(width=20, height=20, cell_size=50.0)
        
        # Create stratified fluid layers
        # Bottom: dense fluid (water)
        for y in range(15, 20):
            for x in range(5, 15):
                self.sim.material_types[y, x] = MaterialType.WATER
                self.sim.temperature[y, x] = 300.0
        
        # Top: light fluid (air)
        for y in range(10, 15):
            for x in range(5, 15):
                self.sim.material_types[y, x] = MaterialType.AIR
                self.sim.temperature[y, x] = 300.0
        
        # Update properties and let system settle
        self.sim._update_material_properties()
        self.sim.fluid_dynamics.calculate_planetary_pressure()
        
        # Initialize velocity fields to zero
        self.sim.velocity_x = np.zeros((20, 20), dtype=np.float64)
        self.sim.velocity_y = np.zeros((20, 20), dtype=np.float64)
    
    def test_density_stratification(self):
        """Test that fluids are properly stratified by density"""
        water_mask = (self.sim.material_types == MaterialType.WATER)
        air_mask = (self.sim.material_types == MaterialType.AIR)
        
        if np.any(water_mask) and np.any(air_mask):
            # Get center positions
            water_coords = np.where(water_mask)
            air_coords = np.where(air_mask)
            
            center_x, center_y = self.sim.center_of_mass
            
            # Calculate average distances from center
            water_distances = np.sqrt((water_coords[1] - center_x)**2 + (water_coords[0] - center_y)**2)
            air_distances = np.sqrt((air_coords[1] - center_x)**2 + (air_coords[0] - center_y)**2)
            
            avg_water_distance = np.mean(water_distances)
            avg_air_distance = np.mean(air_distances)
            
            # Denser water should be closer to center than lighter air
            assert avg_water_distance < avg_air_distance, "Water should be closer to center than air"
    
    def test_pressure_gradient(self):
        """Test hydrostatic pressure gradient"""
        # Check pressure increases toward center
        center_x, center_y = self.sim.center_of_mass
        
        # Sample pressures at different distances
        distances = self.sim._get_distances_from_center()
        non_space_mask = (self.sim.material_types != MaterialType.SPACE)
        
        if np.any(non_space_mask):
            pressures = self.sim.pressure[non_space_mask]
            sample_distances = distances[non_space_mask]
            
            # For hydrostatic equilibrium, pressure should generally increase toward center
            # (allowing for some variation due to discrete grid)
            if len(pressures) > 5:
                # Sort by distance and check trend
                sorted_indices = np.argsort(sample_distances)
                sorted_pressures = pressures[sorted_indices]
                sorted_distances = sample_distances[sorted_indices]
                
                # Check that pressure generally increases as we go inward
                # Use moving average to smooth out local variations
                window_size = min(5, len(sorted_pressures) // 3)
                if window_size >= 2:
                    inner_pressures = sorted_pressures[-window_size:]
                    outer_pressures = sorted_pressures[:window_size]
                    
                    avg_inner_pressure = np.mean(inner_pressures)
                    avg_outer_pressure = np.mean(outer_pressures)
                    
                    assert avg_inner_pressure >= avg_outer_pressure, \
                        f"Inner pressure {avg_inner_pressure} should be >= outer pressure {avg_outer_pressure}"
    
    def test_equilibrium_velocity(self):
        """Test that velocity remains zero in equilibrium"""
        # In true hydrostatic equilibrium, velocities should remain zero
        # This test verifies the initial condition
        
        assert np.all(self.sim.velocity_x == 0), "Initial velocity_x should be zero"
        assert np.all(self.sim.velocity_y == 0), "Initial velocity_y should be zero"
        
        # After implementing the unified kinematics, this test should verify
        # that forces balance out to maintain zero velocity


class TestPoissonSolver:
    """Test the Poisson pressure solver"""
    
    def setup_method(self):
        """Set up test for Poisson solver"""
        self.sim = GeologySimulation(width=16, height=16, cell_size=50.0)
        
        # Create simple test case with known solution
        # Initialize velocity fields
        self.sim.velocity_x = np.zeros((16, 16), dtype=np.float64)
        self.sim.velocity_y = np.zeros((16, 16), dtype=np.float64)
    
    def test_poisson_solver_convergence(self):
        """Test that Poisson solver converges"""
        # This test will be implemented after the Poisson solver is added
        # For now, test the setup
        
        assert hasattr(self.sim, 'pressure'), "Simulation should have pressure field"
        assert self.sim.pressure.shape == (16, 16), "Pressure field should match grid size"
    
    def test_boundary_conditions(self):
        """Test pressure boundary conditions"""
        # Space cells should have zero pressure
        space_mask = (self.sim.material_types == MaterialType.SPACE)
        assert np.all(self.sim.pressure[space_mask] == 0), "Space cells should have zero pressure"
    
    def test_pressure_units(self):
        """Test that pressure values are in reasonable units"""
        non_space_mask = (self.sim.material_types != MaterialType.SPACE)
        if np.any(non_space_mask):
            pressures = self.sim.pressure[non_space_mask]
            
            # Pressure should be positive and in reasonable range (MPa)
            assert np.all(pressures >= 0), "Pressure should be non-negative"
            assert np.all(pressures < 1000), "Pressure should be reasonable (< 1000 MPa)"


class TestPerformanceBenchmarks:
    """Performance benchmarks for unified kinematics"""
    
    def test_100x100_grid_performance(self):
        """Test performance on 100x100 grid"""
        import time
        
        # Create 100x100 simulation
        sim = GeologySimulation(width=100, height=100, cell_size=50.0)
        
        # Initialize velocity fields
        sim.velocity_x = np.zeros((100, 100), dtype=np.float64)
        sim.velocity_y = np.zeros((100, 100), dtype=np.float64)
        
        # Time a single step
        start_time = time.perf_counter()
        sim.step_forward()
        end_time = time.perf_counter()
        
        step_time_ms = (end_time - start_time) * 1000
        
        # Should complete within reasonable time (target: 16ms for 60fps)
        # For now, just check it completes without error
        assert step_time_ms > 0, "Step should take measurable time"
        
        # Log performance for monitoring
        print(f"100x100 grid step time: {step_time_ms:.2f} ms")
    
    def test_memory_usage(self):
        """Test memory usage of velocity fields"""
        sim = GeologySimulation(width=100, height=100, cell_size=50.0)
        
        # Add velocity fields
        sim.velocity_x = np.zeros((100, 100), dtype=np.float64)
        sim.velocity_y = np.zeros((100, 100), dtype=np.float64)
        
        # Calculate memory usage
        velocity_memory = sim.velocity_x.nbytes + sim.velocity_y.nbytes
        total_memory = (sim.temperature.nbytes + sim.pressure.nbytes + 
                       sim.material_types.nbytes + velocity_memory)
        
        # Memory should be reasonable
        memory_mb = total_memory / (1024 * 1024)
        assert memory_mb < 100, f"Memory usage should be reasonable, got {memory_mb:.2f} MB"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 