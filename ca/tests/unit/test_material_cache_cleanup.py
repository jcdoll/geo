"""
Test material cache cleanup after delete operations.

This test ensures that the material properties cache is properly cleaned up
when materials are deleted from the simulation, preventing stale cache entries
that could cause unexpected material conversions.
"""

import pytest
import numpy as np
from geo_game import GeoGame as GeologySimulation
from materials import MaterialType


class TestMaterialCacheCleanup:
    """Test material cache cleanup functionality"""
    
    def test_cache_cleanup_after_delete(self):
        """Test that material properties are correctly updated after delete operations"""
        # Create small simulation for faster testing
        sim = GeologySimulation(10, 10, setup_planet=False)
        
        # Initialize with space
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 290.0
        
        # Add specific materials using public API
        sim.add_material_blob(5, 5, 2, MaterialType.BASALT)
        sim.add_material_blob(3, 3, 1, MaterialType.GRANITE)
        sim.add_material_blob(7, 7, 1, MaterialType.WATER)
        sim._update_material_properties(force=True)
        
        # Verify materials were added
        assert MaterialType.BASALT in sim.material_types
        assert MaterialType.GRANITE in sim.material_types
        assert MaterialType.WATER in sim.material_types
        
        # Check densities are correct
        basalt_pos = np.where(sim.material_types == MaterialType.BASALT)
        if len(basalt_pos[0]) > 0:
            basalt_density = sim.density[basalt_pos[0][0], basalt_pos[1][0]]
            assert basalt_density == sim.material_db.get_properties(MaterialType.BASALT).density
        
        # Delete some materials
        sim.delete_material_blob(5, 5, 3)  # Remove basalt area
        sim.delete_material_blob(3, 3, 2)  # Remove granite area
        
        # Run simulation step to trigger property update
        sim.step_forward(0.1)
        
        # Check that deleted areas now have correct properties
        space_density = sim.material_db.get_properties(MaterialType.SPACE).density
        deleted_area_density = sim.density[5, 5]
        assert deleted_area_density == space_density, f"Deleted area should have space density, got {deleted_area_density} instead of {space_density}"
    
    def test_cache_cleanup_during_phase_transitions(self):
        """Test material properties are updated when materials change due to phase transitions"""
        sim = GeologySimulation(8, 8, setup_planet=False)
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 290.0
        
        # Add water at high temperature to trigger evaporation
        sim.add_material_blob(4, 4, 2, MaterialType.WATER)
        
        # Set high temperature to trigger phase transition
        water_mask = (sim.material_types == MaterialType.WATER)
        sim.temperature[water_mask] = 400.0  # Above boiling point
        
        # Force property update
        sim._update_material_properties(force=True)
        
        # Store initial water density for comparison
        water_density = sim.material_db.get_properties(MaterialType.WATER).density
        
        # Run several steps to allow phase transitions
        for _ in range(3):
            sim.step_forward(0.1)
        
        # Check that properties were updated correctly
        # Some water should have evaporated to water vapor
        vapor_mask = sim.material_types == MaterialType.WATER_VAPOR
        if np.any(vapor_mask):
            vapor_density = sim.density[vapor_mask][0]
            expected_vapor_density = sim.material_db.get_properties(MaterialType.WATER_VAPOR).density
            assert vapor_density == expected_vapor_density, "Vapor should have correct density"
    
    def test_cache_cleanup_with_material_swaps(self):
        """Test material properties remain consistent when materials are swapped during fluid dynamics"""
        sim = GeologySimulation(6, 6, setup_planet=False)
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 290.0
        
        # Create a scenario with different density materials
        sim.add_material_blob(3, 2, 1, MaterialType.GRANITE)  # Heavy
        sim.add_material_blob(3, 4, 1, MaterialType.AIR)     # Light
        
        # Force property update
        sim._update_material_properties(force=True)
        
        # Get initial densities
        granite_density = sim.material_db.get_properties(MaterialType.GRANITE).density
        air_density = sim.material_db.get_properties(MaterialType.AIR).density
        
        # Run steps to allow material movement
        for _ in range(5):
            sim.step_forward(0.1)
            
            # Check that all cells have correct densities
            for j in range(sim.height):
                for i in range(sim.width):
                    mat = sim.material_types[j, i]
                    actual_density = sim.density[j, i]
                    expected_density = sim.material_db.get_properties(mat).density
                    assert actual_density == expected_density, \
                        f"Density mismatch at ({j},{i}): {actual_density} != {expected_density}"
    
    def test_no_memory_leak_in_cache(self):
        """Test that material properties remain consistent with repeated operations"""
        sim = GeologySimulation(8, 8, setup_planet=False)
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 290.0
        
        # Perform many add/delete cycles
        for i in range(10):
            # Add material
            sim.add_material_blob(4, 4, 1, MaterialType.BASALT)
            sim.step_forward(0.1)
            
            # Check basalt density is correct
            basalt_mask = sim.material_types == MaterialType.BASALT
            if np.any(basalt_mask):
                actual = sim.density[basalt_mask][0]
                expected = sim.material_db.get_properties(MaterialType.BASALT).density
                assert actual == expected, f"Basalt density mismatch at cycle {i}"
            
            # Delete material
            sim.delete_material_blob(4, 4, 2)
            sim.step_forward(0.1)
            
            # Check space density is correct where deleted
            space_mask = sim.material_types == MaterialType.SPACE
            if np.any(space_mask):
                actual = sim.density[space_mask][0]
                expected = sim.material_db.get_properties(MaterialType.SPACE).density
                assert actual == expected, f"Space density mismatch at cycle {i}"
