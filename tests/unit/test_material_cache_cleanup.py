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
        """Test that material cache is cleaned up after delete operations"""
        # Create small simulation for faster testing
        sim = GeologySimulation(10, 10)
        
        # Add specific materials to create cache entries using public API
        sim.add_material_blob(5, 5, 2, MaterialType.BASALT)
        sim.add_material_blob(3, 3, 1, MaterialType.GRANITE)
        sim.add_material_blob(7, 7, 1, MaterialType.WATER)
        
        # Get initial state
        initial_grid_materials = set(sim.material_types.flatten())
        initial_cached_materials = set(sim._material_props_cache.keys())
        
        # Cache should match grid initially
        assert initial_cached_materials == initial_grid_materials, \
            "Initial cache should match grid materials"
        
        # Delete some materials
        sim.delete_material_blob(5, 5, 3)  # Remove basalt area
        sim.delete_material_blob(3, 3, 2)  # Remove granite area
        
        # Deletion sets dirty flag; update material properties
        if sim._properties_dirty:
            sim._update_material_properties()
        
        # Run simulation step to trigger cache update
        sim.step_forward()
        
        # Get final state
        final_grid_materials = set(sim.material_types.flatten())
        final_cached_materials = set(sim._material_props_cache.keys())
        
        # Cache should match grid after cleanup
        extra_in_cache = final_cached_materials - final_grid_materials
        missing_from_cache = final_grid_materials - final_cached_materials
        
        assert not extra_in_cache, \
            f"Cache contains materials not in grid: {[m.name for m in extra_in_cache]}"
        assert not missing_from_cache, \
            f"Cache missing materials from grid: {[m.name for m in missing_from_cache]}"
        
        # Properties should not be dirty after update
        assert not sim._properties_dirty, "Properties should be clean after step"
    
    def test_cache_cleanup_during_phase_transitions(self):
        """Test cache cleanup when materials change due to phase transitions"""
        sim = GeologySimulation(8, 8)
        
        # Add water at high temperature to trigger evaporation
        sim.add_material_blob(4, 4, 2, MaterialType.WATER)
        
        # Set high temperature to trigger phase transition
        water_mask = (sim.material_types == MaterialType.WATER)
        sim.temperature[water_mask] = 400.0  # Above boiling point
        
        # Get initial cache state
        initial_cached = set(sim._material_props_cache.keys())
        
        # Run several steps to allow phase transitions
        for _ in range(3):
            sim.step_forward()
        
        # Check final cache state
        final_grid_materials = set(sim.material_types.flatten())
        final_cached_materials = set(sim._material_props_cache.keys())
        
        # Cache should still match grid
        assert final_cached_materials == final_grid_materials, \
            "Cache should match grid after phase transitions"
    
    def test_cache_cleanup_with_material_swaps(self):
        """Test cache cleanup when materials are swapped during fluid dynamics"""
        sim = GeologySimulation(6, 6)
        
        # Create a scenario with different density materials
        sim.add_material_blob(3, 2, 1, MaterialType.GRANITE)  # Heavy
        sim.add_material_blob(3, 4, 1, MaterialType.AIR)     # Light
        
        # Get initial state
        initial_materials = set(sim.material_types.flatten())
        
        # Run steps to allow material movement
        for _ in range(5):
            sim.step_forward()
            
            # Cache should always match grid
            current_grid = set(sim.material_types.flatten())
            current_cache = set(sim._material_props_cache.keys())
            
            assert current_cache == current_grid, \
                f"Cache mismatch at step: grid={len(current_grid)}, cache={len(current_cache)}"
    
    def test_no_memory_leak_in_cache(self):
        """Test that cache doesn't grow indefinitely with repeated operations"""
        sim = GeologySimulation(8, 8)
        
        # Perform many add/delete cycles
        for i in range(10):
            # Add material
            sim.add_material_blob(4, 4, 1, MaterialType.BASALT)
            sim.step_forward()
            
            # Delete material
            sim.delete_material_blob(4, 4, 2)
            sim.step_forward()
            
            # Cache size should remain reasonable
            cache_size = len(sim._material_props_cache)
            grid_materials = len(set(sim.material_types.flatten()))
            
            assert cache_size == grid_materials, \
                f"Cache size ({cache_size}) should match grid materials ({grid_materials}) at cycle {i}"
            
            # Cache should not grow beyond reasonable bounds
            assert cache_size < 20, f"Cache size ({cache_size}) is too large at cycle {i}"
