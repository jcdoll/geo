#!/usr/bin/env python3
"""
Test suite for rock types and rock database functionality.
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rock_types import RockType, RockDatabase, RockProperties


class TestRockType:
    """Test cases for the RockType enum"""
    
    def test_rock_type_values(self):
        """Test that rock types have expected values"""
        assert RockType.GRANITE.value == 'granite'
        assert RockType.BASALT.value == 'basalt'
        assert RockType.LIMESTONE.value == 'limestone'
        assert RockType.SHALE.value == 'shale'
        assert RockType.MAGMA.value == 'magma'
    
    def test_rock_type_enumeration(self):
        """Test that we can enumerate all rock types"""
        rock_types = list(RockType)
        assert len(rock_types) > 10, "Should have multiple rock types defined"
        
        # Check some expected types are present
        expected_types = [RockType.GRANITE, RockType.BASALT, RockType.LIMESTONE, 
                         RockType.SHALE, RockType.SLATE, RockType.MARBLE, RockType.MAGMA]
        for rock_type in expected_types:
            assert rock_type in rock_types, f"{rock_type} should be in rock types"


class TestRockProperties:
    """Test cases for the RockProperties dataclass"""
    
    def test_rock_properties_creation(self):
        """Test creating rock properties"""
        props = RockProperties(
            density=2650.0,
            melting_point=1200.0,
            thermal_conductivity=2.5,
            specific_heat=790.0,
            strength=200.0,
            porosity=0.01,
            metamorphic_threshold_temp=650.0,
            metamorphic_threshold_pressure=200.0,
            color_rgb=(128, 128, 128)
        )
        
        assert props.density == 2650.0
        assert props.melting_point == 1200.0
        assert props.thermal_conductivity == 2.5
        assert props.specific_heat == 790.0
        assert props.strength == 200.0
        assert props.porosity == 0.01
        assert props.color_rgb == (128, 128, 128)
    
    def test_rock_properties_validation(self):
        """Test that rock properties have reasonable values"""
        props = RockProperties(
            density=2650.0,
            melting_point=1200.0,
            thermal_conductivity=2.5,
            specific_heat=790.0,
            strength=200.0,
            porosity=0.01,
            metamorphic_threshold_temp=650.0,
            metamorphic_threshold_pressure=200.0,
            color_rgb=(128, 128, 128)
        )
        
        assert props.density > 0, "Density should be positive"
        assert props.melting_point > 0, "Melting point should be positive"
        assert props.thermal_conductivity > 0, "Thermal conductivity should be positive"
        assert len(props.color_rgb) == 3, "Color should be RGB tuple"
        assert all(0 <= c <= 255 for c in props.color_rgb), "RGB values should be in range [0, 255]"


class TestRockDatabase:
    """Test cases for the RockDatabase class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.db = RockDatabase()
    
    def test_database_initialization(self):
        """Test that database initializes with all rock types"""
        # Should have properties for all rock types
        for rock_type in RockType:
            props = self.db.get_properties(rock_type)
            assert props is not None, f"Should have properties for {rock_type}"
            assert isinstance(props, RockProperties), f"Properties should be RockProperties instance for {rock_type}"
    
    def test_get_properties(self):
        """Test getting rock properties"""
        # Test specific rock types
        granite_props = self.db.get_properties(RockType.GRANITE)
        assert granite_props.density > 2000, "Granite density should be reasonable"
        assert granite_props.melting_point > 1000, "Granite melting point should be high"
        assert granite_props.thermal_conductivity > 0, "Granite should conduct heat"
        
        basalt_props = self.db.get_properties(RockType.BASALT)
        assert basalt_props.density > 2000, "Basalt density should be reasonable"
        assert basalt_props.melting_point < granite_props.melting_point, "Basalt should melt easier than granite"
        
        # Properties should be different for different rocks
        assert granite_props != basalt_props, "Different rocks should have different properties"
    
    def test_metamorphic_transitions_shale_sequence(self):
        """Test the shale -> slate -> schist -> gneiss sequence"""
        # Low grade: shale -> slate
        product = self.db.get_metamorphic_product(RockType.SHALE, 300, 50)
        assert product == RockType.SLATE, f"Shale at low P/T should become slate, got {product}"
        
        # Medium grade: slate -> schist (need higher pressure)
        product = self.db.get_metamorphic_product(RockType.SLATE, 500, 200)
        assert product == RockType.SCHIST, f"Slate at medium P/T should become schist, got {product}"
        
        # High grade: schist -> gneiss (need higher pressure)
        product = self.db.get_metamorphic_product(RockType.SCHIST, 700, 400)
        assert product == RockType.GNEISS, f"Schist at high P/T should become gneiss, got {product}"
    
    def test_metamorphic_transitions_other_rocks(self):
        """Test metamorphic transitions for other rock types"""
        # Limestone -> marble
        product = self.db.get_metamorphic_product(RockType.LIMESTONE, 500, 150)
        assert product == RockType.MARBLE, f"Limestone should become marble, got {product}"
        
        # Sandstone -> quartzite
        product = self.db.get_metamorphic_product(RockType.SANDSTONE, 400, 100)
        assert product == RockType.QUARTZITE, f"Sandstone should become quartzite, got {product}"
        
        # Granite -> gneiss (need higher temperature)
        product = self.db.get_metamorphic_product(RockType.GRANITE, 650, 200)
        assert product == RockType.GNEISS, f"Granite should become gneiss, got {product}"
    
    def test_metamorphic_no_change_conditions(self):
        """Test that rocks don't change under low P/T conditions"""
        # Very low conditions should not cause change (returns None = no change)
        product = self.db.get_metamorphic_product(RockType.GRANITE, 100, 10)
        assert product is None, "Granite should not change at very low P/T (returns None)"
        
        product = self.db.get_metamorphic_product(RockType.LIMESTONE, 200, 20)
        assert product is None, "Limestone should not change at low P/T (returns None)"
    
    def test_melting_behavior(self):
        """Test rock melting behavior"""
        # Test various rocks at high temperature
        assert self.db.should_melt(RockType.GRANITE, 1300), "Granite should melt at 1300°C"
        assert self.db.should_melt(RockType.BASALT, 1200), "Basalt should melt at 1200°C"
        assert self.db.should_melt(RockType.LIMESTONE, 1500), "Limestone should melt at very high temp"
        
        # Test at low temperatures
        assert not self.db.should_melt(RockType.GRANITE, 800), "Granite should not melt at 800°C"
        assert not self.db.should_melt(RockType.BASALT, 700), "Basalt should not melt at 700°C"
    
    def test_melting_edge_cases(self):
        """Test melting behavior at edge cases"""
        # Test exactly at melting point
        granite_props = self.db.get_properties(RockType.GRANITE)
        melting_point = granite_props.melting_point
        
        assert self.db.should_melt(RockType.GRANITE, melting_point + 1), "Should melt just above melting point"
        assert not self.db.should_melt(RockType.GRANITE, melting_point - 1), "Should not melt just below melting point"
        
        # Magma should not "melt" (it's already molten)
        assert not self.db.should_melt(RockType.MAGMA, 2000), "Magma should not melt (already molten)"
    
    def test_color_consistency(self):
        """Test that rock colors are reasonable"""
        for rock_type in RockType:
            props = self.db.get_properties(rock_type)
            color = props.color_rgb
            
            assert len(color) == 3, f"{rock_type} should have RGB color tuple"
            assert all(isinstance(c, int) for c in color), f"{rock_type} color should be integers"
            assert all(0 <= c <= 255 for c in color), f"{rock_type} color values should be in [0, 255]"
    
    def test_physical_property_ranges(self):
        """Test that physical properties are in reasonable ranges"""
        for rock_type in RockType:
            props = self.db.get_properties(rock_type)
            
            # Density should be reasonable for rocks (kg/m³) - pumice and air are exceptions
            if rock_type in [RockType.PUMICE, RockType.AIR]:
                assert props.density > 0, f"{rock_type} density should be positive"
            elif rock_type == RockType.WATER:
                assert props.density == 1000, f"{rock_type} should have water density"
            else:
                assert 1500 <= props.density <= 5000, f"{rock_type} density {props.density} should be in reasonable range"
            
            # Melting point should be reasonable (°C) - special cases for water and air
            if rock_type == RockType.WATER:
                assert props.melting_point == 0, f"Water should have 0°C melting point"
            elif rock_type == RockType.AIR:
                assert props.melting_point < 0, f"Air should have negative melting point"
            else:
                assert 500 <= props.melting_point <= 2000, f"{rock_type} melting point {props.melting_point} should be reasonable"
            
            # Thermal conductivity should be positive
            assert props.thermal_conductivity > 0, f"{rock_type} thermal conductivity should be positive"
            assert props.thermal_conductivity < 100, f"{rock_type} thermal conductivity should be reasonable" 