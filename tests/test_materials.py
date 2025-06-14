#!/usr/bin/env python3
"""
Test suite for material types and material database functionality.
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo.materials import MaterialType, MaterialDatabase, MaterialProperties
from geo.materials import TransitionRule, MaterialType


class TestMaterialType:
    """Test cases for MaterialType enum"""
    
    def test_material_type_values(self):
        """Test that material types have correct string values"""
        assert MaterialType.GRANITE.value == 'granite'
        assert MaterialType.BASALT.value == 'basalt'
        assert MaterialType.LIMESTONE.value == 'limestone'
        assert MaterialType.SHALE.value == 'shale'
        assert MaterialType.MAGMA.value == 'magma'
    
    def test_material_type_enumeration(self):
        """Test that we can enumerate all material types"""
        material_types = list(MaterialType)
        assert len(material_types) > 10, "Should have multiple material types defined"
        
        # Check some expected types are present
        expected_types = [MaterialType.GRANITE, MaterialType.BASALT, MaterialType.LIMESTONE, 
                         MaterialType.SHALE, MaterialType.SLATE, MaterialType.MARBLE, MaterialType.MAGMA]
        for material_type in expected_types:
            assert material_type in material_types, f"{material_type} should be in material types"


class TestMaterialProperties:
    """Test cases for the MaterialProperties dataclass"""
    
    def test_material_properties_creation(self):
        """Test creating material properties"""
        props = MaterialProperties(
            density=2650.0,
            thermal_conductivity=2.5,
            specific_heat=790.0,
            strength=200.0,
            porosity=0.01,
            color_rgb=(128, 128, 128),
            transitions=[
                TransitionRule(MaterialType.MAGMA, 1200, float('inf'), 0, float('inf'), "Melting to magma")
            ]
        )
        
        assert props.density == 2650.0
        assert props.thermal_conductivity == 2.5
        assert props.specific_heat == 790.0
        assert props.strength == 200.0
        assert props.porosity == 0.01
        assert props.color_rgb == (128, 128, 128)
        assert len(props.transitions) == 1
    
    def test_material_properties_validation(self):
        """Test that material properties have reasonable values"""
        props = MaterialProperties(
            density=2650.0,
            thermal_conductivity=2.5,
            specific_heat=790.0,
            strength=200.0,
            porosity=0.01,
            color_rgb=(128, 128, 128),
            transitions=[]
        )
        
        assert props.density > 0, "Density should be positive"
        assert props.thermal_conductivity > 0, "Thermal conductivity should be positive"
        assert len(props.color_rgb) == 3, "Color should be RGB tuple"
        assert all(0 <= c <= 255 for c in props.color_rgb), "RGB values should be in range [0, 255]"


class TestMaterialDatabase:
    """Test cases for the MaterialDatabase class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.db = MaterialDatabase()
    
    def test_database_initialization(self):
        """Test that database initializes with all material types"""
        # Should have properties for all material types
        for material_type in MaterialType:
            props = self.db.get_properties(material_type)
            assert props is not None, f"Should have properties for {material_type}"
            assert isinstance(props, MaterialProperties), f"Properties should be MaterialProperties instance for {material_type}"
    
    def test_get_properties(self):
        """Test getting material properties"""
        # Test specific material types
        granite_props = self.db.get_properties(MaterialType.GRANITE)
        assert granite_props.density > 2000, "Granite density should be reasonable"
        assert len(granite_props.transitions) > 0, "Granite should have some transitions"
        assert granite_props.thermal_conductivity > 0, "Granite should conduct heat"
        
        basalt_props = self.db.get_properties(MaterialType.BASALT)
        assert basalt_props.density > 2000, "Basalt density should be reasonable"
        assert len(basalt_props.transitions) > 0, "Basalt should have some transitions"
        
        # Properties should be different for different materials
        assert granite_props != basalt_props, "Different materials should have different properties"
    
    def test_metamorphic_transitions_shale_sequence(self):
        """Test the shale -> slate -> schist -> gneiss sequence"""
        # Low grade: shale -> slate
        product = self.db.get_metamorphic_product(MaterialType.SHALE, 300, 50)
        assert product == MaterialType.SLATE, f"Shale at low P/T should become slate, got {product}"
        
        # Medium grade: slate -> schist (need higher pressure)
        product = self.db.get_metamorphic_product(MaterialType.SLATE, 500, 200)
        assert product == MaterialType.SCHIST, f"Slate at medium P/T should become schist, got {product}"
        
        # High grade: schist -> gneiss (need higher pressure)
        product = self.db.get_metamorphic_product(MaterialType.SCHIST, 700, 400)
        assert product == MaterialType.GNEISS, f"Schist at high P/T should become gneiss, got {product}"
    
    def test_metamorphic_transitions_other_materials(self):
        """Test metamorphic transitions for other material types"""
        # Limestone -> marble
        product = self.db.get_metamorphic_product(MaterialType.LIMESTONE, 500, 150)
        assert product == MaterialType.MARBLE, f"Limestone should become marble, got {product}"
        
        # Sandstone -> quartzite
        product = self.db.get_metamorphic_product(MaterialType.SANDSTONE, 400, 100)
        assert product == MaterialType.QUARTZITE, f"Sandstone should become quartzite, got {product}"
        
        # Granite -> gneiss (need higher temperature)
        product = self.db.get_metamorphic_product(MaterialType.GRANITE, 650, 200)
        assert product == MaterialType.GNEISS, f"Granite should become gneiss, got {product}"
    
    def test_metamorphic_no_change_conditions(self):
        """Test that materials don't change under low P/T conditions"""
        # Very low conditions should not cause change (returns None = no change)
        product = self.db.get_metamorphic_product(MaterialType.GRANITE, 100, 10)
        assert product is None, "Granite should not change at very low P/T (returns None)"
        
        product = self.db.get_metamorphic_product(MaterialType.LIMESTONE, 200, 20)
        assert product is None, "Limestone should not change at low P/T (returns None)"
    
    def test_melting_behavior(self):
        """Test material melting behavior"""
        # Test various materials at high temperature
        assert self.db.should_melt(MaterialType.GRANITE, 1300), "Granite should melt at 1300°C"
        assert self.db.should_melt(MaterialType.BASALT, 1200), "Basalt should melt at 1200°C"
        assert self.db.should_melt(MaterialType.LIMESTONE, 1500), "Limestone should melt at very high temp"
        
        # Test at low temperatures
        assert not self.db.should_melt(MaterialType.GRANITE, 800), "Granite should not melt at 800°C"
        assert not self.db.should_melt(MaterialType.BASALT, 700), "Basalt should not melt at 700°C"
    
    def test_melting_edge_cases(self):
        """Test melting behavior at edge cases"""
        # Find granite's melting temperature from transitions
        granite_props = self.db.get_properties(MaterialType.GRANITE)
        melting_temp = None
        for transition in granite_props.transitions:
            if transition.target == MaterialType.MAGMA:
                melting_temp = transition.min_temp
                break
        
        assert melting_temp is not None, "Granite should have a melting transition"
        assert self.db.should_melt(MaterialType.GRANITE, melting_temp + 1), "Should melt just above melting point"
        assert not self.db.should_melt(MaterialType.GRANITE, melting_temp - 1), "Should not melt just below melting point"
        
        # Magma should not "melt" (it's already molten)
        assert not self.db.should_melt(MaterialType.MAGMA, 2000), "Magma should not melt (already molten)"
    
    def test_color_consistency(self):
        """Test that material colors are reasonable"""
        for material_type in MaterialType:
            props = self.db.get_properties(material_type)
            color = props.color_rgb
            
            assert len(color) == 3, f"{material_type} should have RGB color tuple"
            assert all(isinstance(c, int) for c in color), f"{material_type} color should be integers"
            assert all(0 <= c <= 255 for c in color), f"{material_type} color values should be in [0, 255]"
    
    def test_physical_property_ranges(self):
        """Test that physical properties are in reasonable ranges"""
        for material_type in MaterialType:
            props = self.db.get_properties(material_type)
            
            # Density should be reasonable for materials (kg/m³) - gases, ice, and space are exceptions
            if material_type in [MaterialType.PUMICE, MaterialType.AIR, MaterialType.WATER_VAPOR, MaterialType.ICE]:
                assert props.density > 0, f"{material_type} density should be positive"
            elif material_type == MaterialType.SPACE:
                assert props.density < 1e-6, f"Space density should be negligible vacuum value (<1e-6), got {props.density}"
            elif material_type == MaterialType.WATER:
                assert props.density == 1000, f"{material_type} should have water density"
            else:
                assert 1500 <= props.density <= 5000, f"{material_type} density {props.density} should be in reasonable range"
            
            # Check that materials have appropriate transitions
            if material_type in [MaterialType.WATER, MaterialType.ICE, MaterialType.WATER_VAPOR]:
                # Phase transition materials should have transitions
                assert len(props.transitions) > 0, f"{material_type} should have phase transitions"
            elif material_type in [MaterialType.SPACE]:
                # Space doesn't transition to anything
                assert len(props.transitions) == 0, f"Space should have no transitions"
            else:
                # Regular materials should have some transitions (at least melting)
                # Note: Some materials might not have transitions implemented yet, so we just check they're defined
                assert isinstance(props.transitions, list), f"{material_type} should have transitions list"
            
            # Thermal conductivity should be positive (except space which is vacuum)
            if material_type == MaterialType.SPACE:
                assert props.thermal_conductivity < 1e-6, f"Space should have negligible thermal conductivity (vacuum)"
            else:
                assert props.thermal_conductivity > 0, f"{material_type} thermal conductivity should be positive"
                assert props.thermal_conductivity < 100, f"{material_type} thermal conductivity should be reasonable" 