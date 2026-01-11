"""Unit tests for material properties and database"""

import numpy as np
import pytest
from materials import MaterialType, MaterialProperties, MaterialDatabase


def test_material_type_enum():
    """Test MaterialType enum completeness"""
    # Check that all expected materials exist
    expected_materials = [
        'SPACE', 'AIR', 'WATER', 'WATER_VAPOR', 'ICE',
        'MAGMA', 'GRANITE', 'BASALT', 'SCHIST', 'GNEISS', 'SANDSTONE'
    ]
    
    for material_name in expected_materials:
        assert hasattr(MaterialType, material_name), f"Missing material: {material_name}"
        material = getattr(MaterialType, material_name)
        assert isinstance(material, MaterialType), f"{material_name} should be MaterialType"
    
    # Check that materials have unique values
    values = [material.value for material in MaterialType]
    assert len(values) == len(set(values)), "Material types should have unique values"
    
    print(f"MaterialType enum test passed")
    print(f"  Found {len(MaterialType)} material types")


def test_material_properties():
    """Test MaterialProperties data structure"""
    # Create sample properties
    props = MaterialProperties(
        density=1000.0,
        thermal_conductivity=0.6,
        specific_heat=4186.0,
        color_rgb=(100, 100, 100),
        emissivity=0.9,
        thermal_expansion=1e-5,
    )
    
    # Check all properties are accessible
    assert props.density == 1000.0
    assert props.thermal_conductivity == 0.6
    assert props.specific_heat == 4186.0
    assert props.color_rgb == (100, 100, 100)
    assert props.emissivity == 0.9
    assert props.thermal_expansion == 1e-5
    
    print(f"MaterialProperties test passed")


def test_material_database():
    """Test MaterialDatabase completeness and consistency"""
    db = MaterialDatabase()
    
    # Test that all MaterialType entries have properties
    for material_type in MaterialType:
        props = db.get_properties(material_type)
        assert props is not None, f"No properties for {material_type}"
        assert isinstance(props, MaterialProperties), f"Invalid properties type for {material_type}"
        
        # Check that properties are physically reasonable
        assert props.density > 0, f"Density must be positive for {material_type}"
        assert props.thermal_conductivity > 0, f"Thermal conductivity must be positive for {material_type}"
        assert props.specific_heat > 0, f"Specific heat must be positive for {material_type}"
        
        # Check color is valid RGB
        assert len(props.color_rgb) == 3, f"Color must be RGB tuple for {material_type}"
        assert all(0 <= c <= 255 for c in props.color_rgb), f"RGB values must be 0-255 for {material_type}"
        
        # Check emissivity is valid
        assert 0 <= props.emissivity <= 1, f"Emissivity must be between 0 and 1 for {material_type}"
    
    print(f"MaterialDatabase completeness test passed")
    print(f"  Verified properties for {len(MaterialType)} materials")


def test_density_relationships():
    """Test that material densities have correct relationships"""
    db = MaterialDatabase()
    
    # Get densities for comparison
    space_density = db.get_properties(MaterialType.SPACE).density
    air_density = db.get_properties(MaterialType.AIR).density
    water_density = db.get_properties(MaterialType.WATER).density
    ice_density = db.get_properties(MaterialType.ICE).density
    granite_density = db.get_properties(MaterialType.GRANITE).density
    basalt_density = db.get_properties(MaterialType.BASALT).density
    
    # Test expected density relationships
    assert space_density < air_density, "Space should be less dense than air"
    assert air_density < water_density, "Air should be less dense than water"
    assert ice_density < water_density, "Ice should be less dense than water (buoyancy)"
    assert water_density < granite_density, "Water should be less dense than granite"
    assert granite_density > 2000, "Granite should have realistic rock density"
    assert basalt_density > granite_density, "Basalt should be denser than granite"
    
    # Check space is very low density
    assert space_density < 1e-6, "Space should have very low density"
    
    print(f"Density relationships test passed")
    print(f"  Space: {space_density:.2e} kg/m³")
    print(f"  Air: {air_density:.1f} kg/m³")
    print(f"  Ice: {ice_density:.1f} kg/m³")
    print(f"  Water: {water_density:.1f} kg/m³")
    print(f"  Granite: {granite_density:.1f} kg/m³")
    print(f"  Basalt: {basalt_density:.1f} kg/m³")


def test_thermal_properties():
    """Test thermal property relationships"""
    db = MaterialDatabase()
    
    # Get thermal properties
    water_props = db.get_properties(MaterialType.WATER)
    granite_props = db.get_properties(MaterialType.GRANITE)
    air_props = db.get_properties(MaterialType.AIR)
    
    # Water should have high specific heat
    assert water_props.specific_heat > 3000, "Water should have high specific heat"
    
    # Rock should have lower specific heat than water
    assert granite_props.specific_heat < water_props.specific_heat, "Rock should have lower specific heat than water"
    
    # Air should have lower thermal conductivity than water
    assert air_props.thermal_conductivity < water_props.thermal_conductivity, "Air should have lower thermal conductivity"
    
    print(f"Thermal properties test passed")
    print(f"  Water specific heat: {water_props.specific_heat:.0f} J/kg·K")
    print(f"  Granite specific heat: {granite_props.specific_heat:.0f} J/kg·K")
    print(f"  Water thermal conductivity: {water_props.thermal_conductivity:.2f} W/m·K")
    print(f"  Air thermal conductivity: {air_props.thermal_conductivity:.4f} W/m·K")


def test_phase_transition_temperatures():
    """Test that materials have proper transition rules"""
    db = MaterialDatabase()
    
    # Water phase transitions
    water_props = db.get_properties(MaterialType.WATER)
    ice_props = db.get_properties(MaterialType.ICE)
    
    # Check that water and ice have transition rules
    assert len(water_props.transitions) > 0, "Water should have transition rules"
    assert len(ice_props.transitions) > 0, "Ice should have transition rules"
    
    # Check ice->water transition exists
    ice_to_water_found = False
    for transition in ice_props.transitions:
        if transition.target == MaterialType.WATER:
            ice_to_water_found = True
            # Check temperature range is reasonable (around 0°C)
            assert -10 < transition.min_temp < 10, f"Ice->water transition should happen near 0°C"
            
    assert ice_to_water_found, "Ice should have transition to water"
    
    # Check water->ice transition exists
    water_to_ice_found = False
    for transition in water_props.transitions:
        if transition.target == MaterialType.ICE:
            water_to_ice_found = True
            
    assert water_to_ice_found, "Water should have transition to ice"
    
    print(f"Phase transition test passed")
    print(f"  Water has {len(water_props.transitions)} transitions")
    print(f"  Ice has {len(ice_props.transitions)} transitions")


if __name__ == "__main__":
    test_material_type_enum()
    test_material_properties()
    test_material_database()
    test_density_relationships()
    test_thermal_properties()
    test_phase_transition_temperatures()
    print("\nAll material unit tests passed!")