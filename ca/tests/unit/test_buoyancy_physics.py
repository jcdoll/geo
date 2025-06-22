"""Unit tests for buoyancy physics calculations"""

import numpy as np
import pytest
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


def test_density_differences():
    """Test that materials have correct density relationships for buoyancy"""
    sim = GeoSimulation(width=10, height=10, cell_size=100)
    
    # Get material densities
    ice_density = sim.material_db.get_properties(MaterialType.ICE).density
    water_density = sim.material_db.get_properties(MaterialType.WATER).density
    granite_density = sim.material_db.get_properties(MaterialType.GRANITE).density
    space_density = sim.material_db.get_properties(MaterialType.SPACE).density
    
    # Test density relationships for buoyancy
    assert ice_density < water_density, f"Ice ({ice_density}) should be less dense than water ({water_density})"
    assert granite_density > water_density, f"Granite ({granite_density}) should be denser than water ({water_density})"
    assert space_density < ice_density, f"Space ({space_density}) should be less dense than ice ({ice_density})"
    
    print(f"Density test passed:")
    print(f"  Space: {space_density} kg/m³")
    print(f"  Ice: {ice_density} kg/m³") 
    print(f"  Water: {water_density} kg/m³")
    print(f"  Granite: {granite_density} kg/m³")


def test_pressure_field_setup():
    """Test that external gravity creates proper pressure field setup"""
    sim = GeoSimulation(width=5, height=10, cell_size=100)
    sim.external_gravity = (0, 10)  # 10 m/s² downward
    sim.enable_self_gravity = False
    
    # Create layered density structure
    sim.material_types[:] = MaterialType.SPACE  # Light material
    sim.material_types[5:, :] = MaterialType.WATER  # Heavy material below
    sim._update_material_properties()
    
    # Check that density field is set up correctly
    space_density = sim.density[0, 0]
    water_density = sim.density[8, 0]
    
    assert space_density < water_density, "Density stratification should be correct"
    assert water_density > 900, "Water density should be realistic"
    assert space_density < 1e-6, "Space density should be very low"
    
    print(f"Pressure field setup test passed")
    print(f"  Space density: {space_density} kg/m³")
    print(f"  Water density: {water_density} kg/m³")


def test_force_calculation():
    """Test that force calculations include both gravity and pressure terms"""
    sim = GeoSimulation(width=5, height=10, cell_size=100)
    sim.external_gravity = (0, 10)
    sim.enable_self_gravity = False
    
    # Simple setup for force calculation
    sim.material_types[:] = MaterialType.WATER
    sim.material_types[0, 0] = MaterialType.ICE  # One ice cell in water
    sim._update_material_properties()
    
    # Step forward to calculate forces
    sim.step_forward(1.0)
    
    # Check that forces were computed
    fx, fy = sim.fluid_dynamics.compute_force_field()
    
    # Ice should experience some force (gravity + pressure)
    ice_fx = fx[0, 0]
    ice_fy = fy[0, 0]
    
    # Forces should be non-zero
    assert abs(ice_fx) > 0 or abs(ice_fy) > 0, "Forces should be computed for ice in water"
    
    # Y-force should be significant (gravity + buoyancy)
    assert abs(ice_fy) > 1000, "Vertical force should be significant"
    
    print(f"Force calculation test passed")
    print(f"  Ice force: fx={ice_fx:.2e} N/m³, fy={ice_fy:.2e} N/m³")


def test_buoyancy_direction():
    """Test that buoyancy forces are in the correct direction"""
    sim = GeoSimulation(width=5, height=10, cell_size=100)
    sim.external_gravity = (0, 10)  # Downward gravity
    sim.enable_self_gravity = False
    
    # Disable thermal processes for clean test
    sim.enable_internal_heating = False
    sim.enable_solar_heating = False
    sim.enable_radiative_cooling = False
    sim.enable_heat_diffusion = False
    sim.enable_material_processes = False
    sim.enable_atmospheric_processes = False
    if hasattr(sim, 'heat_transfer'):
        sim.heat_transfer.enabled = False
    
    # Create ice in water scenario
    sim.material_types[:] = MaterialType.WATER
    sim.material_types[2, 2] = MaterialType.ICE  # Ice in middle of water
    sim.temperature[:] = 275.0
    sim.temperature[2, 2] = 270.0
    sim._update_material_properties()
    
    # Get densities
    ice_density = sim.density[2, 2]
    water_density = sim.material_db.get_properties(MaterialType.WATER).density
    
    # Calculate expected buoyancy force direction
    expected_buoyancy = (water_density - ice_density) * 10  # Should be upward (+)
    
    # Step forward to calculate pressure and forces  
    sim.step_forward(1.0)
    fx, fy = sim.fluid_dynamics.compute_force_field()
    
    ice_force_y = fy[2, 2]
    gravity_force = ice_density * 10  # Downward (-)
    
    print(f"Buoyancy direction test:")
    print(f"  Ice density: {ice_density:.1f} kg/m³")
    print(f"  Water density: {water_density:.1f} kg/m³")
    print(f"  Expected buoyancy force: {expected_buoyancy:.1f} N/m³ (upward)")
    print(f"  Gravity force on ice: {gravity_force:.1f} N/m³ (downward)")
    print(f"  Actual net force on ice: {ice_force_y:.1f} N/m³")
    
    # The net force should be less downward than pure gravity
    # (buoyancy partially counteracts gravity)
    if expected_buoyancy > gravity_force:
        # Ice should float (net upward force)
        print(f"  Expected: Net upward force (ice should float)")
    else:
        # Ice should sink but with reduced downward force
        print(f"  Expected: Reduced downward force (partial buoyancy)")
        assert abs(ice_force_y) < gravity_force, "Buoyancy should reduce downward force"
    
    return True


if __name__ == "__main__":
    test_density_differences()
    test_pressure_field_setup()
    test_force_calculation()
    test_buoyancy_direction()
    print("\nAll buoyancy physics unit tests passed!")