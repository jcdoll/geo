"""Diagnose pressure field and buoyancy force calculation"""

import numpy as np
import pytest
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


@pytest.mark.skip(reason="CA simulation does not have pressure field - this diagnostic test is for future flux-based implementation")
def test_pressure_field_diagnosis():
    """Analyze the pressure field and forces in detail"""
    
    print("=== PRESSURE FIELD DIAGNOSIS ===")
    
    # Simple setup for analysis
    sim = GeoSimulation(width=15, height=20, cell_size=100)
    sim.external_gravity = (0, 10)  # 10 m/s² downward
    sim.enable_self_gravity = False  # Only external gravity
    sim.enable_solid_drag = False   # No damping for clear analysis
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
    
    # Simple layered setup
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 280.0
    
    # Water layer at bottom
    water_start = 10
    for y in range(water_start, sim.height):
        for x in range(sim.width):
            sim.material_types[y, x] = MaterialType.WATER
            sim.temperature[y, x] = 275.0
    
    # Single ice cell above water
    ice_y, ice_x = 8, 7
    sim.material_types[ice_y, ice_x] = MaterialType.ICE
    sim.temperature[ice_y, ice_x] = 270.0
    
    sim._update_material_properties()
    
    # Get material properties
    ice_density = sim.material_db.get_properties(MaterialType.ICE).density
    water_density = sim.material_db.get_properties(MaterialType.WATER).density
    space_density = sim.material_db.get_properties(MaterialType.SPACE).density
    
    print(f"Material densities:")
    print(f"  Space: {space_density} kg/m³")
    print(f"  Ice: {ice_density} kg/m³") 
    print(f"  Water: {water_density} kg/m³")
    print(f"  Expected buoyancy: Ice should float (ρ_ice < ρ_water)")
    
    # Check pressure calculation settings
    print(f"\nSimulation settings:")
    print(f"  Enable self-gravity: {sim.enable_self_gravity}")
    print(f"  External gravity: {getattr(sim, 'external_gravity', 'None')}")
    
    # Before any steps - check initial pressure field
    print(f"\nInitial pressure field:")
    pressure_range = f"[{sim.pressure.min():.3f}, {sim.pressure.max():.3f}] MPa"
    print(f"  Pressure range: {pressure_range}")
    
    # Take one step and analyze forces
    print(f"\n=== AFTER ONE SIMULATION STEP ===")
    sim.step_forward(1.0)
    
    # Check pressure field after step
    print(f"Pressure field after step:")
    pressure_range = f"[{sim.pressure.min():.3f}, {sim.pressure.max():.3f}] MPa"
    print(f"  Pressure range: {pressure_range}")
    
    # Analyze pressure profile in Y direction (middle column)
    mid_x = sim.width // 2
    print(f"\nPressure profile (column X={mid_x}):")
    for y in range(0, sim.height, 2):  # Every 2nd row
        material = sim.material_types[y, mid_x]
        pressure = sim.pressure[y, mid_x]
        density = sim.density[y, mid_x]
        mat_name = material.value if hasattr(material, 'value') else str(material)
        print(f"  Y={y:2d}: {mat_name:8s} ρ={density:4.0f} P={pressure:8.3f} MPa")
    
    # Check force calculations
    fx, fy = sim.fluid_dynamics.compute_force_field()
    
    print(f"\nForce analysis at ice position ({ice_y}, {ice_x}):")
    
    # Find current ice position
    ice_mask = sim.material_types == MaterialType.ICE
    if np.any(ice_mask):
        ice_coords = np.argwhere(ice_mask)[0]
        iy, ix = ice_coords[0], ice_coords[1]
        
        # Forces at ice location
        ice_fx = fx[iy, ix]
        ice_fy = fy[iy, ix]
        ice_density_now = sim.density[iy, ix]
        
        print(f"  Ice now at: ({iy}, {ix})")
        print(f"  Force density: fx={ice_fx:.2e} N/m³, fy={ice_fy:.2e} N/m³")
        print(f"  Ice density: {ice_density_now:.1f} kg/m³")
        
        # Calculate expected gravitational force
        expected_gravity_force = ice_density_now * 10  # ρ * g
        print(f"  Expected gravity force: {expected_gravity_force:.2e} N/m³ (downward)")
        
        # Calculate pressure forces around ice
        if iy > 0 and iy < sim.height-1:
            pressure_above = sim.pressure[iy-1, ix] * 1e6  # Convert to Pa
            pressure_below = sim.pressure[iy+1, ix] * 1e6
            pressure_gradient = (pressure_below - pressure_above) / (2 * sim.cell_size)
            
            print(f"  Pressure above: {pressure_above:.2e} Pa")
            print(f"  Pressure below: {pressure_below:.2e} Pa")
            print(f"  Pressure gradient: {pressure_gradient:.2e} Pa/m")
            print(f"  Pressure force: {-pressure_gradient:.2e} N/m³ (negative gradient = upward force)")
        
        # Compare forces
        net_force = ice_fy
        print(f"  Net force: {net_force:.2e} N/m³")
        
        if net_force > 0:
            print(f"  → Ice is accelerating DOWNWARD (sinking)")
        elif net_force < 0:
            print(f"  → Ice is accelerating UPWARD (floating)")
        else:
            print(f"  → Ice is in equilibrium")
        
        # Expected buoyancy calculation
        expected_buoyant_force = water_density * 10  # Force from displaced water
        expected_net = expected_buoyant_force - expected_gravity_force
        print(f"  Expected buoyant force: {expected_buoyant_force:.2e} N/m³")
        print(f"  Expected net force: {expected_net:.2e} N/m³ (should be upward for ice)")
        
        # Check if forces match expectations
        if abs(net_force - expected_net) < 1000:
            print(f"  ✓ Forces match physical expectations")
        else:
            print(f"  ✗ Forces don't match! Actual: {net_force:.2e}, Expected: {expected_net:.2e}")
    
    else:
        print(f"  Ice not found after step!")
    
    # Check if pressure calculation includes external gravity
    print(f"\n=== PRESSURE CALCULATION ANALYSIS ===")
    
    # The pressure should vary with depth due to external gravity
    # Check if there's a pressure gradient in the water column
    water_column_x = sim.width // 2
    water_pressures = []
    water_depths = []
    
    for y in range(water_start, sim.height):
        if sim.material_types[y, water_column_x] == MaterialType.WATER:
            water_pressures.append(sim.pressure[y, water_column_x])
            water_depths.append(y - water_start)
    
    if len(water_pressures) > 1:
        pressure_increase = water_pressures[-1] - water_pressures[0]
        depth_span = water_depths[-1] - water_depths[0]
        
        print(f"Water column pressure analysis:")
        print(f"  Depth span: {depth_span} cells ({depth_span * sim.cell_size:.0f} m)")
        print(f"  Pressure at surface: {water_pressures[0]:.6f} MPa")
        print(f"  Pressure at bottom: {water_pressures[-1]:.6f} MPa")
        print(f"  Pressure increase: {pressure_increase:.6f} MPa")
        
        # Expected pressure increase due to water column
        expected_increase = water_density * 10 * (depth_span * sim.cell_size) / 1e6  # Convert to MPa
        print(f"  Expected increase: {expected_increase:.6f} MPa (ρgh)")
        
        if abs(pressure_increase - expected_increase) < 0.001:
            print(f"  ✓ Hydrostatic pressure looks correct")
        else:
            print(f"  ✗ Hydrostatic pressure incorrect! Missing external gravity in pressure calc?")
    
    return sim


if __name__ == "__main__":
    sim = test_pressure_field_diagnosis()
    
    print(f"\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("Check the output above to identify pressure/buoyancy calculation issues.")