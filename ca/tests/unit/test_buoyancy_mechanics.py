"""Test buoyancy mechanics - validate basic rigid body-fluid interactions"""

import numpy as np
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


def test_buoyancy_mechanics_basics():
    """Test that the basic mechanics work: rigid bodies can displace fluids"""
    print("=== Testing Basic Displacement Mechanics ===")
    
    sim = GeoSimulation(width=15, height=15, cell_size=100)
    sim.external_gravity = (0, 10)
    sim.enable_self_gravity = False
    sim.enable_solid_drag = True
    sim.debug_rigid_bodies = True
    
    # Disable thermal processes 
    sim.enable_internal_heating = False
    sim.enable_solar_heating = False
    sim.enable_radiative_cooling = False
    sim.enable_heat_diffusion = False
    sim.enable_material_processes = False
    sim.enable_atmospheric_processes = False
    if hasattr(sim, 'heat_transfer'):
        sim.heat_transfer.enabled = False
    
    # Setup: space, water, and granite
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 280.0
    
    # Water layer
    for y in range(10, 15):
        for x in range(15):
            sim.material_types[y, x] = MaterialType.WATER
            sim.temperature[y, x] = 275.0
    
    # Granite above water
    sim.material_types[8, 7] = MaterialType.GRANITE
    sim.temperature[8, 7] = 300.0
    
    sim._update_material_properties()
    
    print("Initial setup:")
    print("- Granite at (8, 7) - above water")
    print("- Water from Y=10 to Y=14")
    
    # Run a few steps and track behavior
    initial_water_count = np.sum(sim.material_types == MaterialType.WATER)
    
    for step in range(15):
        granite_mask = sim.material_types == MaterialType.GRANITE
        water_mask = sim.material_types == MaterialType.WATER
        
        if not np.any(granite_mask):
            print(f"Step {step}: Granite disappeared!")
            return False
        
        granite_pos = np.argwhere(granite_mask)[0]
        water_count = np.sum(water_mask)
        
        print(f"Step {step}: Granite at {granite_pos}, Water cells: {water_count}")
        
        sim.step_forward(1.0)
    
    # Final verification
    final_water_count = np.sum(sim.material_types == MaterialType.WATER)
    granite_mask = sim.material_types == MaterialType.GRANITE
    
    print(f"\nResults:")
    print(f"Initial water: {initial_water_count} cells")
    print(f"Final water: {final_water_count} cells")
    print(f"Water conserved: {abs(final_water_count - initial_water_count) <= 1}")
    
    if np.any(granite_mask):
        final_granite_pos = np.argwhere(granite_mask)[0]
        print(f"Final granite position: {final_granite_pos}")
        
        # Granite should have moved into water area and displaced it
        if final_granite_pos[0] >= 10:
            print("✓ Granite entered water and displaced fluid")
            return True
        else:
            print("✗ Granite didn't enter water")
            return False
    else:
        print("✗ Granite was lost")
        return False


def test_density_based_forces():
    """Test that denser materials experience stronger forces"""
    print("\n=== Testing Density-Based Forces ===")
    
    sim = GeoSimulation(width=10, height=15, cell_size=100)
    sim.external_gravity = (0, 10) 
    sim.enable_self_gravity = False
    sim.enable_solid_drag = False  # No damping for clear force comparison
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
    
    # Setup in space
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 280.0
    
    # Place ice and granite side by side
    sim.material_types[5, 3] = MaterialType.ICE
    sim.material_types[5, 6] = MaterialType.GRANITE
    sim.temperature[5, 3] = 270.0
    sim.temperature[5, 6] = 300.0
    
    sim._update_material_properties()
    
    # Get densities
    ice_density = sim.material_db.get_properties(MaterialType.ICE).density
    granite_density = sim.material_db.get_properties(MaterialType.GRANITE).density
    
    print(f"Ice density: {ice_density} kg/m³")
    print(f"Granite density: {granite_density} kg/m³")
    print(f"Granite/Ice ratio: {granite_density/ice_density:.2f}")
    
    # Track velocities after a few steps
    for step in range(5):
        sim.step_forward(1.0)
    
    ice_mask = sim.material_types == MaterialType.ICE
    granite_mask = sim.material_types == MaterialType.GRANITE
    
    if np.any(ice_mask) and np.any(granite_mask):
        ice_pos = np.argwhere(ice_mask)[0]
        granite_pos = np.argwhere(granite_mask)[0]
        
        ice_vy = sim.fluid_dynamics.velocity_y[ice_pos[0], ice_pos[1]]
        granite_vy = sim.fluid_dynamics.velocity_y[granite_pos[0], granite_pos[1]]
        
        print(f"After 5 steps:")
        print(f"Ice velocity: {ice_vy:.2f} m/s")
        print(f"Granite velocity: {granite_vy:.2f} m/s")
        print(f"Velocity ratio: {granite_vy/ice_vy:.2f}")
        
        # Both should be falling, granite should be faster due to higher density
        if granite_vy > ice_vy > 0:
            print("✓ Granite falls faster than ice (correct density effect)")
            return True
        else:
            print("✗ Velocity relationship incorrect")
            return False
    else:
        print("✗ Materials disappeared")
        return False


def test_fluid_displacement_conservation():
    """Test that fluid displacement conserves mass"""
    print("\n=== Testing Fluid Displacement Conservation ===")
    
    sim = GeoSimulation(width=10, height=10, cell_size=100)
    sim.external_gravity = (0, 5)
    sim.enable_self_gravity = False
    sim.enable_solid_drag = True
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
    
    # Create a water "pool" with space around it
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 280.0
    
    # 3x3 water pool in center
    for y in range(4, 7):
        for x in range(4, 7):
            sim.material_types[y, x] = MaterialType.WATER
            sim.temperature[y, x] = 275.0
    
    # Drop granite into center
    sim.material_types[2, 5] = MaterialType.GRANITE
    sim.temperature[2, 5] = 300.0
    
    sim._update_material_properties()
    
    # Count initial materials
    initial_water = np.sum(sim.material_types == MaterialType.WATER)
    initial_granite = np.sum(sim.material_types == MaterialType.GRANITE)
    initial_space = np.sum(sim.material_types == MaterialType.SPACE)
    
    print(f"Initial: Water={initial_water}, Granite={initial_granite}, Space={initial_space}")
    
    # Run simulation
    for step in range(20):
        sim.step_forward(1.0)
    
    # Count final materials
    final_water = np.sum(sim.material_types == MaterialType.WATER)
    final_granite = np.sum(sim.material_types == MaterialType.GRANITE)
    final_space = np.sum(sim.material_types == MaterialType.SPACE)
    
    print(f"Final: Water={final_water}, Granite={final_granite}, Space={final_space}")
    
    # Check conservation
    water_conserved = abs(final_water - initial_water) <= 1
    granite_conserved = abs(final_granite - initial_granite) == 0
    total_conserved = (final_water + final_granite + final_space) == (initial_water + initial_granite + initial_space)
    
    print(f"Water conserved: {water_conserved}")
    print(f"Granite conserved: {granite_conserved}")
    print(f"Total mass conserved: {total_conserved}")
    
    if water_conserved and granite_conserved and total_conserved:
        print("✓ Mass conservation during displacement")
        return True
    else:
        print("✗ Mass not conserved during displacement")
        return False


if __name__ == "__main__":
    results = []
    
    results.append(test_buoyancy_mechanics_basics())
    results.append(test_density_based_forces())
    results.append(test_fluid_displacement_conservation())
    
    print("\n" + "="*60)
    print("SUMMARY:")
    if all(results):
        print("✓ All basic buoyancy mechanics tests passed!")
        print("The rigid body displacement system is working correctly.")
        print("Next step: Fine-tune buoyancy force calculations for proper floating.")
    else:
        print("✗ Some basic mechanics tests failed:")
        tests = ["Basic Displacement", "Density Forces", "Mass Conservation"]
        for i, result in enumerate(results):
            status = "✓" if result else "✗"
            print(f"  {status} {tests[i]}")
    
    print(f"\nPassed: {sum(results)}/{len(results)} tests")