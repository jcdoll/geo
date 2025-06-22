"""Detailed acceleration analysis for ice vs granite in water"""

import numpy as np
import matplotlib.pyplot as plt
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


def test_detailed_motion_analysis():
    """Analyze position, velocity, and acceleration over time for ice vs granite"""
    
    # Create two identical simulations - one for ice, one for granite
    results = {}
    
    for material_name, material_type in [("Ice", MaterialType.ICE), ("Granite", MaterialType.GRANITE)]:
        print(f"\n=== Analyzing {material_name} Motion ===")
        
        sim = GeoSimulation(width=20, height=40, cell_size=100)
        sim.external_gravity = (0, 10)  # 10 m/s² downward
        sim.enable_self_gravity = False
        sim.enable_solid_drag = True
        sim.debug_rigid_bodies = False  # Reduce output
        
        # Disable thermal processes
        sim.enable_internal_heating = False
        sim.enable_solar_heating = False
        sim.enable_radiative_cooling = False
        sim.enable_heat_diffusion = False
        sim.enable_material_processes = False
        sim.enable_atmospheric_processes = False
        if hasattr(sim, 'heat_transfer'):
            sim.heat_transfer.enabled = False
        
        # Setup: space above, deep water below
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 280.0
        
        # Deep water ocean - 20 cells deep
        water_depth = 20
        water_surface_y = sim.height - water_depth  # Y = 20
        water_bottom_y = sim.height - 1  # Y = 39
        
        for y in range(water_surface_y, sim.height):
            for x in range(sim.width):
                sim.material_types[y, x] = MaterialType.WATER
                sim.temperature[y, x] = 275.0
        
        # Place material high above water
        start_y = 5
        start_x = 10
        sim.material_types[start_y, start_x] = material_type
        sim.temperature[start_y, start_x] = 270.0 if material_type == MaterialType.ICE else 300.0
        
        sim._update_material_properties()
        
        # Get material properties
        material_density = sim.material_db.get_properties(material_type).density
        water_density = sim.material_db.get_properties(MaterialType.WATER).density
        space_density = sim.material_db.get_properties(MaterialType.SPACE).density
        
        print(f"{material_name} density: {material_density} kg/m³")
        print(f"Water density: {water_density} kg/m³")
        print(f"Space density: {space_density} kg/m³")
        print(f"Water surface at Y = {water_surface_y}")
        print(f"Water bottom at Y = {water_bottom_y}")
        print(f"Water depth: {water_depth} cells")
        
        # Track motion over time
        times = []
        positions = []
        velocities = []
        accelerations = []
        phases = []  # Track: space, water_surface, water_deep, bottom
        
        dt = 1.0  # 1 second timesteps
        
        for step in range(80):
            # Get current state
            material_mask = sim.material_types == material_type
            if not np.any(material_mask):
                print(f"Step {step}: {material_name} disappeared!")
                break
            
            # Position
            coords = np.argwhere(material_mask)
            y_pos = np.mean(coords[:, 0])
            x_pos = np.mean(coords[:, 1])
            
            # Velocity
            vy = sim.fluid_dynamics.velocity_y[int(y_pos), int(x_pos)]
            vx = sim.fluid_dynamics.velocity_x[int(y_pos), int(x_pos)]
            
            # Determine phase
            if y_pos < water_surface_y - 1:
                phase = "space"
            elif y_pos < water_surface_y + 2:
                phase = "surface"
            elif y_pos > water_bottom_y - 3:
                phase = "bottom"
            else:
                phase = "deep_water"
            
            # Store data
            times.append(step * dt)
            positions.append(y_pos)
            velocities.append(vy)
            phases.append(phase)
            
            # Calculate acceleration from velocity change
            if len(velocities) >= 2:
                accel = (velocities[-1] - velocities[-2]) / dt
                accelerations.append(accel)
            else:
                accelerations.append(0.0)
            
            # Print detailed info every 10 steps
            if step % 10 == 0:
                accel_str = f", a={accelerations[-1]:+6.2f}" if accelerations else ""
                print(f"Step {step:2d}: Y={y_pos:5.1f}, v={vy:6.2f}{accel_str} ({phase})")
            
            sim.step_forward(dt)
        
        # Store results for comparison
        results[material_name] = {
            'times': times,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'phases': phases,
            'density': material_density,
            'water_surface': water_surface_y,
            'water_bottom': water_bottom_y
        }
        
        # Analysis for this material
        print(f"\n{material_name} Motion Analysis:")
        
        # Check if it reached water
        max_y = max(positions)
        min_y = min(positions)
        final_y = positions[-1]
        
        print(f"  Starting Y: {min_y:.1f}")
        print(f"  Deepest Y: {max_y:.1f}")
        print(f"  Final Y: {final_y:.1f}")
        print(f"  Entered water: {'Yes' if max_y > water_surface_y else 'No'}")
        print(f"  Hit bottom: {'Yes' if max_y > water_bottom_y - 2 else 'No'}")
        
        # Acceleration analysis in different phases
        space_accels = [a for i, a in enumerate(accelerations) if i < len(phases) and phases[i] == "space"]
        water_accels = [a for i, a in enumerate(accelerations) if i < len(phases) and phases[i] in ["surface", "deep_water"]]
        
        if space_accels:
            print(f"  Acceleration in space: {np.mean(space_accels):+6.2f} ± {np.std(space_accels):.2f} m/s²")
        if water_accels:
            print(f"  Acceleration in water: {np.mean(water_accels):+6.2f} ± {np.std(water_accels):.2f} m/s²")
        
        # Check for buoyancy (deceleration in water)
        if water_accels:
            avg_water_accel = np.mean(water_accels)
            if material_type == MaterialType.ICE:
                expected_behavior = "negative (buoyant)"
                if avg_water_accel < -1:
                    print(f"  ✓ Shows buoyancy: {avg_water_accel:+.2f} m/s² (upward)")
                elif avg_water_accel < 5:
                    print(f"  ? Reduced sinking: {avg_water_accel:+.2f} m/s² (some buoyancy)")
                else:
                    print(f"  ✗ No buoyancy: {avg_water_accel:+.2f} m/s² (still sinking fast)")
            else:  # Granite
                expected_behavior = "positive (sinking)"
                if avg_water_accel > 1:
                    print(f"  ✓ Sinks as expected: {avg_water_accel:+.2f} m/s²")
                else:
                    print(f"  ? Slow sinking: {avg_water_accel:+.2f} m/s²")
    
    # Comparison between materials
    print(f"\n" + "="*60)
    print("COMPARISON ANALYSIS:")
    
    if "Ice" in results and "Granite" in results:
        ice_data = results["Ice"]
        granite_data = results["Granite"]
        
        # Compare final positions
        ice_final = ice_data['positions'][-1]
        granite_final = granite_data['positions'][-1]
        water_surface = ice_data['water_surface']
        
        print(f"Final positions:")
        print(f"  Ice: Y={ice_final:.1f} (distance from surface: {ice_final - water_surface:+.1f})")
        print(f"  Granite: Y={granite_final:.1f} (distance from surface: {granite_final - water_surface:+.1f})")
        
        # Compare accelerations in water
        ice_water_accels = [a for i, a in enumerate(ice_data['accelerations']) 
                           if i < len(ice_data['phases']) and ice_data['phases'][i] in ["surface", "deep_water"]]
        granite_water_accels = [a for i, a in enumerate(granite_data['accelerations']) 
                              if i < len(granite_data['phases']) and granite_data['phases'][i] in ["surface", "deep_water"]]
        
        if ice_water_accels and granite_water_accels:
            ice_avg_accel = np.mean(ice_water_accels)
            granite_avg_accel = np.mean(granite_water_accels)
            
            print(f"Average accelerations in water:")
            print(f"  Ice: {ice_avg_accel:+6.2f} m/s²")
            print(f"  Granite: {granite_avg_accel:+6.2f} m/s²")
            
            # Physics check
            if ice_avg_accel < granite_avg_accel:
                print(f"  ✓ Ice accelerates less than granite (buoyancy effect)")
            else:
                print(f"  ✗ No clear buoyancy difference")
        
        # Expected vs actual behavior
        print(f"\nPhysics expectations:")
        print(f"  Ice density: {ice_data['density']} < Water: {water_density} → Should float")
        print(f"  Granite density: {granite_data['density']} > Water: {water_density} → Should sink")
        
        # Success criteria
        ice_floats = ice_final < granite_final  # Ice should be higher than granite
        granite_sinks = granite_final > water_surface + 5  # Granite should be deep
        
        print(f"\nSuccess check:")
        print(f"  Ice floats relative to granite: {'✓' if ice_floats else '✗'}")
        print(f"  Granite sinks deep: {'✓' if granite_sinks else '✗'}")
        
        return ice_floats and granite_sinks
    
    return False


if __name__ == "__main__":
    success = test_detailed_motion_analysis()
    
    print(f"\n" + "="*60)
    if success:
        print("✓ BUOYANCY TEST PASSED: Ice and granite show different behaviors in water")
    else:
        print("✗ BUOYANCY TEST FAILED: Need to improve buoyancy force calculations")
    
    print("\nThis detailed analysis helps identify where the physics needs adjustment.")