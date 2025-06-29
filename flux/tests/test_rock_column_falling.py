#!/usr/bin/env python3
"""Test a column of rock falling through space/air."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from materials import MaterialType


def test_rock_column_in_air():
    """Test rock column falling through air (not pure vacuum)."""
    print("=== Rock Column Falling Through Air Test ===")
    
    # Small domain
    sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario=False)
    sim.gravity_solver.use_self_gravity = False  # Uniform gravity
    
    # Enable required physics
    sim.enable_gravity = True
    sim.enable_momentum = True
    sim.enable_advection = True
    sim.enable_heat_transfer = False
    sim.enable_uranium_heating = False
    sim.enable_solar_heating = False
    sim.enable_phase_transitions = False
    sim.enable_atmospheric = False
    
    # Fill with air (not space) - air can be displaced
    sim.state.vol_frac.fill(0.0)
    sim.state.vol_frac[MaterialType.AIR.value] = 1.0
    
    # Create a column of rock
    col_x = sim.state.nx // 2
    col_height = 5
    start_y = 5
    
    for j in range(start_y, start_y + col_height):
        sim.state.vol_frac[MaterialType.ROCK.value, j, col_x] = 1.0
        sim.state.vol_frac[MaterialType.AIR.value, j, col_x] = 0.0
    
    # Set temperatures
    sim.state.temperature.fill(300.0)
    
    # Update properties
    sim.state.update_mixture_properties(sim.material_db)
    
    print(f"Rock column: x={col_x}, y={start_y} to {start_y + col_height - 1}")
    print(f"Air density: {sim.material_db.get_properties(MaterialType.AIR).density} kg/m³")
    print(f"Rock density: {sim.material_db.get_properties(MaterialType.ROCK).density} kg/m³")
    
    # Run simulation
    sim.paused = False
    positions_y = []
    times = []
    
    for step in range(30):
        # Find rock center of mass
        rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.5
        if np.any(rock_mask):
            rock_cells = np.where(rock_mask)
            y_center = np.mean(rock_cells[0])
            positions_y.append(y_center)
            times.append(sim.state.time)
            
            if step % 5 == 0:
                print(f"Step {step}: t={sim.state.time:.2f}s, y_center={y_center:.1f}")
                
                # Check velocity at rock center
                center_j = int(y_center)
                if 0 <= center_j < sim.state.ny:
                    vy = sim.state.velocity_y[center_j, col_x]
                    print(f"  Velocity at center: vy={vy:.2f} m/s")
        
        sim.step_forward()
        
        # Stop if rock reaches bottom
        if len(positions_y) > 0 and positions_y[-1] > sim.state.ny - 8:
            print("Rock reached bottom!")
            break
    
    # Analyze results
    if len(positions_y) > 5:
        distance_fallen = (positions_y[-1] - positions_y[0]) * sim.state.dx
        time_elapsed = times[-1] - times[0] if len(times) > 1 else 0
        
        print(f"\nResults:")
        print(f"  Distance fallen: {distance_fallen:.1f} m")
        print(f"  Time elapsed: {time_elapsed:.2f} s")
        
        if distance_fallen > 50.0:  # At least 1 cell
            print("✓ PASS: Rock column fell through air!")
            avg_velocity = distance_fallen / time_elapsed if time_elapsed > 0 else 0
            print(f"  Average velocity: {avg_velocity:.1f} m/s")
            return True
        else:
            print("✗ FAIL: Rock column didn't fall significantly")
            return False
    else:
        print("✗ FAIL: Not enough data collected")
        return False


def test_mixed_material():
    """Test with rock that's not 100% solid - has some air mixed in."""
    print("\n=== Mixed Material (Porous Rock) Test ===")
    
    sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario=False)
    sim.gravity_solver.use_self_gravity = False
    
    # Enable physics
    sim.enable_gravity = True
    sim.enable_momentum = True
    sim.enable_advection = True
    sim.enable_heat_transfer = False
    sim.enable_uranium_heating = False
    sim.enable_solar_heating = False
    sim.enable_phase_transitions = False
    sim.enable_atmospheric = False
    
    # Fill with space + small amount of air
    sim.state.vol_frac.fill(0.0)
    sim.state.vol_frac[MaterialType.SPACE.value] = 0.99
    sim.state.vol_frac[MaterialType.AIR.value] = 0.01  # Tiny bit of air
    
    # Create "porous" rock (90% rock, 10% air)
    rock_y = 8
    rock_x = sim.state.nx // 2
    sim.state.vol_frac[MaterialType.ROCK.value, rock_y, rock_x] = 0.9
    sim.state.vol_frac[MaterialType.AIR.value, rock_y, rock_x] = 0.1
    sim.state.vol_frac[MaterialType.SPACE.value, rock_y, rock_x] = 0.0
    
    sim.state.temperature.fill(300.0)
    sim.state.update_mixture_properties(sim.material_db)
    
    print(f"Porous rock at ({rock_x}, {rock_y})")
    print(f"Effective density: {sim.state.density[rock_y, rock_x]:.1f} kg/m³")
    
    # Track position
    initial_y = rock_y
    sim.paused = False
    
    for step in range(20):
        sim.step_forward()
        
        # Find rock position
        rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.5
        if np.any(rock_mask):
            rock_cells = np.where(rock_mask)
            current_y = np.mean(rock_cells[0])
            
            if step % 5 == 0:
                distance = (current_y - initial_y) * sim.state.dx
                print(f"Step {step}: y={current_y:.1f}, moved {distance:.1f} m")
    
    # Final check
    rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.5
    if np.any(rock_mask):
        final_y = np.mean(np.where(rock_mask)[0])
        total_distance = (final_y - initial_y) * sim.state.dx
        
        print(f"\nTotal distance moved: {total_distance:.1f} m")
        
        if total_distance > 25.0:
            print("✓ PASS: Porous rock fell!")
            return True
        else:
            print("✗ FAIL: Porous rock didn't fall much")
            return False
    else:
        print("✗ FAIL: Lost track of rock")
        return False


if __name__ == "__main__":
    print("Testing different approaches to rock falling...\n")
    
    # Test 1: Rock column in air
    test1_pass = test_rock_column_in_air()
    
    # Test 2: Porous rock
    test2_pass = test_mixed_material()
    
    print("\n=== Summary ===")
    print(f"Rock column in air: {'PASS' if test1_pass else 'FAIL'}")
    print(f"Porous rock: {'PASS' if test2_pass else 'FAIL'}")
    
    if test1_pass or test2_pass:
        print("\nAt least one approach works! Rocks CAN fall.")
        print("The key is ensuring there's displaceable material (air/fluid).")
    else:
        print("\nBoth tests failed. Further investigation needed.")