#!/usr/bin/env python3
"""Test that rocks can now fall through space after fixing velocity zeroing."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from materials import MaterialType


def test_rock_falling_through_space():
    """Test rock falling through space with incompressible flow fixed."""
    print("=== Rock Falling Through Space Test (Fixed) ===")
    
    # Medium domain
    sim = FluxSimulation(nx=32, ny=64, dx=50.0, scenario=False)
    sim.gravity_solver.use_self_gravity = False  # Use uniform gravity
    
    # Enable required physics
    sim.enable_gravity = True
    sim.enable_momentum = True
    sim.enable_advection = True
    sim.enable_heat_transfer = False
    sim.enable_uranium_heating = False
    sim.enable_solar_heating = False
    sim.enable_phase_transitions = False
    sim.enable_atmospheric = False
    
    # Create scenario: rock in space
    sim.state.vol_frac.fill(0.0)
    sim.state.vol_frac[MaterialType.SPACE.value] = 1.0
    
    # Create a larger rock blob for better tracking
    rock_y = 10  # Near top
    rock_x = sim.state.nx // 2
    rock_radius = 3
    
    for j in range(rock_y - rock_radius, rock_y + rock_radius + 1):
        for i in range(rock_x - rock_radius, rock_x + rock_radius + 1):
            if 0 <= j < sim.state.ny and 0 <= i < sim.state.nx:
                dist = np.sqrt((j - rock_y)**2 + (i - rock_x)**2)
                if dist <= rock_radius:
                    sim.state.vol_frac[MaterialType.ROCK.value, j, i] = 1.0
                    sim.state.vol_frac[MaterialType.SPACE.value, j, i] = 0.0
    
    # Set temperatures
    sim.state.temperature.fill(300.0)
    
    # Update properties
    sim.state.update_mixture_properties(sim.material_db)
    
    # Get initial rock center of mass
    rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.1
    initial_com_y = np.average(np.where(rock_mask)[0], 
                              weights=sim.state.vol_frac[MaterialType.ROCK.value][rock_mask])
    
    print(f"Initial rock center of mass Y: {initial_com_y:.1f}")
    print(f"Rock blob radius: {rock_radius} cells")
    print(f"Space density: {sim.material_db.get_properties(MaterialType.SPACE).density} kg/m³")
    print(f"Rock density: {sim.material_db.get_properties(MaterialType.ROCK).density} kg/m³")
    
    # Run simulation
    sim.paused = False
    positions_y = [initial_com_y]
    times = [0.0]
    velocities_y = []
    
    max_steps = 100
    for step in range(max_steps):
        # Store previous position
        prev_y = positions_y[-1]
        prev_t = times[-1]
        
        # Step forward
        sim.step_forward()
        
        # Find rock center of mass
        rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.01
        if np.any(rock_mask):
            # Weight by volume fraction for accurate center of mass
            weights = sim.state.vol_frac[MaterialType.ROCK.value][rock_mask]
            com_y = np.average(np.where(rock_mask)[0], weights=weights)
            
            positions_y.append(com_y)
            times.append(sim.state.time)
            
            # Calculate velocity
            if sim.state.time > prev_t:
                vy = (com_y - prev_y) * sim.state.dx / (sim.state.time - prev_t)
                velocities_y.append(vy)
            
            if step % 20 == 0:
                distance = (com_y - initial_com_y) * sim.state.dx
                avg_vy = np.mean(velocities_y[-5:]) if len(velocities_y) >= 5 else (velocities_y[-1] if velocities_y else 0)
                print(f"Step {step}: t={sim.state.time:.2f}s, y={com_y:.1f}, distance={distance:.1f}m, v_y={avg_vy:.1f} m/s")
            
            # Stop if rock reaches bottom
            if com_y > sim.state.ny - 10:
                print("Rock reached bottom!")
                break
        else:
            print(f"Step {step}: Lost track of rock!")
            break
    
    # Analyze results
    if len(positions_y) > 10:
        total_distance = (positions_y[-1] - positions_y[0]) * sim.state.dx
        total_time = times[-1] - times[0]
        
        print(f"\n=== Results ===")
        print(f"Total distance fallen: {total_distance:.1f} m")
        print(f"Total time: {total_time:.2f} s")
        
        if len(velocities_y) > 5:
            initial_v = np.mean(velocities_y[:5])
            final_v = np.mean(velocities_y[-5:])
            avg_acceleration = (final_v - initial_v) / total_time if total_time > 0 else 0
            
            print(f"Initial velocity: {initial_v:.1f} m/s")
            print(f"Final velocity: {final_v:.1f} m/s")
            print(f"Average acceleration: {avg_acceleration:.2f} m/s²")
            print(f"Expected gravity: 9.81 m/s²")
        
        if total_distance > 100.0:  # At least 100m
            print("\n✓ PASS: Rock successfully fell through space!")
            
            # Compare to theoretical free fall
            theoretical_distance = 0.5 * 9.81 * total_time**2
            theoretical_final_v = 9.81 * total_time
            
            print(f"\nComparison to free fall in vacuum:")
            print(f"  Distance: {total_distance:.1f}m (actual) vs {theoretical_distance:.1f}m (theory)")
            print(f"  Final velocity: {final_v:.1f} m/s (actual) vs {theoretical_final_v:.1f} m/s (theory)")
            
            return True
        else:
            print("\n✗ FAIL: Rock didn't fall far enough")
            return False
    else:
        print("\n✗ FAIL: Not enough data collected")
        return False


if __name__ == "__main__":
    success = test_rock_falling_through_space()
    
    if success:
        print("\n🎉 The fix worked! Rocks can now fall through space.")
        print("The issue was that velocities were being artificially zeroed in low-density regions.")
        print("Now the incompressible solver correctly handles materials falling through space.")
    else:
        print("\nSomething is still preventing proper falling behavior.")