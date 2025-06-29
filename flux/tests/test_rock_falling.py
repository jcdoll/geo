#!/usr/bin/env python3
"""Test scenario to verify rocks fall through space under gravity."""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from materials import MaterialType
from gravity_solver import SolverMethod


def create_falling_rock_scenario(sim: FluxSimulation):
    """Create a scenario with rock suspended in space above a planet surface."""
    # Clear to all space first
    sim.state.vol_frac.fill(0.0)
    sim.state.vol_frac[MaterialType.SPACE.value] = 1.0
    
    # Create a ground layer at the bottom (rock)
    ground_height = 10  # cells
    sim.state.vol_frac[MaterialType.ROCK.value, -ground_height:, :] = 1.0
    sim.state.vol_frac[MaterialType.SPACE.value, -ground_height:, :] = 0.0
    
    # Place a blob of rock in space above the ground
    rock_y = sim.state.ny // 2  # Middle height
    rock_x = sim.state.nx // 2  # Center horizontally
    rock_radius = 3  # cells
    
    for j in range(max(0, rock_y - rock_radius), min(sim.state.ny, rock_y + rock_radius + 1)):
        for i in range(max(0, rock_x - rock_radius), min(sim.state.nx, rock_x + rock_radius + 1)):
            dist = np.sqrt((j - rock_y)**2 + (i - rock_x)**2)
            if dist <= rock_radius:
                sim.state.vol_frac[MaterialType.ROCK.value, j, i] = 1.0
                sim.state.vol_frac[MaterialType.SPACE.value, j, i] = 0.0
    
    # Set temperatures
    sim.state.temperature.fill(2.7)  # Space temperature
    rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.5
    sim.state.temperature[rock_mask] = 300.0  # Room temperature for rock
    
    # Update properties
    sim.state.update_mixture_properties(sim.material_db)
    
    return rock_y, rock_x


def test_self_gravity():
    """Test rock falling with self-gravity enabled."""
    print("=== Testing Rock Fall with Self-Gravity ===")
    
    # Create simulation with self-gravity
    sim = FluxSimulation(nx=64, ny=64, dx=50.0, scenario=False)
    sim.gravity_solver.use_self_gravity = True
    
    # Disable other physics to isolate gravity
    sim.enable_heat_transfer = False
    sim.enable_uranium_heating = False
    sim.enable_solar_heating = False
    sim.enable_phase_transitions = False
    sim.enable_atmospheric = False
    
    # Create scenario
    initial_rock_y, rock_x = create_falling_rock_scenario(sim)
    
    # Get initial rock position
    rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.9
    rock_cells = np.where(rock_mask)
    initial_y_mean = np.mean(rock_cells[0])
    
    print(f"Initial rock center Y: {initial_y_mean:.1f}")
    print(f"Domain height: {sim.state.ny} cells")
    
    # Run simulation for several seconds
    sim.paused = False
    total_time = 0.0
    max_time = 10.0  # seconds
    
    positions = []
    times = []
    velocities = []
    
    # Record data every few timesteps for better resolution
    steps_between_records = 5
    step_count = 0
    
    while total_time < max_time:
        sim.step_forward()
        total_time += sim.state.dt
        step_count += 1
        
        # Record data periodically
        if step_count % steps_between_records == 0:
            # Track rock position
            rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.9
            if np.any(rock_mask):
                rock_cells = np.where(rock_mask)
                y_mean = np.mean(rock_cells[0])
                
                # Calculate instantaneous velocity
                if len(positions) > 0:
                    dy = (y_mean - positions[-1]) * sim.state.dx  # meters
                    dt_elapsed = total_time - times[-1]
                    if dt_elapsed > 0:
                        v = dy / dt_elapsed
                        velocities.append(v)
                
                positions.append(y_mean)
                times.append(total_time)
                
                # Check if rock reached ground
                if y_mean > sim.state.ny - 15:
                    print(f"Rock reached ground at t={total_time:.2f}s")
                    print(f"Data points collected: {len(positions)}")
                    break
    
    # Analyze motion
    if len(velocities) > 3:
        # Check if rock accelerated (velocity increased)
        initial_v = np.mean(velocities[:min(3, len(velocities)//3)])
        final_v = np.mean(velocities[-min(3, len(velocities)//3):])
        
        print(f"\nMotion analysis:")
        print(f"  Initial velocity: {initial_v:.2f} m/s")
        print(f"  Final velocity: {final_v:.2f} m/s")
        print(f"  Total distance fallen: {(positions[-1] - positions[0]) * sim.state.dx:.1f} m")
        print(f"  Average acceleration: {(final_v - initial_v) / (times[-1] - times[0]):.2f} m/s²")
        
        # Check gravity field values at rock location
        gx, gy = sim.gravity_solver.solve_gravity()
        rock_g = np.mean(gy[rock_mask])
        print(f"  Gravity at rock location: {rock_g:.2f} m/s²")
        
        if final_v > initial_v + 1.0:
            print("✓ PASS: Rock accelerated under self-gravity")
            return True
        else:
            print("✗ FAIL: Rock did not accelerate significantly")
            return False
    else:
        print(f"✗ FAIL: Not enough velocity data points collected ({len(velocities)})")
        return False


def test_uniform_gravity():
    """Test rock falling with uniform gravity field."""
    print("\n=== Testing Rock Fall with Uniform Gravity ===")
    
    # Create simulation with uniform gravity
    sim = FluxSimulation(nx=64, ny=64, dx=50.0, scenario=False)
    sim.gravity_solver.use_self_gravity = False  # Use uniform gravity
    
    # Disable other physics
    sim.enable_heat_transfer = False
    sim.enable_uranium_heating = False
    sim.enable_solar_heating = False
    sim.enable_phase_transitions = False
    sim.enable_atmospheric = False
    
    # Create scenario
    initial_rock_y, rock_x = create_falling_rock_scenario(sim)
    
    # Get initial rock position
    rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.9
    rock_cells = np.where(rock_mask)
    initial_y_mean = np.mean(rock_cells[0])
    
    print(f"Initial rock center Y: {initial_y_mean:.1f}")
    
    # Run simulation
    sim.paused = False
    total_time = 0.0
    max_time = 10.0
    
    positions = []
    times = []
    velocities = []
    
    # Record data every few timesteps
    steps_between_records = 5
    step_count = 0
    
    while total_time < max_time:
        sim.step_forward()
        total_time += sim.state.dt
        step_count += 1
        
        if step_count % steps_between_records == 0:
            # Track rock position
            rock_mask = sim.state.vol_frac[MaterialType.ROCK.value] > 0.9
            if np.any(rock_mask):
                rock_cells = np.where(rock_mask)
                y_mean = np.mean(rock_cells[0])
                
                # Calculate velocity
                if len(positions) > 0:
                    dy = (y_mean - positions[-1]) * sim.state.dx
                    dt_elapsed = total_time - times[-1]
                    if dt_elapsed > 0:
                        v = dy / dt_elapsed
                        velocities.append(v)
                
                positions.append(y_mean)
                times.append(total_time)
                
                if y_mean > sim.state.ny - 15:
                    print(f"Rock reached ground at t={total_time:.2f}s")
                    print(f"Data points collected: {len(positions)}")
                    break
    
    # Analyze motion
    if len(velocities) > 3:
        # Simple velocity analysis
        initial_v = np.mean(velocities[:min(3, len(velocities)//3)])
        final_v = np.mean(velocities[-min(3, len(velocities)//3):])
        
        print(f"\nMotion analysis:")
        print(f"  Initial velocity: {initial_v:.2f} m/s")
        print(f"  Final velocity: {final_v:.2f} m/s")
        print(f"  Total distance fallen: {(positions[-1] - positions[0]) * sim.state.dx:.1f} m")
        print(f"  Average acceleration: {(final_v - initial_v) / (times[-1] - times[0]):.2f} m/s²")
        print(f"  Expected uniform gravity: 9.81 m/s²")
        
        # Check the actual gravity field
        gx, gy = sim.gravity_solver.solve_gravity()
        print(f"  Gravity field (should be uniform): min={np.min(gy):.2f}, max={np.max(gy):.2f} m/s²")
        
        if final_v > initial_v + 1.0:
            print("✓ PASS: Rock accelerated under uniform gravity")
            return True
        else:
            print("✗ FAIL: Rock did not accelerate")
            return False
    else:
        print(f"✗ FAIL: Not enough velocity data points collected ({len(velocities)})")
        return False


def test_visual_check():
    """Run a visual test to see the rock falling."""
    print("\n=== Visual Test of Rock Falling ===")
    print("Run with: python main.py --scenario falling_rock")
    
    # Create a scenario file that can be loaded
    scenario_code = '''
def setup_falling_rock(state, material_db):
    """Scenario with rock falling through space."""
    from materials import MaterialType
    import numpy as np
    
    # Clear to space
    state.vol_frac.fill(0.0)
    state.vol_frac[MaterialType.SPACE.value] = 1.0
    
    # Ground layer
    state.vol_frac[MaterialType.ROCK.value, -10:, :] = 1.0
    state.vol_frac[MaterialType.SPACE.value, -10:, :] = 0.0
    
    # Falling rock
    cy, cx = state.ny // 2, state.nx // 2
    for j in range(cy-3, cy+4):
        for i in range(cx-3, cx+4):
            if 0 <= j < state.ny and 0 <= i < state.nx:
                dist = np.sqrt((j-cy)**2 + (i-cx)**2)
                if dist <= 3:
                    state.vol_frac[MaterialType.ROCK.value, j, i] = 1.0
                    state.vol_frac[MaterialType.SPACE.value, j, i] = 0.0
    
    # Temperatures
    state.temperature.fill(2.7)
    rock_mask = state.vol_frac[MaterialType.ROCK.value] > 0.5
    state.temperature[rock_mask] = 300.0
'''
    
    print("To add this scenario, add the above function to scenarios.py")
    print("and add 'falling_rock': setup_falling_rock to SCENARIOS dict")


if __name__ == "__main__":
    print("Testing rock falling through space...\n")
    
    # Run tests
    self_gravity_pass = test_self_gravity()
    uniform_gravity_pass = test_uniform_gravity()
    
    print("\n=== Test Summary ===")
    print(f"Self-gravity test: {'PASS' if self_gravity_pass else 'FAIL'}")
    print(f"Uniform gravity test: {'PASS' if uniform_gravity_pass else 'FAIL'}")
    
    if self_gravity_pass and uniform_gravity_pass:
        print("\n✓ All tests passed! Rocks fall through space as expected.")
    else:
        print("\n✗ Some tests failed. Check the implementation.")
    
    # Show how to run visual test
    test_visual_check()