#!/usr/bin/env python3
"""Test scenario with uniform water density and no gravity to isolate instability source."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from state import FluxState
from materials import MaterialType, MaterialDatabase


def test_uniform_water_no_gravity(grid_size, timesteps=100):
    """Test simulation stability with uniform water density and no gravity.
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        timesteps: Number of timesteps to run
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing {grid_size}x{grid_size} grid with uniform water (no gravity)")
    print(f"{'='*60}")
    
    # Create simulation without automatic scenario setup
    sim = FluxSimulation(grid_size, grid_size, scenario=None)
    state = sim.state
    
    # Disable gravity completely
    sim.enable_gravity = False
    state.gravity_x.fill(0.0)
    state.gravity_y.fill(0.0)
    
    # Fill entire domain with water
    water_idx = MaterialType.WATER.value
    state.vol_frac.fill(0.0)
    state.vol_frac[water_idx, :, :] = 1.0
    
    # Set uniform temperature (20°C)
    state.temperature.fill(293.15)
    
    # Initialize with zero velocity
    state.velocity_x.fill(0.0)
    state.velocity_y.fill(0.0)
    state.velocity_x_face.fill(0.0)
    state.velocity_y_face.fill(0.0)
    
    # Update material properties
    material_db = MaterialDatabase()
    state.update_mixture_properties(material_db)
    state.update_face_coefficients()
    
    # Track maximum velocities over time
    max_vx_history = []
    max_vy_history = []
    max_v_magnitude_history = []
    
    # Run simulation
    for step in range(timesteps):
        # Compute timestep using CFL limit
        dt = sim.physics.apply_cfl_limit()
        
        # For debugging: print dt
        if step < 5:
            print(f"Step {step}: dt = {dt:.6f} s")
        
        sim.timestep(dt)
        
        # Get maximum velocities
        max_vx = np.max(np.abs(state.velocity_x))
        max_vy = np.max(np.abs(state.velocity_y))
        v_magnitude = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
        max_v_mag = np.max(v_magnitude)
        
        max_vx_history.append(max_vx)
        max_vy_history.append(max_vy)
        max_v_magnitude_history.append(max_v_mag)
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step:3d}: max |vx|={max_vx:.2e} m/s, "
                  f"max |vy|={max_vy:.2e} m/s, "
                  f"max |v|={max_v_mag:.2e} m/s")
            
        # Stop if velocity explodes
        if max_v_mag > 1e6:
            print(f"ERROR: Velocity explosion at step {step}!")
            break
    
    # Check if simulation is stable
    final_max_v = max_v_magnitude_history[-1] if max_v_magnitude_history else float('inf')
    is_stable = final_max_v < 1e-10  # Very small threshold for "no motion"
    
    # Calculate statistics
    results = {
        "grid_size": grid_size,
        "timesteps": len(max_v_magnitude_history),
        "final_max_velocity": final_max_v,
        "final_max_vx": max_vx_history[-1] if max_vx_history else float('inf'),
        "final_max_vy": max_vy_history[-1] if max_vy_history else float('inf'),
        "average_max_velocity": np.mean(max_v_magnitude_history) if max_v_magnitude_history else float('inf'),
        "is_stable": is_stable,
        "max_vx_history": max_vx_history,
        "max_vy_history": max_vy_history,
        "max_v_magnitude_history": max_v_magnitude_history
    }
    
    # Print summary
    print(f"\nSummary for {grid_size}x{grid_size} grid:")
    print(f"  Final maximum velocity: {final_max_v:.2e} m/s")
    print(f"  Average maximum velocity: {results['average_max_velocity']:.2e} m/s")
    print(f"  Simulation stable: {'YES' if is_stable else 'NO'}")
    
    return results


def test_static_pressure():
    """Test if pressure solver works with static conditions."""
    print(f"\n{'='*60}")
    print("Testing pressure solver with static water")
    print(f"{'='*60}")
    
    grid_size = 32
    sim = FluxSimulation(grid_size, grid_size, scenario=None)
    state = sim.state
    
    # Fill with water
    water_idx = MaterialType.WATER.value
    state.vol_frac.fill(0.0)
    state.vol_frac[water_idx, :, :] = 1.0
    
    # Set zero velocity
    state.velocity_x.fill(0.0)
    state.velocity_y.fill(0.0)
    state.velocity_x_face.fill(0.0)
    state.velocity_y_face.fill(0.0)
    
    # Update properties
    material_db = MaterialDatabase()
    state.update_mixture_properties(material_db)
    state.update_face_coefficients()
    
    # Apply simple uniform gravity
    gx = 0.0
    gy = 9.81
    
    print(f"Initial state:")
    print(f"  Density: {state.density[0,0]:.1f} kg/m³")
    print(f"  Gravity: ({gx:.1f}, {gy:.1f}) m/s²")
    
    # Call physics update with small timestep
    dt = 0.001
    sim.physics.update_momentum(gx * np.ones_like(state.gravity_x), 
                               gy * np.ones_like(state.gravity_y), dt)
    
    # Check results
    max_vx = np.max(np.abs(state.velocity_x))
    max_vy = np.max(np.abs(state.velocity_y))
    print(f"\nAfter momentum update:")
    print(f"  Max |vx|: {max_vx:.2e} m/s")
    print(f"  Max |vy|: {max_vy:.2e} m/s")
    print(f"  Pressure range: [{np.min(state.pressure):.1f}, {np.max(state.pressure):.1f}] Pa")
    
    # Expected pressure gradient: dP/dy = ρ*g = 1000 * 9.81 = 9810 Pa/m
    expected_pressure_diff = 1000 * 9.81 * (grid_size - 1) * 50  # 50m cell size
    actual_pressure_diff = np.max(state.pressure) - np.min(state.pressure)
    print(f"\nPressure gradient check:")
    print(f"  Expected ΔP: {expected_pressure_diff:.1f} Pa")
    print(f"  Actual ΔP: {actual_pressure_diff:.1f} Pa")
    print(f"  Ratio: {actual_pressure_diff/expected_pressure_diff:.3f}")


def main():
    """Run uniform density tests without gravity."""
    print("Testing simulation stability with uniform water density (gravity disabled)")
    print("Water density: 1000 kg/m³")
    print("Gravity: DISABLED")
    print("Initial velocity: 0 m/s everywhere")
    print("Expected behavior: No motion should develop\n")
    
    # First test the pressure solver in isolation
    test_static_pressure()
    
    # Then test full simulation without gravity
    grid_sizes = [32, 64]
    results = {}
    
    for size in grid_sizes:
        results[size] = test_uniform_water_no_gravity(size, timesteps=50)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    all_stable = True
    for size in grid_sizes:
        r = results[size]
        status = "STABLE" if r["is_stable"] else "UNSTABLE"
        print(f"{size}x{size} grid: {status} (final max velocity: {r['final_max_velocity']:.2e} m/s)")
        all_stable = all_stable and r["is_stable"]
    
    if all_stable:
        print("\n✓ All simulations are stable without gravity!")
        print("  This suggests the instability is in the gravity solver or momentum update.")
    else:
        print("\n✗ Simulations are unstable even without gravity!")
        print("  This suggests a fundamental issue in the velocity/pressure coupling.")


if __name__ == "__main__":
    main()