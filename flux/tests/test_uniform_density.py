#!/usr/bin/env python3
"""Test scenario with uniform water density to check simulation stability."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from state import FluxState
from materials import MaterialType


def test_uniform_water(grid_size, timesteps=100):
    """Test simulation stability with uniform water density.
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        timesteps: Number of timesteps to run
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing {grid_size}x{grid_size} grid with uniform water")
    print(f"{'='*60}")
    
    # Create simulation
    sim = FluxSimulation(grid_size, grid_size)
    state = sim.state
    
    # Fill entire domain with water
    water_idx = MaterialType.WATER.value
    state.vol_frac.fill(0.0)
    state.vol_frac[water_idx, :, :] = 1.0
    
    # Set uniform temperature (20°C)
    state.temperature.fill(293.15)
    
    # Initialize with zero velocity
    state.velocity_x.fill(0.0)
    state.velocity_y.fill(0.0)
    
    # Track maximum velocities over time
    max_vx_history = []
    max_vy_history = []
    max_v_magnitude_history = []
    
    # Run simulation
    for step in range(timesteps):
        # Compute timestep using CFL limit
        dt = sim.physics.apply_cfl_limit()
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
    
    # Check if simulation is stable
    final_max_v = max_v_magnitude_history[-1]
    is_stable = final_max_v < 1e-6  # Very small threshold for "no motion"
    
    # Calculate statistics
    results = {
        "grid_size": grid_size,
        "timesteps": timesteps,
        "final_max_velocity": final_max_v,
        "final_max_vx": max_vx_history[-1],
        "final_max_vy": max_vy_history[-1],
        "average_max_velocity": np.mean(max_v_magnitude_history),
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
    
    # Check for velocity growth
    if len(max_v_magnitude_history) > 10:
        early_avg = np.mean(max_v_magnitude_history[:10])
        late_avg = np.mean(max_v_magnitude_history[-10:])
        if late_avg > early_avg * 2:
            print(f"  WARNING: Velocity appears to be growing!")
            print(f"    Early average: {early_avg:.2e} m/s")
            print(f"    Late average: {late_avg:.2e} m/s")
    
    return results


def main():
    """Run uniform density tests at different grid sizes."""
    print("Testing simulation stability with uniform water density")
    print("Water density: 1000 kg/m³")
    print("Initial velocity: 0 m/s everywhere")
    print("Expected behavior: No motion should develop\n")
    
    # Test different grid sizes
    grid_sizes = [64, 128]
    results = {}
    
    for size in grid_sizes:
        results[size] = test_uniform_water(size, timesteps=100)
    
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
        print("\n✓ All simulations are stable with uniform density!")
    else:
        print("\n✗ Some simulations show instability with uniform density!")
        print("  This suggests numerical issues in the solver.")
    
    # Plot velocity history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(grid_sizes), figsize=(12, 5))
        if len(grid_sizes) == 1:
            axes = [axes]
        
        for i, size in enumerate(grid_sizes):
            r = results[size]
            ax = axes[i]
            steps = np.arange(len(r["max_v_magnitude_history"]))
            
            ax.semilogy(steps, r["max_v_magnitude_history"], 'b-', label='|v| max')
            ax.semilogy(steps, r["max_vx_history"], 'r--', alpha=0.7, label='|vx| max')
            ax.semilogy(steps, r["max_vy_history"], 'g--', alpha=0.7, label='|vy| max')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Maximum Velocity (m/s)')
            ax.set_title(f'{size}x{size} Grid')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add stability indicator
            if r["is_stable"]:
                ax.text(0.95, 0.95, 'STABLE', transform=ax.transAxes,
                       ha='right', va='top', color='green', fontweight='bold')
            else:
                ax.text(0.95, 0.95, 'UNSTABLE', transform=ax.transAxes,
                       ha='right', va='top', color='red', fontweight='bold')
        
        plt.suptitle('Uniform Water Density Test - Velocity Evolution')
        plt.tight_layout()
        plt.savefig('uniform_density_test.png', dpi=150)
        print(f"\nVelocity history plot saved to uniform_density_test.png")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")


if __name__ == "__main__":
    main()