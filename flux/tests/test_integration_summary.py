"""
Summary integration test report for all scenarios.
"""

import pytest
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation


def test_all_scenarios_summary():
    """Generate a comprehensive summary of all scenario performance and stability."""
    
    scenarios = ["empty", "planet", "layered", "volcanic", "ice"]
    results = {}
    
    print("\n" + "="*80)
    print("FLUX SIMULATION INTEGRATION TEST SUMMARY")
    print("="*80)
    print("\nRunning 100 steps for each scenario on 64x64 grid...")
    print("-"*80)
    print(f"{'Scenario':<12} {'Steps':>6} {'FPS':>8} {'Max Vel':>10} {'Mass Chg':>10} {'Status':<15}")
    print("-"*80)
    
    for scenario in scenarios:
        # Create and run simulation
        sim = FluxSimulation(nx=64, ny=64, dx=50.0, scenario=scenario)
        sim.paused = False
        
        initial_mass = sim.state.get_total_mass()
        initial_step = sim.step_count
        
        # Run 100 steps
        times = []
        max_velocities = []
        issues = []
        
        for i in range(100):
            t0 = time.perf_counter()
            sim.step_forward()
            step_time = time.perf_counter() - t0
            times.append(step_time)
            
            # Track max velocity
            vel_mag = np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2)
            max_velocities.append(np.max(vel_mag))
            
            # Check for issues
            if np.any(np.isnan(sim.state.temperature)):
                issues.append("NaN in temperature")
                break
            if sim.state.get_total_mass() < 0:
                issues.append("Negative mass")
                break
        
        # Calculate results
        steps_completed = sim.step_count - initial_step
        avg_time = np.mean(times[5:]) if len(times) > 5 else np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        max_vel = max(max_velocities) if max_velocities else 0
        
        final_mass = sim.state.get_total_mass()
        mass_change = (final_mass - initial_mass) / initial_mass * 100 if initial_mass > 0 else 0
        
        # Determine status
        if issues:
            status = f"FAILED: {issues[0]}"
        elif steps_completed < 100:
            status = f"INCOMPLETE: {steps_completed}"
        elif max_vel > 10000:
            status = "UNSTABLE: vel"
        elif abs(mass_change) > 10:
            status = "UNSTABLE: mass"
        else:
            status = "OK"
        
        # Print results
        print(f"{scenario:<12} {steps_completed:>6} {fps:>8.1f} {max_vel:>10.1f} {mass_change:>10.2f}% {status:<15}")
        
        # Store for summary
        results[scenario] = {
            'steps': steps_completed,
            'fps': fps,
            'max_vel': max_vel,
            'mass_change': mass_change,
            'status': status
        }
    
    print("-"*80)
    
    # Performance summary
    print("\nPERFORMANCE SUMMARY:")
    print(f"  Average FPS across all scenarios: {np.mean([r['fps'] for r in results.values()]):.1f}")
    print(f"  Slowest scenario: {min(results.items(), key=lambda x: x[1]['fps'])[0]} "
          f"({min(r['fps'] for r in results.values()):.1f} FPS)")
    print(f"  Fastest scenario: {max(results.items(), key=lambda x: x[1]['fps'])[0]} "
          f"({max(r['fps'] for r in results.values()):.1f} FPS)")
    
    # Stability summary
    stable_count = sum(1 for r in results.values() if r['status'] == 'OK')
    print(f"\nSTABILITY SUMMARY:")
    print(f"  Stable scenarios: {stable_count}/{len(scenarios)}")
    print(f"  All scenarios completed 100 steps: {'YES' if all(r['steps'] == 100 for r in results.values()) else 'NO'}")
    
    # Assert all tests passed
    assert stable_count == len(scenarios), f"Only {stable_count}/{len(scenarios)} scenarios stable"
    assert all(r['steps'] == 100 for r in results.values()), "Not all scenarios completed 100 steps"
    assert all(r['fps'] >= 8 for r in results.values()), "Performance below 8 FPS threshold"
    
    print("\n✓ ALL TESTS PASSED")
    print("="*80)


if __name__ == "__main__":
    test_all_scenarios_summary()