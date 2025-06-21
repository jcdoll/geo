#!/usr/bin/env python3
"""Test script to measure performance improvements from optimizations.

This script compares the performance of:
1. Original vs optimized force-based swapping
2. Original vs optimized stats collection  
3. Original vs optimized material properties update
"""

import time
import numpy as np
import sys
import matplotlib.pyplot as plt

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from geo_game import GeoGame
from materials import MaterialType


def time_function(func, *args, **kwargs):
    """Time a function execution."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000, result  # Return time in ms


def create_test_simulation(size=100):
    """Create a test simulation with diverse materials."""
    sim = GeoGame(size, size, setup_planet=True)
    
    # Add some variety to materials to make tests more realistic
    # Add water blob
    water_center = (size // 4, size // 4)
    for y in range(water_center[1] - 5, water_center[1] + 5):
        for x in range(water_center[0] - 5, water_center[0] + 5):
            if 0 <= x < size and 0 <= y < size:
                dist = np.sqrt((x - water_center[0])**2 + (y - water_center[1])**2)
                if dist < 5:
                    sim.material_types[y, x] = MaterialType.WATER
                    sim.temperature[y, x] = 300.0
    
    # Add some air
    for y in range(5):
        for x in range(size):
            if sim.material_types[y, x] == MaterialType.SPACE:
                sim.material_types[y, x] = MaterialType.AIR
                sim.temperature[y, x] = 280.0
    
    # Update properties after material changes
    sim._update_material_properties()
    
    # Calculate initial gravity and pressure for realistic forces
    sim.calculate_self_gravity()
    if sim.enable_pressure:
        sim.fluid_dynamics.calculate_planetary_pressure()
    
    return sim


def test_force_swapping_performance():
    """Test performance of force-based swapping optimization."""
    print("\n=== Testing Force-Based Swapping Performance ===")
    
    sim = create_test_simulation(100)
    
    # Get force fields for testing - compute them properly
    sim.fluid_dynamics.compute_force_field()
    
    # Test original implementation
    print("Testing original apply_force_based_swapping...")
    times_original = []
    for _ in range(5):
        time_ms, _ = time_function(sim.fluid_dynamics.apply_force_based_swapping_slow)
        times_original.append(time_ms)
    
    avg_original = np.mean(times_original)
    std_original = np.std(times_original)
    print(f"Original: {avg_original:.1f} ± {std_original:.1f} ms")
    
    # Test optimized implementation
    print("Testing optimized apply_force_based_swapping...")
    times_optimized = []
    for _ in range(5):
        time_ms, _ = time_function(sim.fluid_dynamics.apply_force_based_swapping)
        times_optimized.append(time_ms)
    
    avg_optimized = np.mean(times_optimized)
    std_optimized = np.std(times_optimized)
    print(f"Optimized: {avg_optimized:.1f} ± {std_optimized:.1f} ms")
    
    speedup = avg_original / avg_optimized
    print(f"Speedup: {speedup:.1f}x")
    
    return avg_original, avg_optimized


def test_stats_collection_performance():
    """Test performance of stats collection optimization."""
    print("\n=== Testing Stats Collection Performance ===")
    
    sim = create_test_simulation(100)
    
    # Test original implementation
    print("Testing original _record_time_series_data...")
    times_original = []
    for _ in range(10):
        # Advance time to ensure recording happens
        sim.time += 1.0
        time_ms, _ = time_function(sim._record_time_series_data_slow)
        times_original.append(time_ms)
    
    avg_original = np.mean(times_original)
    std_original = np.std(times_original)
    print(f"Original: {avg_original:.1f} ± {std_original:.1f} ms")
    
    # Test optimized implementation
    print("Testing optimized _record_time_series_data...")
    times_optimized = []
    for _ in range(10):
        # Advance time to ensure recording happens
        sim.time += 1.0
        time_ms, _ = time_function(sim._record_time_series_data)
        times_optimized.append(time_ms)
    
    avg_optimized = np.mean(times_optimized)
    std_optimized = np.std(times_optimized)
    print(f"Optimized: {avg_optimized:.1f} ± {std_optimized:.1f} ms")
    
    speedup = avg_original / avg_optimized
    print(f"Speedup: {speedup:.1f}x")
    
    return avg_original, avg_optimized


def test_material_properties_performance():
    """Test performance of material properties update optimization."""
    print("\n=== Testing Material Properties Update Performance ===")
    
    sim = create_test_simulation(100)
    
    # Test original implementation
    print("Testing original _update_material_properties...")
    times_original = []
    for _ in range(10):
        sim._properties_dirty = True
        time_ms, _ = time_function(sim._update_material_properties_slow, force=True)
        times_original.append(time_ms)
    
    avg_original = np.mean(times_original)
    std_original = np.std(times_original)
    print(f"Original: {avg_original:.1f} ± {std_original:.1f} ms")
    
    # Test optimized implementation
    print("Testing optimized _update_material_properties...")
    times_optimized = []
    for _ in range(10):
        sim._properties_dirty = True
        time_ms, _ = time_function(sim._update_material_properties, force=True)
        times_optimized.append(time_ms)
    
    avg_optimized = np.mean(times_optimized)
    std_optimized = np.std(times_optimized)
    print(f"Optimized: {avg_optimized:.1f} ± {std_optimized:.1f} ms")
    
    speedup = avg_original / avg_optimized
    print(f"Speedup: {speedup:.1f}x")
    
    return avg_original, avg_optimized


def test_full_step_performance():
    """Test overall step_forward performance."""
    print("\n=== Testing Full Step Performance ===")
    
    # Create two simulations - one using original methods, one using optimized
    sim_original = create_test_simulation(100)
    sim_optimized = create_test_simulation(100)
    
    # Monkey-patch the original simulation to use old methods
    sim_original._record_time_series_data = sim_original._record_time_series_data_slow
    sim_original._update_material_properties = sim_original._update_material_properties_slow
    sim_original.fluid_dynamics.apply_force_based_swapping = sim_original.fluid_dynamics.apply_force_based_swapping_slow
    
    # Run several steps with original
    print("Testing original step_forward...")
    times_original = []
    for _ in range(5):
        time_ms, _ = time_function(sim_original.step_forward)
        times_original.append(time_ms)
        # Extract UK force swaps time if available
        if hasattr(sim_original, '_perf_times') and 'unified_kinematics' in sim_original._perf_times:
            uk_time = sim_original._perf_times['unified_kinematics'] * 1000
            print(f"  UK time: {uk_time:.1f} ms")
    
    avg_original = np.mean(times_original)
    std_original = np.std(times_original)
    print(f"Original total: {avg_original:.1f} ± {std_original:.1f} ms")
    
    # Run several steps with optimized
    print("\nTesting optimized step_forward...")
    times_optimized = []
    for _ in range(5):
        time_ms, _ = time_function(sim_optimized.step_forward)
        times_optimized.append(time_ms)
        # Extract UK force swaps time if available
        if hasattr(sim_optimized, '_perf_times') and 'unified_kinematics' in sim_optimized._perf_times:
            uk_time = sim_optimized._perf_times['unified_kinematics'] * 1000
            print(f"  UK time: {uk_time:.1f} ms")
    
    avg_optimized = np.mean(times_optimized)
    std_optimized = np.std(times_optimized)
    print(f"Optimized total: {avg_optimized:.1f} ± {std_optimized:.1f} ms")
    
    speedup = avg_original / avg_optimized
    print(f"Overall speedup: {speedup:.1f}x")
    
    # Calculate target FPS
    fps_original = 1000.0 / avg_original
    fps_optimized = 1000.0 / avg_optimized
    print(f"\nEstimated FPS:")
    print(f"  Original: {fps_original:.1f} FPS")
    print(f"  Optimized: {fps_optimized:.1f} FPS")
    print(f"  Target: 30-60 FPS")
    
    return avg_original, avg_optimized


def plot_results(results):
    """Plot performance comparison results."""
    categories = list(results.keys())
    original_times = [results[cat][0] for cat in categories]
    optimized_times = [results[cat][1] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, original_times, width, label='Original', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, optimized_times, width, label='Optimized', color='green', alpha=0.7)
    
    ax.set_xlabel('Operation')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Performance Optimization Results')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add speedup labels
    for i, (orig, opt) in enumerate(zip(original_times, optimized_times)):
        if opt > 0:
            speedup = orig / opt
            ax.text(i, max(orig, opt) * 1.05, f'{speedup:.1f}x', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_optimization_results.png', dpi=150)
    print("\nResults saved to performance_optimization_results.png")


def main():
    """Run all performance tests."""
    print("Performance Optimization Test Suite")
    print("===================================")
    
    results = {}
    
    # Test individual optimizations
    try:
        results['Force Swapping'] = test_force_swapping_performance()
    except Exception as e:
        print(f"Force swapping test failed: {e}")
        results['Force Swapping'] = (0, 0)
    
    try:
        results['Stats Collection'] = test_stats_collection_performance()
    except Exception as e:
        print(f"Stats collection test failed: {e}")
        results['Stats Collection'] = (0, 0)
    
    try:
        results['Material Properties'] = test_material_properties_performance()
    except Exception as e:
        print(f"Material properties test failed: {e}")
        results['Material Properties'] = (0, 0)
    
    # Test overall performance
    try:
        results['Full Step'] = test_full_step_performance()
    except Exception as e:
        print(f"Full step test failed: {e}")
        results['Full Step'] = (0, 0)
    
    # Plot results
    try:
        plot_results(results)
    except Exception as e:
        print(f"Failed to plot results: {e}")
    
    # Summary
    print("\n=== SUMMARY ===")
    total_original = sum(r[0] for r in results.values() if r != results.get('Full Step', (0, 0)))
    total_optimized = sum(r[1] for r in results.values() if r != results.get('Full Step', (0, 0)))
    
    if total_optimized > 0:
        component_speedup = total_original / total_optimized
        print(f"Component speedup: {component_speedup:.1f}x")
    
    if 'Full Step' in results and results['Full Step'][1] > 0:
        overall_speedup = results['Full Step'][0] / results['Full Step'][1]
        print(f"Overall speedup: {overall_speedup:.1f}x")
        
        fps_optimized = 1000.0 / results['Full Step'][1]
        print(f"Optimized FPS: {fps_optimized:.1f}")
        
        if fps_optimized >= 30:
            print("✓ Performance target achieved (30-60 FPS)")
        else:
            print(f"✗ Performance below target (need {30 - fps_optimized:.1f} more FPS)")


if __name__ == "__main__":
    main()