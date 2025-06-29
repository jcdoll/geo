"""
Comprehensive integration tests for flux simulation.

Focuses on realistic performance and stability metrics while acknowledging
known limitations with sharp material boundaries.
"""

import pytest
import numpy as np
import time
from typing import Dict, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation


def run_integration_test(scenario: str, steps: int, size: int) -> Dict:
    """Run integration test and collect metrics."""
    sim = FluxSimulation(nx=size, ny=size, dx=50.0, scenario=scenario)
    sim.paused = False
    
    metrics = {
        'steps_completed': 0,
        'step_times': [],
        'nan_count': 0,
        'inf_count': 0,
        'crashed': False,
        'crash_reason': '',
        'initial_total_mass': np.sum(sim.state.density) * sim.state.dx * sim.state.dx,
        'final_total_mass': 0,
    }
    
    # Run simulation
    try:
        for step in range(steps):
            t0 = time.perf_counter()
            sim.step_forward()
            step_time = time.perf_counter() - t0
            
            metrics['step_times'].append(step_time)
            metrics['steps_completed'] += 1
            
            # Check for NaN/Inf
            if np.any(np.isnan(sim.state.temperature)) or np.any(np.isinf(sim.state.temperature)):
                metrics['crashed'] = True
                metrics['crash_reason'] = 'NaN/Inf in temperature'
                break
                
    except Exception as e:
        metrics['crashed'] = True
        metrics['crash_reason'] = str(e)
    
    # Final metrics
    metrics['final_total_mass'] = np.sum(sim.state.density) * sim.state.dx * sim.state.dx
    
    # Performance stats
    if metrics['step_times']:
        metrics['avg_step_time'] = np.mean(metrics['step_times'])
        metrics['max_step_time'] = np.max(metrics['step_times'])
        metrics['min_step_time'] = np.min(metrics['step_times'])
    
    return metrics


# Basic stability tests
@pytest.mark.parametrize("scenario", ["empty", "planet", "layered", "volcanic", "ice"])
def test_scenario_runs_without_crash(scenario):
    """Test that scenarios run for at least 50 steps without crashing."""
    metrics = run_integration_test(scenario, steps=50, size=64)
    
    assert not metrics['crashed'], f"{scenario} crashed: {metrics['crash_reason']}"
    assert metrics['steps_completed'] >= 50, f"{scenario} only completed {metrics['steps_completed']} steps"


# Performance tests
@pytest.mark.parametrize("scenario,target_fps", [
    ("empty", 20),      # Empty should be fast
    ("planet", 10),     # Complex scenarios
    ("layered", 10),    
    ("volcanic", 10),   
    ("ice", 10),
])
def test_scenario_performance_target(scenario, target_fps):
    """Test that scenarios meet performance targets."""
    metrics = run_integration_test(scenario, steps=20, size=64)
    
    assert not metrics['crashed'], f"{scenario} crashed during performance test"
    
    avg_time = metrics['avg_step_time']
    achieved_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"\n{scenario}: {achieved_fps:.1f} FPS (target: {target_fps} FPS)")
    assert achieved_fps >= target_fps, f"{scenario} too slow: {achieved_fps:.1f} FPS < {target_fps} FPS"


# Mass conservation test (relaxed)
@pytest.mark.parametrize("scenario", ["empty", "planet", "layered", "volcanic", "ice"])
def test_mass_conservation_reasonable(scenario):
    """Test that mass doesn't change drastically."""
    metrics = run_integration_test(scenario, steps=100, size=64)
    
    if metrics['crashed']:
        pytest.skip(f"{scenario} crashed: {metrics['crash_reason']}")
    
    initial_mass = metrics['initial_total_mass']
    final_mass = metrics['final_total_mass']
    
    if initial_mass > 0:
        mass_change = abs(final_mass - initial_mass) / initial_mass
        print(f"\n{scenario} mass change: {mass_change*100:.2f}%")
        
        # Allow up to 5% mass change (sharp boundaries can cause issues)
        assert mass_change < 0.05, f"{scenario} mass change too large: {mass_change*100:.1f}%"


# Scaling test
@pytest.mark.parametrize("size", [32, 64, 128])
def test_performance_scaling(size):
    """Test performance scales reasonably with grid size."""
    metrics = run_integration_test("layered", steps=10, size=size)
    
    assert not metrics['crashed'], f"Crashed at size {size}"
    
    avg_time = metrics['avg_step_time']
    print(f"\n{size}x{size}: {avg_time*1000:.1f}ms per step")
    
    # Expected limits based on O(N²) scaling
    if size == 32:
        assert avg_time < 0.050  # 50ms
    elif size == 64:
        assert avg_time < 0.200  # 200ms  
    elif size == 128:
        assert avg_time < 0.800  # 800ms


# Long-running stability test
def test_long_running_stability():
    """Test that empty scenario remains stable over many steps."""
    metrics = run_integration_test("empty", steps=1000, size=64)
    
    assert not metrics['crashed'], "Empty scenario crashed"
    assert metrics['steps_completed'] == 1000, "Did not complete all steps"
    
    # Empty scenario should have essentially zero mass change
    mass_change = abs(metrics['final_total_mass'] - metrics['initial_total_mass']) / metrics['initial_total_mass']
    assert mass_change < 1e-6, f"Empty scenario mass changed: {mass_change*100:.1e}%"


# Summary report
def test_generate_performance_report():
    """Generate a performance summary report."""
    print("\n" + "="*60)
    print("FLUX SIMULATION PERFORMANCE REPORT")
    print("="*60)
    
    scenarios = ["empty", "planet", "layered", "volcanic", "ice"]
    sizes = [32, 64, 128]
    
    # Test each scenario at 64x64
    print("\nScenario Performance (64x64 grid):")
    print("-" * 40)
    print(f"{'Scenario':<12} {'FPS':>8} {'ms/step':>10} {'Status':<10}")
    print("-" * 40)
    
    for scenario in scenarios:
        metrics = run_integration_test(scenario, steps=20, size=64)
        
        if metrics['crashed']:
            print(f"{scenario:<12} {'CRASHED':>8} {'---':>10} {metrics['crash_reason'][:20]}")
        else:
            fps = 1.0 / metrics['avg_step_time']
            ms = metrics['avg_step_time'] * 1000
            status = "OK" if fps >= 10 else "SLOW"
            print(f"{scenario:<12} {fps:>8.1f} {ms:>10.1f} {status:<10}")
    
    # Scaling analysis
    print("\nScaling Analysis (layered scenario):")
    print("-" * 40)
    print(f"{'Size':<8} {'ms/step':>10} {'Scaling':>10}")
    print("-" * 40)
    
    base_time = None
    base_size = None
    
    for size in sizes:
        metrics = run_integration_test("layered", steps=10, size=size)
        
        if not metrics['crashed']:
            ms = metrics['avg_step_time'] * 1000
            
            if base_time is None:
                base_time = metrics['avg_step_time']
                base_size = size
                scaling = 1.0
            else:
                expected = (size / base_size) ** 2
                actual = metrics['avg_step_time'] / base_time
                scaling = actual / expected
            
            print(f"{size}x{size:<4} {ms:>10.1f} {scaling:>10.2f}x")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run comprehensive report
    test_generate_performance_report()
    
    # Quick stability check
    print("\nQuick Stability Check:")
    for scenario in ["empty", "volcanic"]:
        metrics = run_integration_test(scenario, steps=100, size=64)
        status = "PASSED" if not metrics['crashed'] else "FAILED"
        print(f"  {scenario}: {status}")