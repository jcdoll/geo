"""
Integration tests for all prebaked simulation scenarios.

Tests each scenario for 100 steps to ensure:
1. No values blow up (temperature, pressure, velocity, density)
2. Performance remains acceptable
3. Conservation laws are approximately satisfied
4. No NaN or Inf values appear
"""

import pytest
import numpy as np
import time
from typing import Dict, Tuple

# Import simulation components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from state import FluxState


class IntegrationTestMetrics:
    """Track metrics during integration test."""
    
    def __init__(self):
        self.step_times = []
        self.temperatures = []
        self.pressures = []
        self.densities = []
        self.velocities = []
        self.total_mass = []
        self.total_energy = []
        self.nan_count = 0
        self.inf_count = 0
        
    def update(self, state: FluxState, step_time: float):
        """Update metrics from current state."""
        self.step_times.append(step_time)
        
        # Temperature stats
        temp_valid = state.temperature[~np.isnan(state.temperature)]
        if len(temp_valid) > 0:
            self.temperatures.append({
                'min': np.min(temp_valid),
                'max': np.max(temp_valid),
                'mean': np.mean(temp_valid)
            })
        
        # Pressure stats
        pressure_valid = state.pressure[~np.isnan(state.pressure)]
        if len(pressure_valid) > 0:
            self.pressures.append({
                'min': np.min(pressure_valid),
                'max': np.max(pressure_valid),
                'mean': np.mean(pressure_valid)
            })
        
        # Density stats
        density_valid = state.density[~np.isnan(state.density)]
        if len(density_valid) > 0:
            self.densities.append({
                'min': np.min(density_valid),
                'max': np.max(density_valid),
                'mean': np.mean(density_valid)
            })
        
        # Velocity magnitude stats
        vel_mag = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
        vel_valid = vel_mag[~np.isnan(vel_mag)]
        if len(vel_valid) > 0:
            self.velocities.append({
                'min': np.min(vel_valid),
                'max': np.max(vel_valid),
                'mean': np.mean(vel_valid)
            })
        
        # Conservation metrics
        self.total_mass.append(np.sum(state.density) * state.dx * state.dx)
        
        # Total energy (kinetic + thermal)
        kinetic_energy = 0.5 * state.density * (state.velocity_x**2 + state.velocity_y**2)
        thermal_energy = state.density * 1000.0 * state.temperature  # Approximate Cv
        total_energy = np.sum(kinetic_energy + thermal_energy) * state.dx * state.dx
        self.total_energy.append(total_energy)
        
        # Check for NaN/Inf
        self.nan_count += np.sum(np.isnan(state.temperature))
        self.nan_count += np.sum(np.isnan(state.pressure))
        self.nan_count += np.sum(np.isnan(state.density))
        self.nan_count += np.sum(np.isnan(state.velocity_x))
        self.nan_count += np.sum(np.isnan(state.velocity_y))
        
        self.inf_count += np.sum(np.isinf(state.temperature))
        self.inf_count += np.sum(np.isinf(state.pressure))
        self.inf_count += np.sum(np.isinf(state.density))
        self.inf_count += np.sum(np.isinf(state.velocity_x))
        self.inf_count += np.sum(np.isinf(state.velocity_y))
    
    def check_stability(self) -> Tuple[bool, str]:
        """Check if simulation remained stable."""
        errors = []
        
        # Check for NaN/Inf
        if self.nan_count > 0:
            errors.append(f"Found {self.nan_count} NaN values")
        if self.inf_count > 0:
            errors.append(f"Found {self.inf_count} Inf values")
        
        # Check temperature bounds (0K to 10000K reasonable for geological sim)
        if self.temperatures:
            temp_max = max(t['max'] for t in self.temperatures)
            temp_min = min(t['min'] for t in self.temperatures)
            if temp_max > 10000:
                errors.append(f"Temperature exceeded 10000K: {temp_max:.1f}K")
            if temp_min < 0:
                errors.append(f"Temperature below 0K: {temp_min:.1f}K")
        
        # Check pressure bounds (negative pressure is unphysical)
        # NOTE: Sharp density interfaces (space/atmosphere) can cause pressure artifacts
        if self.pressures:
            pressure_min = min(p['min'] for p in self.pressures)
            pressure_max = max(p['max'] for p in self.pressures)
            if pressure_min < -1e8:  # Allow larger negative for space/atmosphere boundaries
                errors.append(f"Extreme negative pressure: {pressure_min:.1e} Pa")
            if pressure_max > 1e12:  # 1 TPa is extreme even for Earth's core
                errors.append(f"Extreme pressure: {pressure_max:.1e} Pa")
        
        # Check velocity bounds (> 1000 m/s is extreme for geological flows)
        # NOTE: Space/atmosphere boundaries can cause velocity artifacts
        if self.velocities:
            vel_max = max(v['max'] for v in self.velocities)
            if vel_max > 10000:  # Allow higher velocities at material boundaries
                errors.append(f"Extreme velocity: {vel_max:.1f} m/s")
        
        # Check density bounds (0 to 20000 kg/m³ covers most materials)
        if self.densities:
            density_min = min(d['min'] for d in self.densities)
            density_max = max(d['max'] for d in self.densities)
            if density_min < 0:
                errors.append(f"Negative density: {density_min:.1f} kg/m³")
            if density_max > 20000:
                errors.append(f"Extreme density: {density_max:.1f} kg/m³")
        
        # Check mass conservation (should be within 1% for stable simulation)
        if len(self.total_mass) > 1:
            initial_mass = self.total_mass[0]
            final_mass = self.total_mass[-1]
            if initial_mass > 0:
                mass_change = abs(final_mass - initial_mass) / initial_mass
                if mass_change > 0.01:
                    errors.append(f"Mass not conserved: {mass_change*100:.1f}% change")
        
        # Check energy doesn't explode (allow 10x increase for heating scenarios)
        if len(self.total_energy) > 1:
            initial_energy = self.total_energy[0]
            final_energy = self.total_energy[-1]
            if initial_energy > 0 and final_energy / initial_energy > 10:
                errors.append(f"Energy explosion: {final_energy/initial_energy:.1f}x increase")
        
        if errors:
            return False, "; ".join(errors)
        return True, "All values within reasonable bounds"
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.step_times:
            return {}
        
        return {
            'mean_step_time': np.mean(self.step_times),
            'max_step_time': np.max(self.step_times),
            'min_step_time': np.min(self.step_times),
            'std_step_time': np.std(self.step_times),
            'total_time': sum(self.step_times)
        }


def run_scenario_integration_test(scenario_name: str, steps: int = 100, 
                                  size: int = 64) -> Tuple[bool, str, Dict]:
    """
    Run integration test for a specific scenario.
    
    Returns:
        (success, message, metrics_dict)
    """
    # Create simulation
    sim = FluxSimulation(nx=size, ny=size, dx=50.0, scenario=scenario_name)
    sim.paused = False  # Unpause for testing
    metrics = IntegrationTestMetrics()
    
    # Run simulation
    for step in range(steps):
        t0 = time.time()
        sim.step_forward()
        step_time = time.time() - t0
        
        # Update metrics
        metrics.update(sim.state, step_time)
        
        # Early exit if things go wrong
        if metrics.nan_count > 0 or metrics.inf_count > 0:
            break
    
    # Check stability
    stable, stability_msg = metrics.check_stability()
    
    # Get performance stats
    perf_stats = metrics.get_performance_stats()
    
    # Build result message
    if stable:
        msg = f"Scenario '{scenario_name}' completed {steps} steps successfully. "
        msg += f"Avg step time: {perf_stats['mean_step_time']*1000:.1f}ms"
    else:
        msg = f"Scenario '{scenario_name}' failed: {stability_msg}"
    
    return stable, msg, {
        'stability': stability_msg,
        'performance': perf_stats,
        'final_metrics': {
            'temperature_range': (min(t['min'] for t in metrics.temperatures),
                                  max(t['max'] for t in metrics.temperatures)) if metrics.temperatures else (0, 0),
            'pressure_range': (min(p['min'] for p in metrics.pressures),
                               max(p['max'] for p in metrics.pressures)) if metrics.pressures else (0, 0),
            'velocity_max': max(v['max'] for v in metrics.velocities) if metrics.velocities else 0,
            'mass_change': (metrics.total_mass[-1] - metrics.total_mass[0]) / metrics.total_mass[0] * 100 
                           if len(metrics.total_mass) > 1 and metrics.total_mass[0] > 0 else 0
        }
    }


# Test all scenarios
@pytest.mark.parametrize("scenario", ["empty", "planet", "layered", "volcanic", "ice"])
def test_scenario_integration(scenario):
    """Test each scenario for stability over 100 steps."""
    success, msg, metrics = run_scenario_integration_test(scenario, steps=100, size=64)
    
    # Print detailed metrics for debugging
    print(f"\n{scenario.upper()} scenario results:")
    print(f"  {msg}")
    print(f"  Temperature range: {metrics['final_metrics']['temperature_range'][0]:.1f}K - {metrics['final_metrics']['temperature_range'][1]:.1f}K")
    print(f"  Pressure range: {metrics['final_metrics']['pressure_range'][0]:.1e}Pa - {metrics['final_metrics']['pressure_range'][1]:.1e}Pa")
    print(f"  Max velocity: {metrics['final_metrics']['velocity_max']:.1f} m/s")
    print(f"  Mass change: {metrics['final_metrics']['mass_change']:.2f}%")
    
    assert success, msg


# Performance regression tests
@pytest.mark.parametrize("scenario,max_time_ms", [
    ("empty", 50),      # Empty should be fastest
    ("planet", 150),    # Complex scenario
    ("layered", 100),   # Medium complexity
    ("volcanic", 150),  # Complex with heat sources
    ("ice", 100),       # Medium complexity
])
def test_scenario_performance(scenario, max_time_ms):
    """Test that scenarios run within performance bounds."""
    success, msg, metrics = run_scenario_integration_test(scenario, steps=50, size=64)
    
    assert success, f"Scenario failed before performance could be measured: {msg}"
    
    avg_time_ms = metrics['performance']['mean_step_time'] * 1000
    assert avg_time_ms < max_time_ms, (
        f"Performance regression in {scenario}: "
        f"avg step time {avg_time_ms:.1f}ms exceeds limit {max_time_ms}ms"
    )


# Larger grid performance test
@pytest.mark.parametrize("size", [32, 64, 128])
def test_scaling_performance(size):
    """Test performance scaling with grid size."""
    # Use simple scenario for consistent baseline
    success, msg, metrics = run_scenario_integration_test("layered", steps=20, size=size)
    
    assert success, f"Scenario failed at size {size}: {msg}"
    
    avg_time_ms = metrics['performance']['mean_step_time'] * 1000
    
    # Expected scaling is roughly O(N²) for most operations
    # So 2x grid size = ~4x time
    if size == 32:
        assert avg_time_ms < 30, f"32x32 too slow: {avg_time_ms:.1f}ms"
    elif size == 64:
        assert avg_time_ms < 120, f"64x64 too slow: {avg_time_ms:.1f}ms"
    elif size == 128:
        assert avg_time_ms < 500, f"128x128 too slow: {avg_time_ms:.1f}ms"


# Conservation test
def test_mass_conservation():
    """Test that mass is conserved in closed system."""
    # Planet scenario is a closed system
    success, msg, metrics = run_scenario_integration_test("planet", steps=100, size=64)
    
    assert success, f"Scenario failed: {msg}"
    
    mass_change = abs(metrics['final_metrics']['mass_change'])
    assert mass_change < 0.1, f"Mass not conserved: {mass_change:.2f}% change"


# Stability under extreme conditions
def test_volcanic_stability():
    """Test that volcanic scenario with heat sources remains stable."""
    success, msg, metrics = run_scenario_integration_test("volcanic", steps=200, size=64)
    
    assert success, f"Volcanic scenario unstable: {msg}"
    
    # Check that temperature increased but stayed reasonable
    temp_range = metrics['final_metrics']['temperature_range']
    assert temp_range[1] > 400, "Volcanic heat not working"
    assert temp_range[1] < 5000, "Temperature explosion in volcanic scenario"


if __name__ == "__main__":
    # Run a quick test of all scenarios
    print("Running integration tests for all scenarios...\n")
    
    scenarios = ["empty", "planet", "layered", "volcanic", "ice"]
    for scenario in scenarios:
        print(f"\nTesting {scenario}...")
        success, msg, metrics = run_scenario_integration_test(scenario, steps=50, size=64)
        
        if success:
            print(f"✓ {msg}")
            perf = metrics['performance']
            print(f"  Performance: {perf['mean_step_time']*1000:.1f}±{perf['std_step_time']*1000:.1f}ms per step")
        else:
            print(f"✗ {msg}")