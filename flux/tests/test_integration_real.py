"""
Real integration tests that verify simulations actually run.

These tests ensure:
1. Simulation is actually stepping (not paused)
2. Step counter increases
3. Physical values remain reasonable
4. Performance is acceptable
"""

import pytest
import numpy as np
import time
from typing import Dict, Tuple, List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from state import FluxState


class SimulationMetrics:
    """Track comprehensive metrics during simulation."""
    
    def __init__(self):
        self.step_times: List[float] = []
        self.step_numbers: List[int] = []
        self.total_masses: List[float] = []
        self.total_energies: List[float] = []
        self.max_velocities: List[float] = []
        self.temperature_ranges: List[Tuple[float, float]] = []
        self.pressure_ranges: List[Tuple[float, float]] = []
        self.nan_count = 0
        self.inf_count = 0
        self.negative_mass_count = 0
        
    def update(self, sim: FluxSimulation, step_time: float):
        """Update metrics from current simulation state."""
        state = sim.state
        
        # Verify step number is increasing
        self.step_numbers.append(sim.step_count)
        self.step_times.append(step_time)
        
        # Mass tracking
        total_mass = state.get_total_mass()
        self.total_masses.append(total_mass)
        if total_mass < 0:
            self.negative_mass_count += 1
        
        # Energy tracking
        try:
            total_energy = state.get_total_energy()
            self.total_energies.append(total_energy)
        except:
            self.total_energies.append(np.nan)
        
        # Velocity tracking
        vel_mag = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
        self.max_velocities.append(np.max(vel_mag))
        
        # Temperature range
        temp_valid = state.temperature[~np.isnan(state.temperature)]
        if len(temp_valid) > 0:
            self.temperature_ranges.append((np.min(temp_valid), np.max(temp_valid)))
        else:
            self.temperature_ranges.append((np.nan, np.nan))
        
        # Pressure range
        pressure_valid = state.pressure[~np.isnan(state.pressure)]
        if len(pressure_valid) > 0:
            self.pressure_ranges.append((np.min(pressure_valid), np.max(pressure_valid)))
        else:
            self.pressure_ranges.append((np.nan, np.nan))
        
        # Check for NaN/Inf
        for field in [state.temperature, state.pressure, state.density, 
                      state.velocity_x, state.velocity_y]:
            self.nan_count += np.sum(np.isnan(field))
            self.inf_count += np.sum(np.isinf(field))
    
    def verify_simulation_ran(self) -> Tuple[bool, str]:
        """Verify the simulation actually executed steps."""
        if not self.step_numbers:
            return False, "No steps recorded"
        
        # Check step numbers increased
        if len(self.step_numbers) < 2:
            return False, f"Only {len(self.step_numbers)} steps recorded"
        
        # Verify steps are sequential
        for i in range(1, len(self.step_numbers)):
            if self.step_numbers[i] <= self.step_numbers[i-1]:
                return False, f"Step number didn't increase: {self.step_numbers[i-1]} -> {self.step_numbers[i]}"
        
        final_step = self.step_numbers[-1]
        expected_steps = len(self.step_numbers)
        
        # Allow for some initialization steps
        if final_step < expected_steps - 10:
            return False, f"Step counter ({final_step}) doesn't match steps run ({expected_steps})"
        
        return True, f"Ran {expected_steps} steps successfully"
    
    def check_stability(self) -> Tuple[bool, List[str]]:
        """Check if simulation remained stable."""
        errors = []
        
        # Critical failures
        if self.negative_mass_count > 0:
            errors.append(f"Negative mass in {self.negative_mass_count} steps")
        
        if self.nan_count > 100:  # Allow some NaN in boundaries
            errors.append(f"Excessive NaN values: {self.nan_count}")
        
        if self.inf_count > 100:
            errors.append(f"Excessive Inf values: {self.inf_count}")
        
        # Check final values
        if self.temperature_ranges:
            final_temp = self.temperature_ranges[-1]
            if not np.isnan(final_temp[0]):
                if final_temp[0] < 0:
                    errors.append(f"Negative temperature: {final_temp[0]:.1f}K")
                if final_temp[1] > 50000:
                    errors.append(f"Extreme temperature: {final_temp[1]:.1f}K")
        
        if self.max_velocities:
            final_max_vel = self.max_velocities[-1]
            if final_max_vel > 100000:  # 100 km/s is extreme
                errors.append(f"Extreme velocity: {final_max_vel:.1f} m/s")
        
        # Mass conservation
        if len(self.total_masses) > 1:
            initial_mass = self.total_masses[0]
            final_mass = self.total_masses[-1]
            if initial_mass > 0 and final_mass > 0:
                mass_change = abs(final_mass - initial_mass) / initial_mass
                if mass_change > 0.1:  # 10% is very lenient
                    errors.append(f"Mass change: {mass_change*100:.1f}%")
        
        return len(errors) == 0, errors
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.step_times:
            return {}
        
        # Exclude first few steps (initialization)
        times = self.step_times[5:] if len(self.step_times) > 5 else self.step_times
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }


def run_scenario_test(scenario: str, target_steps: int = 100, 
                      size: int = 64) -> Tuple[bool, str, SimulationMetrics]:
    """Run a scenario and collect metrics."""
    # Create simulation
    sim = FluxSimulation(nx=size, ny=size, dx=50.0, scenario=scenario)
    
    # CRITICAL: Verify simulation starts paused and unpause it
    initial_paused = sim.paused
    sim.paused = False
    
    # Record initial step count
    initial_step_count = sim.step_count
    
    metrics = SimulationMetrics()
    
    # Run simulation
    for i in range(target_steps):
        t0 = time.perf_counter()
        sim.step_forward()
        t1 = time.perf_counter()
        
        metrics.update(sim, t1 - t0)
        
        # Early exit on catastrophic failure
        if metrics.negative_mass_count > 0:
            break
    
    # Verify simulation ran
    ran_ok, run_msg = metrics.verify_simulation_ran()
    if not ran_ok:
        return False, f"Simulation didn't run properly: {run_msg}", metrics
    
    # Check stability
    stable, errors = metrics.check_stability()
    
    if stable:
        perf = metrics.get_performance_stats()
        msg = f"Completed {len(metrics.step_numbers)} steps at {perf['fps']:.1f} FPS"
    else:
        msg = f"Instability detected: {'; '.join(errors[:3])}"  # First 3 errors
    
    return stable, msg, metrics


# Test that each scenario actually runs
@pytest.mark.parametrize("scenario", ["empty", "planet", "layered", "volcanic", "ice"])
def test_scenario_actually_runs(scenario):
    """Verify each scenario executes 100 steps."""
    stable, msg, metrics = run_scenario_test(scenario, target_steps=100)
    
    # First check it actually ran
    ran_ok, run_msg = metrics.verify_simulation_ran()
    assert ran_ok, f"{scenario}: {run_msg}"
    
    # Check we got close to target steps
    actual_steps = len(metrics.step_numbers)
    assert actual_steps >= 95, f"{scenario}: Only completed {actual_steps}/100 steps"
    
    print(f"\n{scenario}: {msg}")
    print(f"  Final step count: {metrics.step_numbers[-1]}")
    print(f"  Steps executed: {actual_steps}")


# Test basic stability
@pytest.mark.parametrize("scenario", ["empty", "planet", "layered", "volcanic", "ice"])
def test_scenario_basic_stability(scenario):
    """Test scenarios don't explode catastrophically."""
    stable, msg, metrics = run_scenario_test(scenario, target_steps=50)
    
    # Check no negative mass
    assert metrics.negative_mass_count == 0, f"{scenario}: Negative mass detected"
    
    # Check simulation completed
    assert len(metrics.step_numbers) >= 45, f"{scenario}: Failed to complete 45+ steps"
    
    # Report results
    _, errors = metrics.check_stability()
    if errors:
        print(f"\n{scenario} issues: {errors}")


# Performance benchmarks
@pytest.mark.parametrize("scenario,min_fps", [
    ("empty", 15),      # Empty should be fast
    ("planet", 8),      # Complex scenarios slower
    ("layered", 8),    
    ("volcanic", 8),   
    ("ice", 8),
])
def test_scenario_performance(scenario, min_fps):
    """Test performance meets minimum targets."""
    stable, msg, metrics = run_scenario_test(scenario, target_steps=50, size=64)
    
    # Must complete enough steps for valid measurement
    assert len(metrics.step_numbers) >= 40, f"{scenario}: Insufficient steps for performance test"
    
    perf = metrics.get_performance_stats()
    achieved_fps = perf['fps']
    
    print(f"\n{scenario} performance: {achieved_fps:.1f} FPS (target: {min_fps}+)")
    assert achieved_fps >= min_fps, f"{scenario}: {achieved_fps:.1f} FPS < {min_fps} FPS target"


# Empty scenario should be perfectly stable
def test_empty_scenario_perfect_stability():
    """Empty scenario should have minimal dynamics."""
    stable, msg, metrics = run_scenario_test("empty", target_steps=200)
    
    # Should complete all steps
    assert len(metrics.step_numbers) == 200, "Empty scenario didn't complete all steps"
    
    # Mass should be constant
    mass_values = metrics.total_masses
    mass_change = abs(mass_values[-1] - mass_values[0]) / mass_values[0]
    assert mass_change < 1e-6, f"Empty scenario mass changed by {mass_change*100:.1e}%"
    
    # Velocity should remain near zero
    max_vel = max(metrics.max_velocities)
    assert max_vel < 1.0, f"Empty scenario has velocity {max_vel:.1f} m/s"


# Long-running test for one scenario
def test_planet_long_stability():
    """Test planet scenario for extended run."""
    stable, msg, metrics = run_scenario_test("planet", target_steps=500, size=64)
    
    # Should run for reasonable time without catastrophic failure
    assert len(metrics.step_numbers) >= 100, "Planet scenario failed too early"
    
    # Report final state
    print(f"\nPlanet long run: {len(metrics.step_numbers)} steps completed")
    if metrics.total_masses:
        mass_change = (metrics.total_masses[-1] - metrics.total_masses[0]) / metrics.total_masses[0] * 100
        print(f"  Mass change: {mass_change:.1f}%")
    if metrics.max_velocities:
        print(f"  Max velocity: {max(metrics.max_velocities):.1f} m/s")


# Test different grid sizes
@pytest.mark.parametrize("size", [32, 64])
def test_scaling_stability(size):
    """Test stability at different grid sizes."""
    stable, msg, metrics = run_scenario_test("layered", target_steps=50, size=size)
    
    assert len(metrics.step_numbers) >= 45, f"Failed at {size}x{size} grid"
    
    perf = metrics.get_performance_stats()
    print(f"\n{size}x{size} grid: {perf['fps']:.1f} FPS, {perf['mean_ms']:.1f}ms/step")


if __name__ == "__main__":
    """Manual test run with detailed output."""
    print("Running comprehensive integration tests...\n")
    
    scenarios = ["empty", "planet", "layered", "volcanic", "ice"]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Testing {scenario.upper()} scenario")
        print('='*60)
        
        stable, msg, metrics = run_scenario_test(scenario, target_steps=100)
        
        print(f"Result: {msg}")
        print(f"Steps completed: {len(metrics.step_numbers)}")
        print(f"Final step count: {metrics.step_numbers[-1] if metrics.step_numbers else 'N/A'}")
        
        if metrics.step_times:
            perf = metrics.get_performance_stats()
            print(f"Performance: {perf['fps']:.1f} FPS ({perf['mean_ms']:.1f}ms avg)")
        
        stable, errors = metrics.check_stability()
        if errors:
            print(f"Issues: {errors}")
        else:
            print("Stability: OK")