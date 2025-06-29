"""
Test whether gravity ramping is necessary for stable initialization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from simulation import FluxSimulation
from scenarios import setup_scenario
from state import FluxState
from materials import MaterialDatabase
from gravity_solver import GravitySolver, SolverMethod
from physics import FluxPhysics
from pressure_solver import PressureSolver


def run_initialization_test(use_ramp: bool, scenario: str = "planet", verbose: bool = True):
    """Test initialization with and without gravity ramping."""
    
    # Create state and setup scenario
    nx, ny = 64, 64
    dx = 50.0
    state = FluxState(nx, ny, dx)
    material_db = MaterialDatabase()
    setup_scenario(scenario, state, material_db)
    
    # Create solvers
    gravity_solver = GravitySolver(state, method=SolverMethod.MULTIGRID)
    physics = FluxPhysics(state)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {scenario} initialization {'WITH' if use_ramp else 'WITHOUT'} gravity ramp")
        print('='*60)
    
    # Initial state
    initial_mass = state.get_total_mass()
    initial_max_density = np.max(state.density)
    
    if verbose:
        print(f"Initial state:")
        print(f"  Total mass: {initial_mass:.3e} kg")
        print(f"  Max density: {initial_max_density:.1f} kg/m³")
        print(f"  Density range: [{np.min(state.density):.3f}, {np.max(state.density):.1f}] kg/m³")
    
    # Step 1: Solve gravity field
    gx, gy = gravity_solver.solve_gravity()
    state.gravity_x[:] = gx
    state.gravity_y[:] = gy
    
    # Step 2: Initialize pressure
    if use_ramp:
        # Gradual initialization with ramping
        n_init_steps = 10
        dt_init = 0.001
        
        for i in range(n_init_steps):
            ramp_factor = (i + 1) / n_init_steps
            gx_ramped = gx * ramp_factor
            gy_ramped = gy * ramp_factor
            
            physics.update_momentum(gx_ramped, gy_ramped, dt_init)
            state.update_face_coefficients()
            
            if verbose and i % 3 == 0:
                vel_mag = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
                print(f"  Ramp step {i+1}: max velocity = {np.max(vel_mag):.1f} m/s")
    else:
        # Direct initialization
        dt_init = 0.01  # Standard small timestep
        physics.update_momentum(gx, gy, dt_init)
        state.update_face_coefficients()
    
    # Check final state
    final_mass = state.get_total_mass()
    vel_mag = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
    max_velocity = np.max(vel_mag)
    
    # Find location of max velocity
    y_max, x_max = np.unravel_index(np.argmax(vel_mag), vel_mag.shape)
    density_at_max_vel = state.density[y_max, x_max]
    
    if verbose:
        print(f"\nFinal state after initialization:")
        print(f"  Total mass: {final_mass:.3e} kg")
        print(f"  Mass change: {(final_mass - initial_mass)/initial_mass * 100:.3f}%")
        print(f"  Max velocity: {max_velocity:.1f} m/s")
        print(f"  Location of max velocity: ({x_max}, {y_max})")
        print(f"  Density at max velocity: {density_at_max_vel:.3f} kg/m³")
        print(f"  Pressure range: [{np.min(state.pressure):.1e}, {np.max(state.pressure):.1e}] Pa")
    
    # Check for instabilities
    is_stable = True
    issues = []
    
    if max_velocity > 1000:  # 1 km/s is extreme
        is_stable = False
        issues.append(f"Extreme velocity: {max_velocity:.1f} m/s")
    
    if final_mass < 0:
        is_stable = False
        issues.append("Negative mass")
    
    if np.any(np.isnan(state.pressure)):
        is_stable = False
        issues.append("NaN in pressure")
    
    # Check pressure at boundaries
    space_mask = state.density < 0.1
    if np.any(space_mask):
        pressure_in_space = state.pressure[space_mask]
        if np.any(pressure_in_space < -1e9):
            issues.append(f"Large negative pressure in space: {np.min(pressure_in_space):.1e} Pa")
    
    return {
        'stable': is_stable,
        'max_velocity': max_velocity,
        'mass_change': (final_mass - initial_mass) / initial_mass * 100,
        'issues': issues,
        'final_mass': final_mass
    }


def compare_methods():
    """Compare initialization with and without ramping."""
    scenarios = ["planet", "layered", "volcanic"]
    
    print("\nCOMPARISON OF INITIALIZATION METHODS")
    print("="*80)
    
    for scenario in scenarios:
        # Test without ramp
        result_no_ramp = run_initialization_test(False, scenario, verbose=False)
        
        # Test with ramp
        result_ramp = run_initialization_test(True, scenario, verbose=False)
        
        print(f"\n{scenario.upper()}:")
        print(f"  Without ramp: max_vel={result_no_ramp['max_velocity']:.1f} m/s, "
              f"mass_change={result_no_ramp['mass_change']:.3f}%, "
              f"stable={result_no_ramp['stable']}")
        if result_no_ramp['issues']:
            print(f"    Issues: {result_no_ramp['issues']}")
        
        print(f"  With ramp:    max_vel={result_ramp['max_velocity']:.1f} m/s, "
              f"mass_change={result_ramp['mass_change']:.3f}%, "
              f"stable={result_ramp['stable']}")
        if result_ramp['issues']:
            print(f"    Issues: {result_ramp['issues']}")


def test_single_scenario_detail():
    """Detailed test of planet scenario."""
    # First without ramp
    print("\nDETAILED TEST: Planet scenario")
    result_no_ramp = run_initialization_test(False, "planet", verbose=True)
    
    # Then with ramp
    result_ramp = run_initialization_test(True, "planet", verbose=True)
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Without ramp: {'STABLE' if result_no_ramp['stable'] else 'UNSTABLE'}")
    print(f"With ramp:    {'STABLE' if result_ramp['stable'] else 'UNSTABLE'}")
    
    # Test if we can run a few steps after initialization
    print("\nTesting post-initialization stability (5 steps)...")
    
    for use_ramp in [False, True]:
        sim = FluxSimulation(nx=64, ny=64, dx=50.0, scenario="planet", init_gravity_ramp=use_ramp)
        sim.paused = False
        
        try:
            for i in range(5):
                sim.step_forward()
            vel_mag = np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2)
            print(f"  {'With' if use_ramp else 'Without'} ramp after 5 steps: max_vel={np.max(vel_mag):.1f} m/s")
        except Exception as e:
            print(f"  {'With' if use_ramp else 'Without'} ramp: FAILED - {str(e)}")


if __name__ == "__main__":
    # Run comparison
    compare_methods()
    
    # Run detailed test
    test_single_scenario_detail()