#!/usr/bin/env python3
"""Debug NaN issue at startup."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from materials import MaterialType

def test_nan_simple():
    """Test with simplest possible scenario."""
    print("=== NaN Debug Test ===")
    
    # Very small grid for debugging
    sim = FluxSimulation(nx=8, ny=8, dx=50.0, scenario=False)
    
    # Disable everything except gravity and momentum
    sim.enable_gravity = True
    sim.enable_momentum = True
    sim.enable_advection = False
    sim.enable_heat_transfer = False
    sim.enable_uranium_heating = False
    sim.enable_solar_heating = False
    sim.enable_phase_transitions = False
    sim.enable_atmospheric = False
    
    # Simple scenario: all rock
    sim.state.vol_frac.fill(0.0)
    sim.state.vol_frac[MaterialType.ROCK.value] = 1.0
    sim.state.temperature.fill(300.0)
    
    # Update properties
    sim.state.update_mixture_properties(sim.material_db)
    
    print(f"Initial state:")
    print(f"  Density min/max: {sim.state.density.min():.1f} / {sim.state.density.max():.1f}")
    print(f"  Velocity_x min/max: {sim.state.velocity_x.min():.3f} / {sim.state.velocity_x.max():.3f}")
    print(f"  Velocity_y min/max: {sim.state.velocity_y.min():.3f} / {sim.state.velocity_y.max():.3f}")
    
    # Run initialization
    print("\nRunning initialization...")
    sim._solve_initial_state()
    
    print(f"\nAfter initialization:")
    print(f"  Velocity_x min/max: {sim.state.velocity_x.min():.3f} / {sim.state.velocity_x.max():.3f}")
    print(f"  Velocity_y min/max: {sim.state.velocity_y.min():.3f} / {sim.state.velocity_y.max():.3f}")
    print(f"  Pressure min/max: {sim.state.pressure.min():.1f} / {sim.state.pressure.max():.1f}")
    
    # Check for NaN
    if np.any(np.isnan(sim.state.velocity_x)) or np.any(np.isnan(sim.state.velocity_y)):
        print("\nERROR: NaN detected in velocities!")
        return False
    else:
        print("\nNo NaN detected - initialization successful")
        return True

def test_nan_planet():
    """Test with planet scenario."""
    print("\n=== NaN Debug Test - Planet Scenario ===")
    
    # Small grid
    sim = FluxSimulation(nx=16, ny=16, dx=50.0, scenario="planet")
    
    print(f"Initial state after scenario setup:")
    print(f"  Density min/max: {sim.state.density.min():.1f} / {sim.state.density.max():.1f}")
    print(f"  Has space: {np.any(sim.state.vol_frac[MaterialType.SPACE.value] > 0.5)}")
    print(f"  Has rock: {np.any(sim.state.vol_frac[MaterialType.ROCK.value] > 0.5)}")
    
    # Check for extreme densities
    very_low_density = sim.state.density < 0.01
    if np.any(very_low_density):
        print(f"  Cells with very low density: {np.sum(very_low_density)}")
        
    return True

if __name__ == "__main__":
    test_nan_simple()
    test_nan_planet()