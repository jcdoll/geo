#!/usr/bin/env python3
"""Debug test to understand why rocks don't fall through space."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from materials import MaterialType


def test_debug_falling():
    """Debug test with detailed output."""
    print("=== Debug Rock Falling Test ===")
    
    # Small domain for quick test
    sim = FluxSimulation(nx=16, ny=16, dx=50.0, scenario=False)
    sim.gravity_solver.use_self_gravity = False  # Use uniform gravity
    
    # Enable only gravity and momentum
    sim.enable_gravity = True
    sim.enable_momentum = True
    sim.enable_advection = True
    sim.enable_heat_transfer = False
    sim.enable_uranium_heating = False
    sim.enable_solar_heating = False
    sim.enable_phase_transitions = False
    sim.enable_atmospheric = False
    
    # Create simple scenario: single rock cell in space
    sim.state.vol_frac.fill(0.0)
    sim.state.vol_frac[MaterialType.SPACE.value] = 1.0
    
    # Single rock cell
    rock_y = 3
    rock_x = 8
    sim.state.vol_frac[MaterialType.ROCK.value, rock_y, rock_x] = 1.0
    sim.state.vol_frac[MaterialType.SPACE.value, rock_y, rock_x] = 0.0
    
    # Set temperatures
    sim.state.temperature.fill(300.0)
    
    # Update properties
    sim.state.update_mixture_properties(sim.material_db)
    
    print(f"\nInitial state:")
    print(f"Rock at ({rock_x}, {rock_y})")
    print(f"Rock density: {sim.state.density[rock_y, rock_x]:.1f} kg/m³")
    print(f"Rock viscosity: {sim.state.viscosity[rock_y, rock_x]:.4f}")
    
    # Check gravity
    gx, gy = sim.gravity_solver.solve_gravity()
    print(f"\nGravity field:")
    print(f"  At rock: gx={gx[rock_y, rock_x]:.2f}, gy={gy[rock_y, rock_x]:.2f} m/s²")
    print(f"  Below rock: gy={gy[rock_y+1, rock_x]:.2f} m/s²")
    
    # Run ONE step with detailed debugging
    sim.paused = False
    
    print(f"\n=== STEP 1 ===")
    
    # Calculate dt before step
    dt = sim.physics.apply_cfl_limit()
    print(f"CFL dt = {dt:.3f} s")
    
    # Before step
    print(f"\nBefore step:")
    print(f"  velocity_y[{rock_y},{rock_x}] = {sim.state.velocity_y[rock_y, rock_x]:.3f} m/s")
    print(f"  velocity_y_face[{rock_y},{rock_x}] = {sim.state.velocity_y_face[rock_y, rock_x]:.3f} m/s")
    print(f"  velocity_y_face[{rock_y+1},{rock_x}] = {sim.state.velocity_y_face[rock_y+1, rock_x]:.3f} m/s")
    
    # Step forward
    print(f"\nStepping forward with dt={dt:.3f}...")
    
    # Check volume fractions before advection
    rock_before = sim.state.vol_frac[MaterialType.ROCK.value, rock_y, rock_x]
    space_before = sim.state.vol_frac[MaterialType.SPACE.value, rock_y, rock_x]
    rock_below_before = sim.state.vol_frac[MaterialType.ROCK.value, rock_y+1, rock_x]
    
    sim.step_forward()
    
    # Check volume fractions after advection
    rock_after = sim.state.vol_frac[MaterialType.ROCK.value, rock_y, rock_x]
    space_after = sim.state.vol_frac[MaterialType.SPACE.value, rock_y, rock_x]
    rock_below_after = sim.state.vol_frac[MaterialType.ROCK.value, rock_y+1, rock_x]
    
    print(f"\nVolume fraction changes:")
    print(f"  Rock at ({rock_x},{rock_y}): {rock_before:.3f} -> {rock_after:.3f} (delta={rock_after-rock_before:.3f})")
    print(f"  Space at ({rock_x},{rock_y}): {space_before:.3f} -> {space_after:.3f} (delta={space_after-space_before:.3f})")
    print(f"  Rock below ({rock_x},{rock_y+1}): {rock_below_before:.3f} -> {rock_below_after:.3f} (delta={rock_below_after-rock_below_before:.3f})")
    
    # After step
    print(f"\nAfter step:")
    print(f"  velocity_y[{rock_y},{rock_x}] = {sim.state.velocity_y[rock_y, rock_x]:.3f} m/s")
    print(f"  velocity_y_face[{rock_y},{rock_x}] = {sim.state.velocity_y_face[rock_y, rock_x]:.3f} m/s")
    print(f"  velocity_y_face[{rock_y+1},{rock_x}] = {sim.state.velocity_y_face[rock_y+1, rock_x]:.3f} m/s")
    
    # Check divergence
    div = sim.pressure_solver._compute_divergence()
    print(f"\nDivergence at rock: {div[rock_y, rock_x]:.6f}")
    print(f"Divergence below: {div[rock_y+1, rock_x]:.6f}")
    
    # Check beta coefficients
    print(f"\nBeta coefficients:")
    print(f"  beta_y[{rock_y},{rock_x}] = {sim.state.beta_y[rock_y, rock_x]:.6f}")
    print(f"  beta_y[{rock_y+1},{rock_x}] = {sim.state.beta_y[rock_y+1, rock_x]:.6f}")
    
    # Check where rock moved
    rock_cells = np.where(sim.state.vol_frac[MaterialType.ROCK.value] > 0.5)
    if len(rock_cells[0]) > 0:
        new_y = rock_cells[0][0]
        new_x = rock_cells[1][0]
        print(f"\nRock now at ({new_x}, {new_y})")
        if new_y > rock_y:
            print("✓ Rock moved down!")
        else:
            print("✗ Rock didn't move down")
    else:
        print("\n✗ Rock disappeared or dispersed")
        
    # Check volume fractions
    print(f"\nVolume fractions at original position ({rock_x}, {rock_y}):")
    for mat_idx in range(sim.state.n_materials):
        vol = sim.state.vol_frac[mat_idx, rock_y, rock_x]
        if vol > 0.01:
            mat_name = MaterialType(mat_idx).name
            print(f"  {mat_name}: {vol:.3f}")


if __name__ == "__main__":
    test_debug_falling()