#!/usr/bin/env python3
"""Simple test to verify rocks fall through space."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FluxSimulation
from materials import MaterialType


def test_falling_rock():
    """Simple test of rock falling in uniform gravity."""
    print("=== Simple Rock Falling Test ===")
    
    # Small domain for quick test
    sim = FluxSimulation(nx=32, ny=32, dx=50.0, scenario=False)
    sim.gravity_solver.use_self_gravity = False  # Use uniform gravity
    
    # Disable other physics but ensure momentum is enabled
    sim.enable_gravity = True  # Must be enabled!
    sim.enable_momentum = True  # Must be enabled!
    sim.enable_advection = True  # Must be enabled!
    sim.enable_heat_transfer = False
    sim.enable_uranium_heating = False
    sim.enable_solar_heating = False
    sim.enable_phase_transitions = False
    sim.enable_atmospheric = False
    
    print(f"Physics enabled: gravity={sim.enable_gravity}, momentum={sim.enable_momentum}, advection={sim.enable_advection}")
    
    # Create simple scenario: rock in space
    sim.state.vol_frac.fill(0.0)
    sim.state.vol_frac[MaterialType.SPACE.value] = 1.0
    
    # Single rock cell in middle-top
    rock_y = 5  # Near top
    rock_x = sim.state.nx // 2
    sim.state.vol_frac[MaterialType.ROCK.value, rock_y, rock_x] = 1.0
    sim.state.vol_frac[MaterialType.SPACE.value, rock_y, rock_x] = 0.0
    
    # Set reasonable temperatures
    sim.state.temperature.fill(300.0)
    
    # Update properties
    sim.state.update_mixture_properties(sim.material_db)
    
    print(f"Initial rock position: ({rock_x}, {rock_y})")
    print(f"Rock density: {sim.state.density[rock_y, rock_x]:.1f} kg/m³")
    
    # Check gravity field
    gx, gy = sim.gravity_solver.solve_gravity()
    print(f"Gravity at rock: gx={gx[rock_y, rock_x]:.2f}, gy={gy[rock_y, rock_x]:.2f} m/s²")
    
    # Run for a few steps
    sim.paused = False
    positions_y = [rock_y]
    times = [0.0]
    
    for step in range(20):
        # Get rock position before step
        rock_cells = np.where(sim.state.vol_frac[MaterialType.ROCK.value] > 0.5)
        if len(rock_cells[0]) > 0:
            y_pos = np.mean(rock_cells[0])
            
            # Step simulation
            sim.step_forward()
            
            # Record new position
            rock_cells_after = np.where(sim.state.vol_frac[MaterialType.ROCK.value] > 0.5)
            if len(rock_cells_after[0]) > 0:
                y_pos_after = np.mean(rock_cells_after[0])
                positions_y.append(y_pos_after)
                times.append(sim.state.time)
                
                print(f"Step {step}: t={sim.state.time:.3f}s, y={y_pos_after:.2f}, "
                      f"moved {(y_pos_after - y_pos) * sim.state.dx:.2f} m")
                
                # Check velocity and forces
                rock_j = int(y_pos_after)
                vy = sim.state.velocity_y[rock_j, rock_x]
                vx = sim.state.velocity_x[rock_j, rock_x]
                density = sim.state.density[rock_j, rock_x]
                viscosity = sim.state.viscosity[rock_j, rock_x]
                
                print(f"  Velocity: vx={vx:.2f}, vy={vy:.2f} m/s")
                print(f"  Density: {density:.1f} kg/m³, Viscosity: {viscosity:.4f}")
                
                # Check volume fractions
                for mat_idx in range(sim.state.n_materials):
                    vol = sim.state.vol_frac[mat_idx, rock_j, rock_x]
                    if vol > 0.01:
                        mat_name = MaterialType(mat_idx).name
                        print(f"  {mat_name}: {vol:.2f}")
            else:
                print(f"Step {step}: Rock disappeared!")
                break
        else:
            print(f"Step {step}: Can't find rock!")
            break
    
    # Check if rock moved down
    if len(positions_y) > 2:
        total_movement = (positions_y[-1] - positions_y[0]) * sim.state.dx
        print(f"\nTotal movement: {total_movement:.1f} m in {times[-1]:.2f} s")
        
        if total_movement > 10.0:  # Moved at least 10m down
            print("✓ PASS: Rock fell through space!")
            return True
        else:
            print("✗ FAIL: Rock didn't fall significantly")
            return False
    else:
        print("✗ FAIL: Not enough data collected")
        return False


if __name__ == "__main__":
    success = test_falling_rock()
    
    if success:
        print("\nRocks CAN fall through space! You can test visually with:")
        print("  python main.py --scenario falling_rock")
    else:
        print("\nSomething is still preventing rocks from falling.")
        print("Check that:")
        print("  1. Gravity is applied in space regions")
        print("  2. Velocities aren't zeroed in space") 
        print("  3. Space material can be advected")