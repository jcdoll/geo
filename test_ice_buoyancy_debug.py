#!/usr/bin/env python3
"""
Debug script for ice buoyancy issue.
Tests whether ice floats properly in water.
"""

import numpy as np
import matplotlib.pyplot as plt
from geo_game import GeoGame
from materials import MaterialType

def test_ice_buoyancy():
    """Test ice floating in water with detailed diagnostics."""
    
    # Create simulation
    sim = GeoGame(width=40, height=30, cell_size=1.0)  # Small cells for detail
    sim.dt = 0.01  # Small timestep
    sim.enable_pressure = True
    sim.enable_self_gravity = False  # Use external gravity only
    sim.external_gravity = (0.0, 9.81)  # Standard Earth gravity
    
    # Fill bottom half with water
    water_level = 20
    sim.material_types[:water_level, :] = MaterialType.WATER
    sim.temperature[:water_level, :] = 280.0  # 7°C water
    
    # Fill top with air
    sim.material_types[water_level:, :] = MaterialType.AIR
    sim.temperature[water_level:, :] = 280.0
    
    # Place ice cube in middle of water
    ice_y1, ice_y2 = 10, 15  # 5x5 ice cube
    ice_x1, ice_x2 = 17, 22
    sim.material_types[ice_y1:ice_y2, ice_x1:ice_x2] = MaterialType.ICE
    sim.temperature[ice_y1:ice_y2, ice_x1:ice_x2] = 270.0  # -3°C ice
    
    # Update properties
    sim._properties_dirty = True
    sim._update_material_properties()
    
    print("Initial setup:")
    print(f"Water density: {sim.material_db.properties[MaterialType.WATER].density} kg/m³")
    print(f"Ice density: {sim.material_db.properties[MaterialType.ICE].density} kg/m³")
    print(f"Buoyancy ratio: {sim.material_db.properties[MaterialType.ICE].density / sim.material_db.properties[MaterialType.WATER].density:.3f}")
    print(f"Expected: Ice should float with {(1 - 920/1000)*100:.1f}% above water\n")
    
    # Track ice position over time
    ice_positions = []
    pressure_gradients = []
    forces = []
    
    for step in range(100):
        # Find ice cells
        ice_mask = (sim.material_types == MaterialType.ICE)
        if not np.any(ice_mask):
            print(f"Ice disappeared at step {step}!")
            break
            
        # Calculate center of mass of ice
        ys, xs = np.where(ice_mask)
        ice_com_y = np.mean(ys)
        ice_com_x = np.mean(xs)
        ice_positions.append((step * sim.dt, ice_com_y))
        
        # Sample pressure and forces at ice center
        cy, cx = int(ice_com_y), int(ice_com_x)
        if 1 <= cy < sim.height-1 and 1 <= cx < sim.width-1:
            # Pressure gradient
            P = sim.pressure * 1e6  # MPa to Pa
            dP_dy = (P[cy+1, cx] - P[cy-1, cx]) / (2 * sim.cell_size)
            pressure_gradients.append(dP_dy)
            
            # Net force on ice
            if hasattr(sim, 'force_y'):
                forces.append(sim.force_y[cy, cx])
            else:
                forces.append(0)
        
        # Step simulation
        sim.step_forward()
        
        # Print diagnostics every 10 steps
        if step % 10 == 0:
            print(f"Step {step}: Ice COM at y={ice_com_y:.2f}")
            if len(pressure_gradients) > 0:
                print(f"  Pressure gradient dP/dy: {pressure_gradients[-1]:.1f} Pa/m")
                print(f"  Force density: {forces[-1]:.1f} N/m³")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Ice position over time
    times = [t for t, y in ice_positions]
    ys = [y for t, y in ice_positions]
    ax1.plot(times, ys, 'b-', label='Ice COM Y position')
    ax1.axhline(y=water_level, color='cyan', linestyle='--', label='Water surface')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Y Position (cells)')
    ax1.set_title('Ice Position Over Time')
    ax1.legend()
    ax1.grid(True)
    ax1.invert_yaxis()  # Y increases downward
    
    # Pressure gradient
    if len(pressure_gradients) > 0:
        ax2.plot(times[:len(pressure_gradients)], pressure_gradients, 'r-')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('dP/dy (Pa/m)')
        ax2.set_title('Vertical Pressure Gradient at Ice Location')
        ax2.grid(True)
    
    # Force density
    if len(forces) > 0:
        ax3.plot(times[:len(forces)], forces, 'g-')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Force Density (N/m³)')
        ax3.set_title('Net Vertical Force on Ice')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('ice_buoyancy_debug.png', dpi=150)
    print(f"\nSaved diagnostic plot to ice_buoyancy_debug.png")
    
    # Final analysis
    if len(ys) > 10:
        initial_y = ys[0]
        final_y = ys[-1]
        if final_y < initial_y - 2:
            print("\n✓ SUCCESS: Ice floated upward!")
        elif final_y > initial_y + 2:
            print("\n✗ FAILURE: Ice sank downward!")
        else:
            print("\n? UNCLEAR: Ice didn't move significantly")
            
        # Expected equilibrium position
        ice_density = sim.material_db.properties[MaterialType.ICE].density
        water_density = sim.material_db.properties[MaterialType.WATER].density
        submerged_fraction = ice_density / water_density
        ice_height = ice_y2 - ice_y1
        expected_y = water_level - ice_height * (1 - submerged_fraction)
        print(f"\nExpected equilibrium: Ice top at y={expected_y:.1f} (submerged fraction={submerged_fraction:.3f})")
        print(f"Actual final position: Ice COM at y={final_y:.1f}")

if __name__ == "__main__":
    test_ice_buoyancy()