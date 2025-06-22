#!/usr/bin/env python3
"""Clean test of rock stability in vacuum."""

import numpy as np
import matplotlib.pyplot as plt
from geo_game import GeoGame
from materials import MaterialType

def test_clean_rock_stability():
    """Test rock stability with all processes disabled."""
    # Create simulation without default planet
    sim = GeoGame(width=20, height=20, cell_size=50.0, setup_planet=False)
    
    # Initialize everything as space with reasonable temperature
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 273.15 + 20  # 20°C room temperature
    sim._update_material_properties()
    
    # Disable ALL material transformations
    sim.enable_material_processes = False
    sim.enable_weathering = False
    
    # Add a granite blob in the center
    center_x, center_y = 10, 10
    radius = 3
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                sim.material_types[y, x] = MaterialType.GRANITE
    
    sim._update_material_properties()
    
    # Record initial state
    initial_rock_count = np.sum(sim.material_types == MaterialType.GRANITE)
    initial_blob = sim.material_types.copy()
    
    print(f"Initial rock count: {initial_rock_count}")
    print(f"Temperature: {sim.temperature[center_y, center_x] - 273.15:.1f}°C")
    print(f"Pressure: {sim.pressure[center_y, center_x]/1e6:.3f} MPa")
    
    # Step forward and monitor
    rock_counts = [initial_rock_count]
    
    for step in range(50):
        sim.step_forward(0.1)
        rock_count = np.sum(sim.material_types == MaterialType.GRANITE)
        rock_counts.append(rock_count)
        
        if step % 10 == 0:
            print(f"Step {step}: rock count = {rock_count}")
    
    # Final check
    final_rock_count = rock_counts[-1]
    rock_change = abs(final_rock_count - initial_rock_count)
    
    print(f"\nResults:")
    print(f"  Initial rock cells: {initial_rock_count}")
    print(f"  Final rock cells: {final_rock_count}")
    print(f"  Change: {rock_change}")
    
    if rock_change == 0:
        print("  ✅ PASS: Rock blob is perfectly stable!")
    else:
        print(f"  ❌ FAIL: Lost {rock_change} rock cells")
        
        # Show what materials replaced the rock
        final_materials = np.unique(sim.material_types)
        print("\nFinal materials in simulation:")
        for mat in final_materials:
            count = np.sum(sim.material_types == mat)
            print(f"  {mat.name}: {count} cells")

if __name__ == "__main__":
    test_clean_rock_stability()