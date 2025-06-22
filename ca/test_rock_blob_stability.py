#!/usr/bin/env python3
"""Test to reproduce rock blob instability issue."""

import numpy as np
import matplotlib.pyplot as plt
from geo_game import GeoGame
from materials import MaterialType

def test_rock_blob_stability():
    """Test that a rock blob in empty space remains stable."""
    # Create simulation without default planet
    sim = GeoGame(width=40, height=40, cell_size=50.0, setup_planet=False)
    
    # Initialize everything as space with reasonable temperature
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 273.15 + 20  # 20°C
    sim._update_material_properties()
    
    # Add a granite blob in the center
    center_x, center_y = 20, 20
    radius = 5
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                sim.material_types[y, x] = MaterialType.GRANITE
    
    sim._update_material_properties()
    
    # Record initial state
    initial_rock_count = np.sum(sim.material_types == MaterialType.GRANITE)
    initial_blob = sim.material_types.copy()
    
    print(f"Initial rock count: {initial_rock_count}")
    
    # Step forward and monitor
    rock_counts = [initial_rock_count]
    max_velocities = []
    
    for step in range(100):
        sim.step_forward()
        rock_count = np.sum(sim.material_types == MaterialType.GRANITE)
        rock_counts.append(rock_count)
        
        # Track maximum velocity in rock cells
        rock_mask = sim.material_types == MaterialType.GRANITE
        if np.any(rock_mask):
            max_vel_x = np.max(np.abs(sim.velocity_x[rock_mask]))
            max_vel_y = np.max(np.abs(sim.velocity_y[rock_mask]))
            max_vel = max(max_vel_x, max_vel_y)
            max_velocities.append(max_vel)
        else:
            max_velocities.append(0)
        
        if step % 10 == 0:
            print(f"Step {step}: rock count = {rock_count}, max velocity = {max_velocities[-1]:.3f}")
    
    # Visualize results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Initial state
    ax1.imshow(initial_blob == MaterialType.GRANITE, cmap='gray')
    ax1.set_title('Initial Rock Blob')
    ax1.axis('off')
    
    # Final state
    ax2.imshow(sim.material_types == MaterialType.GRANITE, cmap='gray')
    ax2.set_title(f'Final State (after {len(rock_counts)-1} steps)')
    ax2.axis('off')
    
    # Rock count over time
    ax3.plot(rock_counts)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Rock Cell Count')
    ax3.set_title('Rock Blob Stability')
    ax3.grid(True)
    
    # Maximum velocity over time
    ax4.plot(max_velocities)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Max Velocity in Rock Cells')
    ax4.set_title('Rock Velocity (should be near zero)')
    ax4.grid(True)
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('rock_blob_stability_test.png', dpi=150)
    print(f"\nSaved visualization to rock_blob_stability_test.png")
    
    # Check for instability
    final_rock_count = rock_counts[-1]
    rock_change = abs(final_rock_count - initial_rock_count)
    max_velocity_overall = max(max_velocities) if max_velocities else 0
    
    print(f"\nResults:")
    print(f"  Rock count change: {rock_change} cells")
    print(f"  Maximum velocity: {max_velocity_overall:.3f}")
    
    if rock_change > 5 or max_velocity_overall > 0.1:
        print("  ❌ FAIL: Rock blob is unstable!")
    else:
        print("  ✅ PASS: Rock blob is stable")

if __name__ == "__main__":
    test_rock_blob_stability()