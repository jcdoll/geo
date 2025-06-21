#!/usr/bin/env python3
"""Test that rigid body movement doesn't destroy internal structures.

This test creates a scenario similar to the planet with uranium core
inside magma inside basalt shell, and verifies that the core remains
intact when forces are applied.
"""

import numpy as np
import sys

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from geo_game import GeoGame
from materials import MaterialType


def create_nested_structure(sim, center_x, center_y):
    """Create a nested structure: core inside fluid inside shell."""
    # Outer shell (basalt)
    for y in range(center_y - 10, center_y + 11):
        for x in range(center_x - 10, center_x + 11):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if 8 < dist <= 10:
                if 0 <= x < sim.width and 0 <= y < sim.height:
                    sim.material_types[y, x] = MaterialType.BASALT
                    sim.temperature[y, x] = 300.0
    
    # Middle fluid layer (magma)
    for y in range(center_y - 8, center_y + 9):
        for x in range(center_x - 8, center_x + 9):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if 4 < dist <= 8:
                if 0 <= x < sim.width and 0 <= y < sim.height:
                    sim.material_types[y, x] = MaterialType.MAGMA
                    sim.temperature[y, x] = 1500.0
    
    # Inner core (uranium)
    for y in range(center_y - 4, center_y + 5):
        for x in range(center_x - 4, center_x + 5):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= 4:
                if 0 <= x < sim.width and 0 <= y < sim.height:
                    sim.material_types[y, x] = MaterialType.URANIUM
                    sim.temperature[y, x] = 1000.0


def count_uranium_cells(sim):
    """Count the number of uranium cells in the simulation."""
    return np.sum(sim.material_types == MaterialType.URANIUM)


def verify_uranium_connectivity(sim, expected_center_x, expected_center_y):
    """Verify uranium core is still connected and near expected position."""
    uranium_mask = sim.material_types == MaterialType.URANIUM
    uranium_coords = np.where(uranium_mask)
    
    if len(uranium_coords[0]) == 0:
        return False, "No uranium cells found"
    
    # Calculate center of uranium mass
    center_y = np.mean(uranium_coords[0])
    center_x = np.mean(uranium_coords[1])
    
    # Check if uranium has moved significantly from expected position
    # Allow some movement but not too much
    dist_moved = np.sqrt((center_x - expected_center_x)**2 + (center_y - expected_center_y)**2)
    if dist_moved > 15:  # More than 15 cells away
        return False, f"Uranium core moved too far: {dist_moved:.1f} cells"
    
    # Check connectivity using scipy
    from scipy import ndimage
    labels, num_components = ndimage.label(uranium_mask)
    
    if num_components != 1:
        return False, f"Uranium core fragmented into {num_components} pieces"
    
    return True, f"Uranium core intact at ({center_x:.1f}, {center_y:.1f})"


def main():
    """Run the test."""
    print("Testing rigid body movement with nested structures...")
    
    # Create simulation
    sim = GeoGame(80, 80, setup_planet=False)
    sim.external_gravity = (0, 9.81)  # Earth-like gravity
    sim.debug_rigid_bodies = True  # Enable debug output
    
    # Create nested structure in center
    center_x, center_y = 40, 40
    create_nested_structure(sim, center_x, center_y)
    
    # Update material properties
    sim._update_material_properties()
    
    # Count initial uranium cells
    initial_uranium_count = count_uranium_cells(sim)
    print(f"Initial uranium cells: {initial_uranium_count}")
    
    # Verify initial structure
    connected, msg = verify_uranium_connectivity(sim, center_x, center_y)
    print(f"Initial structure: {msg}")
    
    # Check rigid body labeling
    from scipy import ndimage
    rigid_labels, num_groups = sim.fluid_dynamics.identify_rigid_groups()
    print(f"\nRigid body groups identified: {num_groups}")
    
    # Check if uranium and basalt are in same group
    uranium_mask = sim.material_types == MaterialType.URANIUM
    basalt_mask = sim.material_types == MaterialType.BASALT
    
    if np.any(uranium_mask) and np.any(basalt_mask):
        uranium_labels = rigid_labels[uranium_mask]
        basalt_labels = rigid_labels[basalt_mask]
        print(f"Uranium labels: {np.unique(uranium_labels)}")
        print(f"Basalt labels: {np.unique(basalt_labels)}")
    
    # Run simulation for several steps
    print("\nRunning simulation...")
    for step in range(20):
        sim.step_forward()
        
        # Check uranium count
        current_count = count_uranium_cells(sim)
        if current_count != initial_uranium_count:
            print(f"ERROR at step {step}: Uranium count changed from {initial_uranium_count} to {current_count}")
            break
        
        # Check connectivity every 5 steps
        if step % 5 == 0:
            connected, msg = verify_uranium_connectivity(sim, center_x, center_y)
            print(f"Step {step}: {msg}")
            if not connected:
                print("ERROR: Structure integrity lost!")
                break
    
    # Final verification
    print("\nFinal verification:")
    final_count = count_uranium_cells(sim)
    connected, msg = verify_uranium_connectivity(sim, center_x, center_y)
    
    print(f"Final uranium cells: {final_count}")
    print(f"Final structure: {msg}")
    
    # Determine success
    if final_count == initial_uranium_count and connected:
        print("\n✓ TEST PASSED: Uranium core remained intact!")
        return 0
    else:
        print("\n✗ TEST FAILED: Structure was destroyed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())