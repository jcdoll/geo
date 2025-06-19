"""Test rigid body motion - granite blocks should fall as coherent units"""

import numpy as np
import pytest
from geo_game import GeoGame as GeoSimulation
from materials import MaterialType


def test_rigid_body_fall():
    """Test that a granite block falls as a rigid body under external gravity"""
    # Create simulation with external gravity pointing down
    sim = GeoSimulation(width=30, height=30, cell_size=100)
    sim.external_gravity = (0, 10)  # Downward gravity (positive Y is down)
    sim.enable_self_gravity = False  # Disable self gravity for simplicity
    sim.enable_solid_drag = False  # Disable solid drag to allow free fall
    
    # Clear to space
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 2.7  # Space temperature
    
    # Create a granite block in space (3x3 block)
    block_y, block_x = 5, 15  # Starting position
    block_size = 3
    for dy in range(block_size):
        for dx in range(block_size):
            sim.material_types[block_y + dy, block_x + dx] = MaterialType.GRANITE
            sim.temperature[block_y + dy, block_x + dx] = 300.0
    
    # Create a solid surface at the bottom to catch the block
    for x in range(sim.width):
        sim.material_types[-3:, x] = MaterialType.BASALT  # Bottom 3 rows
        sim.temperature[-3:, x] = 300.0
    
    # Update properties
    sim._update_material_properties()
    
    # Check expected forces
    granite_density = sim.material_db.get_properties(MaterialType.GRANITE).density
    expected_force_density = granite_density * 10  # ρ * g
    cell_volume = sim.cell_size ** 2 * sim.cell_depth
    expected_force_per_cell = expected_force_density * cell_volume
    
    print(f"\nExpected force calculations:")
    print(f"Granite density: {granite_density} kg/m³")
    print(f"Gravity: 10 m/s²")
    print(f"Expected force density: {expected_force_density} N/m³")
    print(f"Cell volume: {cell_volume} m³")
    print(f"Expected force per cell: {expected_force_per_cell:.2e} N")
    print(f"Granite-granite binding: {sim.fluid_dynamics.solid_binding_force:.2e} N")
    print(f"Force ratio: {expected_force_per_cell / sim.fluid_dynamics.solid_binding_force:.2e}")
    
    # Record initial block position
    granite_mask = sim.material_types == MaterialType.GRANITE
    initial_positions = np.argwhere(granite_mask)
    initial_y_coords = initial_positions[:, 0]
    initial_x_coords = initial_positions[:, 1]
    initial_center_y = np.mean(initial_y_coords)
    initial_center_x = np.mean(initial_x_coords)
    
    print(f"\nInitial block center: ({initial_center_x:.1f}, {initial_center_y:.1f})")
    print(f"Initial granite cells: {len(initial_positions)}")
    
    # Run simulation for several steps
    positions_over_time = []
    for step in range(20):
        sim.step_forward(1.0)
        
        # Track block position
        granite_mask = sim.material_types == MaterialType.GRANITE
        if np.any(granite_mask):
            positions = np.argwhere(granite_mask)
            center_y = np.mean(positions[:, 0])
            center_x = np.mean(positions[:, 1])
            positions_over_time.append((center_x, center_y, len(positions)))
            print(f"Step {step+1}: center=({center_x:.1f}, {center_y:.1f}), cells={len(positions)}")
        else:
            print(f"Step {step+1}: No granite found!")
            break
    
    # Verify rigid body behavior
    assert len(positions_over_time) > 0, "Block disappeared immediately"
    
    # 1. Check that block maintains its shape (cell count stays constant)
    cell_counts = [p[2] for p in positions_over_time]
    assert all(count == 9 for count in cell_counts), f"Block fragmented! Cell counts: {cell_counts}"
    
    # 2. Check that block fell (Y coordinate increased)
    y_positions = [p[1] for p in positions_over_time]
    assert y_positions[-1] > initial_center_y, "Block didn't fall downward"
    
    # 3. Check that block moved as a unit (all cells moved together)
    # The block shape should be preserved
    final_granite_mask = sim.material_types == MaterialType.GRANITE
    if np.any(final_granite_mask):
        final_positions = np.argwhere(final_granite_mask)
        # Check if it's still a connected 3x3 block
        y_coords = final_positions[:, 0]
        x_coords = final_positions[:, 1]
        y_range = y_coords.max() - y_coords.min() + 1
        x_range = x_coords.max() - x_coords.min() + 1
        assert y_range == block_size and x_range == block_size, \
            f"Block deformed! Final size: {x_range}x{y_range}"
    
    print("\n✓ Rigid body fall test passed!")


def test_rigid_body_acceleration():
    """Test that rigid body acceleration matches expected physics"""
    sim = GeoSimulation(width=30, height=50, cell_size=100)
    sim.external_gravity = (0, 9.81)  # Standard gravity
    sim.enable_self_gravity = False
    sim.enable_solid_drag = False  # Disable drag for clean physics
    
    # Clear to space
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 2.7
    
    # Create granite block high up
    block_y = 5
    block_x = 15
    for dy in range(3):
        for dx in range(3):
            sim.material_types[block_y + dy, block_x + dx] = MaterialType.GRANITE
            sim.temperature[block_y + dy, block_x + dx] = 300.0
    
    sim._update_material_properties()
    
    # Track position over time
    dt = 1.0  # 1 second timesteps
    positions = []
    times = []
    
    for step in range(10):
        granite_mask = sim.material_types == MaterialType.GRANITE
        if np.any(granite_mask):
            y_center = np.mean(np.argwhere(granite_mask)[:, 0])
            positions.append(y_center * sim.cell_size / 100)  # Convert to grid units
            times.append(step * dt)
        
        sim.step_forward(dt)
    
    # Check if motion follows physics (with some tolerance for discrete simulation)
    if len(positions) >= 3:
        # Calculate velocities from positions
        velocities = []
        for i in range(1, len(positions)):
            v = (positions[i] - positions[i-1]) / dt
            velocities.append(v)
        
        # Check if velocity is increasing (acceleration)
        for i in range(1, len(velocities)):
            assert velocities[i] >= velocities[i-1], \
                f"Velocity should increase under constant gravity: {velocities}"
    
    print("\n✓ Rigid body acceleration test passed!")


def test_multiple_rigid_bodies():
    """Test that multiple separate rigid bodies fall independently"""
    sim = GeoSimulation(width=40, height=40, cell_size=100)
    sim.external_gravity = (0, 10)
    sim.enable_self_gravity = False
    sim.enable_solid_drag = False
    
    # Clear to space
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 2.7
    
    # Create two separate granite blocks
    # Block 1 at (10, 5)
    for dy in range(2):
        for dx in range(2):
            sim.material_types[5 + dy, 10 + dx] = MaterialType.GRANITE
            sim.temperature[5 + dy, 10 + dx] = 300.0
    
    # Block 2 at (25, 8) - starts lower
    for dy in range(2):
        for dx in range(2):
            sim.material_types[8 + dy, 25 + dx] = MaterialType.GRANITE  
            sim.temperature[8 + dy, 25 + dx] = 300.0
    
    # Add floor
    sim.material_types[-2:, :] = MaterialType.BASALT
    sim.temperature[-2:, :] = 300.0
    
    sim._update_material_properties()
    
    # Run simulation
    for step in range(15):
        sim.step_forward(1.0)
    
    # Both blocks should have fallen
    granite_positions = np.argwhere(sim.material_types == MaterialType.GRANITE)
    
    # Check that we still have 8 granite cells (two 2x2 blocks)
    assert len(granite_positions) == 8, f"Lost granite cells! Found {len(granite_positions)}"
    
    # Check that blocks are at different X positions (didn't merge)
    x_coords = granite_positions[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    assert x_max - x_min > 10, "Blocks seem to have merged horizontally"
    
    print("\n✓ Multiple rigid bodies test passed!")


if __name__ == "__main__":
    test_rigid_body_fall()
    print("\n" + "="*50)
    test_rigid_body_acceleration()
    print("\n" + "="*50)
    test_multiple_rigid_bodies() 