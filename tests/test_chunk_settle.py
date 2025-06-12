import numpy as np

from geo.simulation_engine import GeologySimulation
from geo.materials import MaterialType


def test_chunk_settle_respects_terminal_velocity():
    """Unsupported solid chunk should fall at most `terminal_settle_velocity` cells per settle pass."""

    width = height = 10
    sim = GeologySimulation(width, height, log_level="INFO")

    # Clear existing grid to SPACE & uniform temperature
    sim.material_types[:, :] = MaterialType.SPACE
    sim.temperature[:, :] = 300.0

    # Place a single unsupported basalt cell directly above the COM column
    col = width // 2  # column 5
    sim.material_types[0, col] = MaterialType.BASALT

    # Fix centre of mass so that direction is purely downward for this column
    sim.center_of_mass = (height // 2, width // 2)  # (5, 5)

    # Ensure terminal velocity exactly 3 cells
    sim.terminal_settle_velocity = 3

    moved = sim._settle_unsupported_chunks()
    assert moved, "Chunk should have moved"

    # Expected new row 3 for basalt cell (moved down by 3)
    basalt_positions = np.column_stack(np.where(sim.material_types == MaterialType.BASALT))
    assert basalt_positions.shape[0] == 1, "Basalt cell count mismatch"
    new_y, new_x = basalt_positions[0]
    assert new_x == col and new_y == 3, "Basalt cell did not move by terminal velocity"

    # Old position should now be SPACE
    assert sim.material_types[0, col] == MaterialType.SPACE 


def test_chunk_settle_inf_velocity_falls_all_the_way():
    """With terminal velocity = inf the chunk should fall until the first non-fluid cell or grid edge."""

    width = height = 10
    sim = GeologySimulation(width, height, log_level="INFO")
    sim.material_types[:, :] = MaterialType.SPACE
    sim.temperature[:, :] = 300.0

    # Single unsupported basalt cell at top-middle
    col = width // 2
    sim.material_types[0, col] = MaterialType.BASALT
    sim.center_of_mass = (height // 2, width // 2)
    sim.terminal_settle_velocity = float("inf")

    moved = sim._settle_unsupported_chunks()
    assert moved, "Chunk should have moved with infinite velocity"

    # Should now be at bottom row (index 9)
    assert sim.material_types[height - 1, col] == MaterialType.BASALT
    # Original cell became SPACE
    assert sim.material_types[0, col] == MaterialType.SPACE 