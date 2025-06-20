"""Regression test: ensure space (vacuum) does not erode the planet/atmosphere.

The number of cells with MaterialType.SPACE should remain constant (or shrink)
after the simulation advances. An increase would mean atmosphere/rock is
being replaced by vacuum, indicating a bug like the one previously observed
where the boundary eroded.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_game import GeoGame as GeologySimulation
from materials import MaterialType


def test_space_cell_count_stable():
    """Run several steps and assert SPACE-cell count does not grow."""
    # Medium-sized grid to make erosion evident but keep test fast
    sim = GeologySimulation(width=80, height=80)

    # Record initial count of SPACE cells (vacuum outside planet)
    initial_space = np.count_nonzero(sim.material_types == MaterialType.SPACE)

    # Advance the simulation for a modest number of steps
    steps = 30
    for _ in range(steps):
        sim.step_forward()

    final_space = np.count_nonzero(sim.material_types == MaterialType.SPACE)

    # Allow a tiny tolerance because stochastic processes may occasionally
    # eject single cells into space. The leak we saw produced large growth,
    # so a delta of 1-2 cells is acceptable.
    tolerance = 5  # cells
    assert final_space - initial_space <= tolerance, (
        f"SPACE cell count increased from {initial_space} to {final_space},"
        " indicating atmosphere/planet erosion." ) 