import numpy as np
import pytest

from geo.simulation_engine import GeologySimulation
from geo.materials import MaterialType


def test_water_cell_conservation():
    """Verify that total number of water-bearing cells (water, ice, vapor) is conserved.

    This test is expected to fail with the current implementation and serves
    as a regression check once the leak is fixed.
    """

    np.random.seed(42)

    # Medium grid with deliberate surface cavities
    sim = GeologySimulation(width=60, height=60, log_level="INFO")

    # Carve random SPACE craters in the surface to stress gas re-entry logic
    rng = np.random.default_rng(123)
    for _ in range(50):
        y = rng.integers(low=50, high=59)  # near surface (outer ring)
        x = rng.integers(low=0, high=59)
        sim.delete_material_blob(x, y, radius=rng.integers(1, 3))

    def water_cells_count():
        mask = (
            (sim.material_types == MaterialType.WATER) |
            (sim.material_types == MaterialType.ICE) |
            (sim.material_types == MaterialType.WATER_VAPOR)
        )
        return int(np.sum(mask))

    initial_count = water_cells_count()

    # Advance several macro-steps
    steps = 400
    for _ in range(steps):
        sim.step_forward()

    final_count = water_cells_count()

    # Allow 1 % numerical/ stochastic variation
    tolerance = int(initial_count * 0.01)
    assert final_count >= initial_count - tolerance, (
        f"Water-bearing cells dropped from {initial_count} to {final_count}") 