"""Test simulate-reset-simulate workflow for GeologySimulation

This regression test guards against issues where a simulation
is advanced, completely reset via repeated `step_backward` calls,
then advanced again. The simulation should be able to continue
running without numerical errors or state corruption.
"""

import numpy as np
import sys
import os

# Ensure the geo package root is on the path so we can import directly when
# running the tests from the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_game import GeoGame as GeologySimulation


def test_simulate_reset_simulate_workflow():
    """Advance, fully reset, and advance again – should run stably."""
    sim = GeologySimulation(width=12, height=10)

    # --- First simulation phase ------------------------------------------------
    first_steps = 5
    for _ in range(first_steps):
        sim.step_forward()

    # Basic sanity after first phase
    assert sim.time > 0.0
    assert len(sim.history) == first_steps  # One history entry per step

    # --- Reset phase -----------------------------------------------------------
    # Use the new reset helper which mirrors the visualizer's 'R' key.

    sim.reset()

    # After the reset the simulation should resemble a fresh instance
    assert sim.time == 0.0
    assert len(sim.history) == 0

    # Ensure core arrays remain intact
    assert sim.temperature.shape == (10, 12)
    assert sim.pressure.shape == (10, 12)

    # --- Second simulation phase ----------------------------------------------
    second_steps = 5
    for _ in range(second_steps):
        sim.step_forward()

    # Time should advance again from zero
    assert sim.time > 0.0
    assert len(sim.history) == second_steps

    # Validate numerical stability after second phase
    assert np.all(np.isfinite(sim.temperature))
    assert np.all(np.isfinite(sim.pressure))

    # Guard against negative pressures (should stay non-negative in model)
    assert np.all(sim.pressure >= 0.0) 