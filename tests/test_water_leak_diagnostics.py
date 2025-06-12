import numpy as np
import pytest

from geo.simulation_engine import GeologySimulation
from geo.materials import MaterialType

PHASE_ATTRS = [
    "_apply_fluid_dynamics_vectorized",
    "_apply_density_stratification_local_vectorized",
    "_apply_gravitational_collapse_vectorized",
    "_apply_weathering",
]

def count_water(sim):
    mask = (
        (sim.material_types == MaterialType.WATER) |
        (sim.material_types == MaterialType.ICE) |
        (sim.material_types == MaterialType.WATER_VAPOR)
    )
    return int(np.sum(mask))

@pytest.mark.parametrize("disabled_phase", [None] + PHASE_ATTRS)
def test_water_loss_by_phase(disabled_phase, capsys):
    """Diagnostics: run 100 steps with one phase disabled and report water loss.

    The test never fails; it prints the percentage change so we can see which
    phase causes leakage.  Once leaks are fixed we can turn this into a real
    assertion.
    """
    np.random.seed(0)
    sim = GeologySimulation(50, 50, log_level="INFO")

    # Deliberately poke holes (SPACE) near surface to stress leakage.
    for y in range(45, 50):
        for x in range(0, 50, 5):
            sim.delete_material_blob(x, y, radius=1)

    if disabled_phase is not None:
        # Monkey-patch the phase to a no-op that returns False
        setattr(sim, disabled_phase, lambda *a, **kw: False)

    start = count_water(sim)
    for _ in range(100):
        sim.step_forward()
    end = count_water(sim)

    pct_change = (end - start) / start * 100.0
    phase_name = disabled_phase or "None (all active)"
    print(f"Phase disabled: {phase_name:40}  ΔWater = {pct_change:+5.2f}%")

    # Always pass – diagnostics only.
    assert True 