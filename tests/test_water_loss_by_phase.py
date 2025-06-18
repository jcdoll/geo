import numpy as np
import pytest

from geo_game import GeoGame
from materials import MaterialType

def _disable_phase(sim, dotted_attr: str):
    parts = dotted_attr.split('.')
    target = sim
    for p in parts[:-1]:
        target = getattr(target, p)
    setattr(target, parts[-1], lambda *a, **kw: False)

PHASE_ATTRS = [
    "fluid_dynamics_module.apply_fluid_dynamics",
    "fluid_dynamics_module.apply_density_stratification",
    "fluid_dynamics_module.apply_gravitational_collapse",
    "material_processes_module.apply_weathering",
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
    """Diagnose which physics phase still leaks water by disabling them one at a time."""
    np.random.seed(0)
    sim = GeoGame(50, 50, log_level="INFO")

    # Punch SPACE craters near the surface to stress leakage paths
    for y in range(45, 50):
        for x in range(0, 50, 5):
            sim.delete_material_blob(x, y, radius=1)

    if disabled_phase is not None:
        _disable_phase(sim, disabled_phase)

    start = count_water(sim)
    for _ in range(100):
        sim.step_forward()
    end = count_water(sim)

    pct = 0.0 if start == 0 else (end - start) / start * 100.0
    label = disabled_phase or "None (all active)"
    print(f"\nPhase disabled: {label:<45} ΔWater = {pct:+6.2f} %")
    assert True  # diagnostics only 