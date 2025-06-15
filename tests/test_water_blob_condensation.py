import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_game import GeoGame
from materials import MaterialType

def _bounding_box(mask: np.ndarray):
    ys, xs = np.where(mask)
    return ys.min(), ys.max(), xs.min(), xs.max()


def _aspect_ratio(mask: np.ndarray):
    y0, y1, x0, x1 = _bounding_box(mask)
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    return max(w / h, h / w)  # always >=1

def test_water_condenses_to_circular_blob():
    # Create simulation domain mostly SPACE
    sim = GeoGame(width=40, height=40, cell_size=50.0)

    # Fill rectangular bar of WATER across centre
    bar_top, bar_bottom = 18, 22
    bar_left, bar_right = 5, 35
    water_mask = np.zeros((40, 40), dtype=bool)
    water_mask[bar_top:bar_bottom, bar_left:bar_right] = True
    sim.material_types[water_mask] = MaterialType.WATER
    sim.temperature[water_mask] = 293.15  # room temp

    # Update derived fields
    sim._update_material_properties()
    sim.fluid_dynamics.calculate_planetary_pressure()

    initial_ratio = _aspect_ratio(sim.material_types == MaterialType.WATER)
    assert initial_ratio > 2.0  # sanity check bar is elongated

    # Run simulation for 120 macro-steps (reasonable settling time)
    for _ in range(120):
        sim.step_forward()

    final_ratio = _aspect_ratio(sim.material_types == MaterialType.WATER)

    # Expect water to approach circular blob: aspect ratio significantly reduced
    assert final_ratio < 1.6, f"Water blob still elongated (ratio {final_ratio:.2f})" 