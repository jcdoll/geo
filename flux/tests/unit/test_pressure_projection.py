# Ensure project root is on sys.path
import sys, pathlib
import numpy as np
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# If still cannot import state, fall back to dynamic import via path
try:
    from state import FluxState
except ModuleNotFoundError:
    spec = (ROOT / "state.py").as_posix()
    import importlib.util, types
    module_name = "state"
    spec_obj = importlib.util.spec_from_file_location(module_name, spec)
    mod = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = mod
    spec_obj.loader.exec_module(mod)
    FluxState = mod.FluxState

from physics import FluxPhysics
from pressure_solver import PressureSolver

G = 9.81  # gravity magnitude


def _build_layered_density(state: FluxState):
    """Three-layer density: rock, water, air."""
    ny = state.ny
    rock_end = ny // 3
    water_end = 2 * ny // 3

    state.density[:rock_end, :] = 2700.0
    state.density[rock_end:water_end, :] = 1000.0
    state.density[water_end:, :] = 1.2


def test_hydrostatic_pressure_water_column():
    """Water occupies bottom half; top half is space. Verify P(y)=ρ g h."""

    nx, ny = 32, 64
    dx = 1.0
    state = FluxState(nx, ny, dx)

    # Initialise density: space (≈0) top, water bottom
    water_density = 1000.0
    state.density.fill(1e-6)  # space
    state.density[ny//2 :, :] = water_density  # lower half water

    # Gravity field (downwards)
    gx = np.zeros((ny, nx), dtype=np.float32)
    gy = -G * np.ones((ny, nx), dtype=np.float32)

    solver = PressureSolver(state)

    # Solve with Neumann BC then shift so P=0 at top boundary (Dirichlet in effect)
    pressure = solver.solve_pressure(gx, gy, bc_type="neumann")
    pressure -= pressure[0, :].mean()

    # Compute pressure gradient
    _, dpdy = solver.compute_pressure_gradient_consistent(pressure)

    # Residual r = ∂P/∂y + ρ g (should be 0 in water)
    residual = dpdy + state.density * gy

    water_mask = state.density > 10.0
    max_res = np.abs(residual[water_mask]).max()

    # Expect high accuracy (<1e-2 Pa/m) for hydrostatic balance
    assert max_res < 1e-2, f"Hydrostatic residual too large: {max_res:.3e} Pa/m"

    # Additional sanity: pressure increases roughly linearly in water
    p_top_water = pressure[ny//2, 0]
    p_bottom = pressure[-1, 0]
    expected = water_density * G * (ny//2 * dx)
    assert np.isclose(p_bottom - p_top_water, expected, rtol=1e-2), "Pressure does not integrate to ρ g h"


def test_hydrostatic_equilibrium(capsys):
    nx, ny = 16, 48
    state = FluxState(nx, ny, dx=1.0)

    _build_layered_density(state)
    physics = FluxPhysics(state)

    gx = np.zeros((ny, nx), dtype=np.float32)
    gy = -9.81 * np.ones((ny, nx), dtype=np.float32)

    dt = 0.05
    n_steps = 10
    for step in range(n_steps):
        physics.update_momentum(gx, gy, dt)
        if step % 1 == 0:
            vmax_step = max(abs(state.velocity_x).max(), abs(state.velocity_y).max())
            with capsys.disabled():
                print(f"Step {step}: vmax = {vmax_step:.3e} m/s")
                sys.stdout.flush()
        if vmax_step < 1e-4:
            break

    vmax = max(abs(state.velocity_x).max(), abs(state.velocity_y).max())
    assert vmax < 1e-3, f"Residual velocity too high: {vmax}"

    # ------------------------------------------------------------------
    # Additional check: pressure gradient should balance gravity (∂P/∂y ≈ ρ g)
    # ------------------------------------------------------------------
    pres_solver = PressureSolver(state)
    # Use pressure field accumulated during projection
    _, dpdy = pres_solver.compute_pressure_gradient_consistent(state.pressure)

    hydro_residual = dpdy + state.density * gy  # should be ~0
    max_res = np.max(np.abs(hydro_residual[state.density > 10]))  # ignore near-vacuum
    with capsys.disabled():
        print(f"Hydrostatic residual: {max_res:.3e} Pa/m")
    assert max_res < 1e-2, "Pressure gradient does not balance gravity" 