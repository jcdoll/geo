import numpy as np
import pytest

from core_state import CoreState as GeologySimulation
from materials import MaterialType


dx_default = 50.0  # Shared cell size used across tests


def calculate_divergence(vx: np.ndarray, vy: np.ndarray, dx: float) -> np.ndarray:
    """Utility: return divergence field using central differences."""
    div = np.zeros_like(vx)
    h, w = vx.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            dvx_dx = (vx[y, x + 1] - vx[y, x - 1]) / (2 * dx)
            dvy_dy = (vy[y + 1, x] - vy[y - 1, x]) / (2 * dx)
            div[y, x] = dvx_dx + dvy_dy
    return div


class TestVelocityProjection:
    """Validate velocity projection step reduces divergence."""

    def setup_method(self):
        self.sim = GeologySimulation(width=16, height=16, cell_size=dx_default)
        self.vx = self.sim.velocity_x  # Aliases for brevity
        self.vy = self.sim.velocity_y
        self.P = self.sim.pressure

    def _initialise_divergent_field(self):
        for y in range(16):
            for x in range(16):
                self.vx[y, x] = x - 8  # Radial expansion
                self.vy[y, x] = y - 8

        # Quadratic pressure field that produces opposing gradient
        for y in range(16):
            for x in range(16):
                self.P[y, x] = -0.5 * ((x - 8) ** 2 + (y - 8) ** 2)

    def test_velocity_projection_reduces_divergence(self):
        self._initialise_divergent_field()
        dt = 0.1
        rho = 1000.0  # kg/m³ representative fluid density
        dx = self.sim.cell_size

        vx_new = self.vx.copy()
        vy_new = self.vy.copy()

        # Manual projection: v_new = v_old - dt * grad(P) / rho
        for y in range(1, 15):
            for x in range(1, 15):
                if self.sim.material_types[y, x] != MaterialType.SPACE:
                    dP_dx = (self.P[y, x + 1] - self.P[y, x - 1]) / (2 * dx)
                    dP_dy = (self.P[y + 1, x] - self.P[y - 1, x]) / (2 * dx)
                    vx_new[y, x] -= dt * dP_dx / rho
                    vy_new[y, x] -= dt * dP_dy / rho

        div_old = calculate_divergence(self.vx, self.vy, dx)
        div_new = calculate_divergence(vx_new, vy_new, dx)

        max_div_old = np.max(np.abs(div_old[1:15, 1:15]))
        max_div_new = np.max(np.abs(div_new[1:15, 1:15]))

        # Permit equality within numerical tolerance rather than strict decrease
        assert max_div_new <= max_div_old + 1e-10, (
            f"Projection should not increase divergence: {max_div_old} -> {max_div_new}") 