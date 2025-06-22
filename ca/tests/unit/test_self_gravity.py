import numpy as np
import pytest

from geo.gravity_solver import solve_potential, potential_to_gravity, G_SI


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """5-point Laplacian with zero Dirichlet edges."""
    lap = np.zeros_like(field)
    lap[1:-1, 1:-1] = (
        field[1:-1, 2:] + field[1:-1, :-2] + field[2:, 1:-1] + field[:-2, 1:-1] - 4 * field[1:-1, 1:-1]
    ) / (dx * dx)
    return lap


def test_constant_density_poisson_residual():
    """Uniform density block should satisfy Poisson to numerical precision."""
    nx = ny = 32
    dx = 100.0  # m
    rho = np.ones((ny, nx)) * 2500.0  # kg/m³

    phi = solve_potential(rho, dx, G=G_SI)
    residual = laplacian(phi, dx) - 4.0 * np.pi * G_SI * rho

    # Ignore boundary (first/last rows/cols)
    interior = residual[1:-1, 1:-1]
    max_res = np.max(np.abs(interior))

    # With DST solver we expect residual at round-off (~1e-8).
    assert max_res < 1e-4, f"Poisson residual too large: {max_res}"


def test_gravity_points_inward():
    """Gravity vectors should point toward centre for uniform density disk."""
    nx = ny = 40
    dx = 50.0
    rho = np.zeros((ny, nx))
    yy, xx = np.ogrid[:ny, :nx]
    cy, cx = ny // 2, nx // 2
    radius = 10
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    rho[mask] = 3000.0

    phi = solve_potential(rho, dx)
    gx, gy = potential_to_gravity(phi, dx)

    # Pick 8 sample points on axes outside core.
    samples = [(cy, cx + radius + 3), (cy, cx - radius - 3),
               (cy + radius + 3, cx), (cy - radius - 3, cx)]
    for y, x in samples:
        dx_c = cx - x
        dy_c = cy - y
        dot = gx[y, x] * dx_c + gy[y, x] * dy_c  # Positive dot means pointing inward
        assert dot > 0, "Gravity should point toward high-density region" 