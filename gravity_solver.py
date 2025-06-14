#!/usr/bin/env python3
"""gravity_solver.py – FFT-based Poisson solver for self-gravity.

The solver assumes homogeneous Dirichlet boundary conditions (Φ = 0 at the
edges) which is equivalent to an isolated planet embedded in vacuum, as long
as the computational domain is large enough to encompass the whole body.

We use the discrete sine transform (DST-II) along both axes, because for a
function that vanishes at the boundaries the Laplacian is diagonal in that
basis.  SciPy 1.10+ provides scipy.fft.dst / idst which are contiguous and
thread-parallel.

Notation
--------
ρ   – density field (kg m⁻³), 2-D NumPy array (ny, nx)
Φ   – gravitational potential (m² s⁻²) satisfying ∇²Φ = 4 π G ρ
G   – gravitational constant (m³ kg⁻¹ s⁻²).  We use the real value but the
      caller is free to scale it for exaggerated geology.
Δx  – grid spacing (m); we assume square cells.

This module is intentionally self-contained (NumPy + SciPy only) so it can be
unit-tested without importing the entire simulation engine.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.fft import dstn, idstn  # type: ignore – present since SciPy 1.10

# DST type-II / IDST type-III pair is orthonormal up to a constant scale.
_DST_TYPE = 2
_IDST_TYPE = 2  # SciPy's idst uses the same type enum.

# Gravity constant in SI units; can be overridden by caller.
G_SI = 6.67430e-11  # m³ kg⁻¹ s⁻²


def solve_potential(
    density: np.ndarray,
    dx: float,
    *,
    G: float = G_SI,
) -> np.ndarray:
    """Return gravitational potential Φ solving ∇²Φ = 4 π G ρ.

    Parameters
    ----------
    density : ndarray (ny, nx)
        Mass density field (kg m⁻³).
    dx : float
        Cell size (m).  Assumed equal in x and y.
    G : float, optional
        Gravitational constant to use.  Default is the SI value.

    Notes
    -----
    The DST treats the first and last grid lines as boundaries where Φ = 0.
    This mirrors an isolated body in space if the outermost cells are SPACE
    with almost zero density.
    """
    rho = np.asarray(density, dtype=np.float64)
    ny, nx = rho.shape

    # Right-hand side scaled for the grid spacing.
    rhs = 4.0 * np.pi * G * rho * dx * dx  # multiply by Δx² to keep units consistent

    # Forward DST-II in both directions (orthogonal transform → invertible by same type).
    rhs_hat = dstn(rhs, type=_DST_TYPE, norm="ortho")

    # Eigenvalues of the Laplacian with Dirichlet BCs on a square grid.
    j = np.arange(1, ny + 1)  # 1-based index in DST basis
    i = np.arange(1, nx + 1)
    sin_j = np.sin(np.pi * j / (ny + 1))
    sin_i = np.sin(np.pi * i / (nx + 1))
    lambda_y = 2.0 * (1.0 - sin_j[:, None] ** 2)  # shape (ny, 1)
    lambda_x = 2.0 * (1.0 - sin_i[None, :] ** 2)  # shape (1, nx)
    eigvals = (lambda_x + lambda_y)  # Broadcasted sum (ny, nx)

    # Solve in transform domain: Φ_hat = rhs_hat / (-λ)
    phi_hat = rhs_hat / (-eigvals)

    # Zero mean potential → set (1,1) component to zero already (handled by eigvals≠0).

    # Inverse DST to obtain Φ.
    phi = idstn(phi_hat, type=_IDST_TYPE, norm="ortho")
    return phi


def potential_to_gravity(phi: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return gravitational acceleration components from potential.

    g = −∇Φ
    Central differences interior, one-sided at borders.
    """
    gy, gx = np.gradient(phi, dx, edge_order=2)  # gy = ∂Φ/∂y, gx = ∂Φ/∂x
    gx = -gx
    gy = -gy
    return gx, gy 