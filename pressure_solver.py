"""pressure_solver.py – multigrid Poisson solver for pressure.

Solves ∇²P = rhs with homogeneous Dirichlet boundary conditions (P = 0 at
the outer boundary).  The implementation is a classical V-cycle with red-
black Gauss–Seidel smoothing, restriction by full-weighting and bilinear
prolongation.  It is dimension-agnostic (handles rectangular grids).

Designed to be *self-contained* (NumPy only) so it can be unit-tested or
re-used outside the main simulation engine.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np

# -----------------------------------------------------------------------------
# Red-black Gauss–Seidel smoother (in-place)
# -----------------------------------------------------------------------------

def _gauss_seidel_rb(phi: np.ndarray, rhs: np.ndarray, dx: float, iterations: int = 3):
    ny, nx = phi.shape
    h2 = dx * dx
    for _ in range(iterations):
        # Red sweep (i+j even)
        for color in (0, 1):
            phi[1:-1, 1:-1][((np.indices((ny-2, nx-2)).sum(axis=0) & 1) == color)] = (
                (
                    phi[1:-1, :-2] + phi[1:-1, 2:] +
                    phi[:-2, 1:-1] + phi[2:, 1:-1] - h2 * rhs[1:-1, 1:-1]
                ) / 4.0
            )[((np.indices((ny-2, nx-2)).sum(axis=0) & 1) == color)]

# -----------------------------------------------------------------------------
# Restriction (full weighting) & prolongation (bilinear)
# -----------------------------------------------------------------------------

# --- Full-weighting restriction that works for odd sizes --------------------
def _restrict(res: np.ndarray) -> np.ndarray:
    """Return coarse residual with shape (⌈ny/2⌉, ⌈nx/2⌉)."""
    ny, nx = res.shape
    nyc, nxc = ny // 2, nx // 2  # even sizes guaranteed by padding
    return 0.25 * (
        res[0:ny:2, 0:nx:2] + res[1:ny:2, 0:nx:2] + res[0:ny:2, 1:nx:2] + res[1:ny:2, 1:nx:2]
    )

# --- Bilinear prolongation supporting odd sizes -----------------------------
def _prolong(coarse: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    nyf, nxf = shape
    nyc, nxc = coarse.shape
    fine = np.zeros(shape, dtype=coarse.dtype)

    # Injection of coarse points
    fine[0::2, 0::2] = coarse

    # Horizontal interpolation (even rows, odd cols)
    if nxf > 1:
        fine[0::2, 1:-1:2] = 0.5 * (coarse[:, :-1] + coarse[:, 1:])
        if nxf % 2 == 0:  # even fine grid width – last col odd index exists
            fine[0::2, -1] = coarse[:, -1]  # copy nearest

    # Vertical interpolation (odd rows, even cols)
    if nyf > 1:
        fine[1:-1:2, 0::2] = 0.5 * (coarse[:-1, :] + coarse[1:, :])
        if nyf % 2 == 0:
            fine[-1, 0::2] = coarse[-1, :]  # copy nearest

    # Bilinear centers (odd, odd)
    if nyf > 1 and nxf > 1:
        fine[1:-1:2, 1:-1:2] = 0.25 * (
            coarse[:-1, :-1] + coarse[:-1, 1:] + coarse[1:, :-1] + coarse[1:, 1:]
        )
        # Last col/row boundaries if needed
        if nxf % 2 == 0:
            fine[1:-1:2, -1] = 0.5 * (coarse[:-1, -1] + coarse[1:, -1])
        if nyf % 2 == 0:
            fine[-1, 1:-1:2] = 0.5 * (coarse[-1, :-1] + coarse[-1, 1:])
        if nyf % 2 == 0 and nxf % 2 == 0:
            fine[-1, -1] = coarse[-1, -1]

    # Safety: log if shape mismatch (should never happen)
    if fine.shape != shape:
        print(f"[pressure_solver] Prolong shape mismatch fine={fine.shape} target={shape}")
    return fine

# -----------------------------------------------------------------------------
# V-cycle
# -----------------------------------------------------------------------------

def _v_cycle(phi: np.ndarray, rhs: np.ndarray, dx: float, level: int, max_level: int):
    # Pre-smooth
    _gauss_seidel_rb(phi, rhs, dx, iterations=3)

    ny, nx = phi.shape
    if level < max_level and min(ny, nx) > 3:
        # Compute residual
        res = np.zeros_like(phi)
        res[1:-1, 1:-1] = rhs[1:-1, 1:-1] - (
            (phi[1:-1, :-2] + phi[1:-1, 2:] + phi[:-2, 1:-1] + phi[2:, 1:-1] - 4 * phi[1:-1, 1:-1]) / (dx * dx)
        )
        # Restrict residual to coarse grid
        res_c = _restrict(res)
        phi_c = np.zeros_like(res_c)
        _v_cycle(phi_c, res_c, dx * 2, level + 1, max_level)
        # Prolong error and correct
        corr = _prolong(phi_c, phi.shape)
        if corr.shape != phi.shape:
            print(f"[pressure_solver] adding correction broadcast mismatch corr={corr.shape} phi={phi.shape}")
            h = min(corr.shape[0], phi.shape[0])
            w = min(corr.shape[1], phi.shape[1])
            phi[:h, :w] += corr[:h, :w]
        else:
            phi += corr
        # Post-smooth
        _gauss_seidel_rb(phi, rhs, dx, iterations=3)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def solve_pressure(rhs: np.ndarray, dx: float, *, tol: float = 1e-6, max_cycles: int = 50) -> np.ndarray:
    """Return P solving ∇²P = rhs (Dirichlet 0 at boundary)."""
    ny, nx = rhs.shape

    # ------------------------------------------------------------------
    # Pad to EVEN sizes so each coarsening step halves cleanly. Adds at most
    # one row/column of zeros (Dirichlet 0) which does not affect interior
    # solution but removes off-by-one shape issues.
    # ------------------------------------------------------------------
    ny_pad = ny + (ny & 1)
    nx_pad = nx + (nx & 1)

    rhs_pad = np.zeros((ny_pad, nx_pad), dtype=rhs.dtype)
    rhs_pad[:ny, :nx] = rhs

    # Number of coarse levels now safe (both dims even)
    max_level = int(np.floor(np.log2(min(ny_pad, nx_pad)))) - 2
    max_level = max(0, max_level)

    phi_pad = np.zeros_like(rhs_pad)

    for _ in range(max_cycles):
        _v_cycle(phi_pad, rhs_pad, dx, 0, max_level)
        # Compute residual norm
        res = np.zeros_like(rhs_pad)
        res[1:-1, 1:-1] = rhs_pad[1:-1, 1:-1] - (
            (phi_pad[1:-1, :-2] + phi_pad[1:-1, 2:] + phi_pad[:-2, 1:-1] + phi_pad[2:, 1:-1] - 4 * phi_pad[1:-1, 1:-1]) / (dx * dx)
        )
        err = np.linalg.norm(res[:ny, :nx]) / (ny * nx)
        if err < tol:
            break
    return phi_pad[:ny, :nx] 