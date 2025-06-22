"""Verify the gradient issue in bulk water."""

import numpy as np
from pressure_solver import solve_pressure

# Simple 1D-ish water column
height = 30
width = 10
cell_size = 50.0

# Create density field
density = np.ones((height, width)) * 1.2  # air
density[10:, :] = 1000.0  # water

# RHS for pressure equation: g·∇ρ
g_y = 9.81
grad_rho_y = np.zeros_like(density)
grad_rho_y[1:-1, :] = (density[2:, :] - density[:-2, :]) / (2 * cell_size)
rhs = g_y * grad_rho_y

# Solve for pressure
pressure = solve_pressure(rhs, cell_size, bc_type='neumann')

# Calculate pressure gradient
grad_p_y = np.zeros_like(pressure)
grad_p_y[1:-1, :] = (pressure[2:, :] - pressure[:-2, :]) / (2 * cell_size)

# What we want for equilibrium: ∇P = ρg
target_grad_p = density * g_y

# Compare at different depths
print("Checking equilibrium condition: ∇P = ρg")
print("Location | ∇P (actual) | ρg (target) | Error")
print("-" * 50)

x = 5
for y in [8, 10, 12, 15, 20, 25]:
    actual = grad_p_y[y, x]
    target = target_grad_p[y, x]
    error_pct = abs(actual - target) / target * 100 if target > 0 else 0
    mat = "AIR" if y < 10 else "WATER"
    print(f"y={y:2d} ({mat:5s}) | {actual:11.1f} | {target:11.1f} | {error_pct:5.1f}%")

print("\nThe problem is clear:")
print("- In bulk water, ∇P ≈ 0 instead of ρg = 9810")
print("- Only at the interface do we get non-zero ∇P")
print("- This is why water accelerates downward!")