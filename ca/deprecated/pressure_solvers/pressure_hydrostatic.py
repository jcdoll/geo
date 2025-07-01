"""Direct hydrostatic pressure calculation.

For a fluid in hydrostatic equilibrium under gravity, pressure increases
with depth according to:

dP/dy = ρg

This can be integrated directly to get pressure.
"""

import numpy as np


def compute_hydrostatic_pressure(density: np.ndarray, gravity_y: np.ndarray, 
                                 cell_size: float) -> np.ndarray:
    """Compute pressure by direct integration of hydrostatic equation.
    
    Integrates dP/dy = ρg from top to bottom.
    
    Args:
        density: Density field (kg/m³)
        gravity_y: Y-component of gravity (m/s²) 
        cell_size: Grid spacing (m)
        
    Returns:
        pressure: Pressure field (Pa)
    """
    ny, nx = density.shape
    pressure = np.zeros_like(density)
    
    # Integrate from top to bottom
    # P[i+1] = P[i] + ρ[i] * g[i] * Δy
    # We use the average density between cells for better accuracy
    for i in range(ny - 1):
        # Average density and gravity between cells i and i+1
        rho_avg = 0.5 * (density[i, :] + density[i+1, :])
        g_avg = 0.5 * (gravity_y[i, :] + gravity_y[i+1, :])
        
        # Pressure increment
        dp = rho_avg * g_avg * cell_size
        
        # Update pressure
        pressure[i+1, :] = pressure[i, :] + dp
    
    return pressure


def compute_hydrostatic_forces(density: np.ndarray, gravity_x: np.ndarray,
                               gravity_y: np.ndarray, pressure: np.ndarray,
                               cell_size: float) -> tuple:
    """Compute forces with hydrostatic pressure gradient.
    
    Uses centered differences for pressure gradient, matching the 
    discretization used in the main simulation.
    """
    # Gravity forces
    fx_gravity = density * gravity_x
    fy_gravity = density * gravity_y
    
    # Pressure gradient forces (centered differences)
    fx_pressure = np.zeros_like(pressure)
    fy_pressure = np.zeros_like(pressure)
    
    # Interior points: centered differences
    fx_pressure[1:-1, 1:-1] = -(pressure[1:-1, 2:] - pressure[1:-1, :-2]) / (2 * cell_size)
    fy_pressure[1:-1, 1:-1] = -(pressure[2:, 1:-1] - pressure[:-2, 1:-1]) / (2 * cell_size)
    
    # Boundaries: one-sided differences
    # Left/right
    fx_pressure[:, 0] = -(pressure[:, 1] - pressure[:, 0]) / cell_size
    fx_pressure[:, -1] = -(pressure[:, -1] - pressure[:, -2]) / cell_size
    
    # Top/bottom
    fy_pressure[0, :] = -(pressure[1, :] - pressure[0, :]) / cell_size
    fy_pressure[-1, :] = -(pressure[-1, :] - pressure[-2, :]) / cell_size
    
    # Total forces
    fx_total = fx_gravity + fx_pressure
    fy_total = fy_gravity + fy_pressure
    
    return fx_total, fy_total


def test_direct_hydrostatic():
    """Test direct hydrostatic pressure calculation."""
    # Setup
    ny, nx = 30, 10
    cell_size = 50.0  # meters
    g = 9.81  # m/s²
    
    # Density field
    density = np.ones((ny, nx)) * 1.2  # air
    density[10:, :] = 1000.0  # water
    
    # Uniform gravity
    gravity_x = np.zeros((ny, nx))
    gravity_y = np.full((ny, nx), g)
    
    # Compute hydrostatic pressure
    pressure = compute_hydrostatic_pressure(density, gravity_y, cell_size)
    
    # Compute forces
    fx, fy = compute_hydrostatic_forces(density, gravity_x, gravity_y, 
                                         pressure, cell_size)
    
    print("Direct Hydrostatic Pressure Test")
    print("-" * 70)
    print("   y | Material | Pressure (Pa) | ρg (N/m³) | -∇P (N/m³) | Net Force")
    print("-" * 70)
    
    x = 5
    for y in [0, 5, 9, 10, 11, 15, 20, 25, 29]:
        p = pressure[y, x]
        rho_g = density[y, x] * gravity_y[y, x]
        pressure_force = fy[y, x] - rho_g
        net = fy[y, x]
        mat = "AIR" if y < 10 else "WATER"
        print(f"{y:4d} | {mat:8s} | {p:13.1f} | {rho_g:9.1f} | {pressure_force:10.1f} | {net:9.3f}")
    
    # Check equilibrium in bulk regions
    print("\nEquilibrium check in bulk regions:")
    
    # Air region (y=5)
    air_forces = fy[5, :]
    print(f"Air (y=5): max |force| = {np.max(np.abs(air_forces)):.3f} N/m³")
    
    # Water region (y=15-25) 
    water_forces = fy[15:25, :]
    print(f"Water (y=15-25): max |force| = {np.max(np.abs(water_forces)):.3f} N/m³")
    
    # Interface region (y=9-11)
    interface_forces = fy[9:12, :]
    print(f"Interface (y=9-11): max |force| = {np.max(np.abs(interface_forces)):.3f} N/m³")
    
    # Overall
    print(f"\nOverall max |force|: {np.max(np.abs(fy)):.3f} N/m³")
    print(f"Expected ρ_water * g: {1000 * g:.1f} N/m³")
    
    # Detailed gradient check
    print("\nDetailed gradient check at x=5:")
    print("   y | dP/dy (actual) | ρg (expected) | Error")
    print("-" * 45)
    
    for y in [5, 10, 11, 15, 20]:
        if y == 0:
            dp_dy = (pressure[1, x] - pressure[0, x]) / cell_size
        elif y == ny-1:
            dp_dy = (pressure[-1, x] - pressure[-2, x]) / cell_size
        else:
            dp_dy = (pressure[y+1, x] - pressure[y-1, x]) / (2 * cell_size)
        
        expected = density[y, x] * gravity_y[y, x]
        error = abs(dp_dy - expected)
        mat = "AIR" if y < 10 else "WATER"
        print(f"{y:4d} | {dp_dy:14.1f} | {expected:13.1f} | {error:7.1f}")


if __name__ == "__main__":
    test_direct_hydrostatic()