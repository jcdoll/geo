"""Unit tests for pressure solver algorithms"""

import numpy as np
import pytest
from pressure_solver import solve_pressure, solve_poisson_variable


def test_pressure_solver_basic():
    """Test basic pressure solver functionality"""
    # Simple test case: uniform RHS should give parabolic solution
    ny, nx = 10, 10
    dx = 1.0
    
    # Uniform RHS = 1 should give P = (x²+y²)/4 - boundary terms
    rhs = np.ones((ny, nx))
    pressure = solve_pressure(rhs, dx)
    
    # Check that boundary conditions are satisfied (P=0 at boundary)
    assert np.allclose(pressure[0, :], 0), "Top boundary should be zero"
    assert np.allclose(pressure[-1, :], 0), "Bottom boundary should be zero"
    assert np.allclose(pressure[:, 0], 0), "Left boundary should be zero"
    assert np.allclose(pressure[:, -1], 0), "Right boundary should be zero"
    
    # Check that interior has reasonable values
    interior_max = np.max(pressure[1:-1, 1:-1])
    assert interior_max > 0, "Interior pressure should be positive for positive RHS"
    
    print(f"Basic pressure solver test passed")
    print(f"  Max interior pressure: {interior_max:.6f}")


def test_pressure_solver_zero_rhs():
    """Test pressure solver with zero RHS"""
    ny, nx = 8, 8
    dx = 1.0
    
    # Zero RHS should give zero solution
    rhs = np.zeros((ny, nx))
    pressure = solve_pressure(rhs, dx)
    
    # Should be zero everywhere
    assert np.allclose(pressure, 0), "Zero RHS should give zero pressure field"
    
    print(f"Zero RHS test passed")


def test_pressure_solver_point_source():
    """Test pressure solver with point source"""
    ny, nx = 10, 10
    dx = 1.0
    
    # Point source at center
    rhs = np.zeros((ny, nx))
    rhs[ny//2, nx//2] = 1.0
    
    pressure = solve_pressure(rhs, dx)
    
    # Should have maximum at or near the point source
    max_y, max_x = np.unravel_index(np.argmax(pressure), pressure.shape)
    center_y, center_x = ny//2, nx//2
    
    # Maximum should be near the center
    distance_from_center = np.sqrt((max_y - center_y)**2 + (max_x - center_x)**2)
    assert distance_from_center <= 2, "Maximum should be near point source"
    
    # Values should decay away from source
    center_value = pressure[center_y, center_x]
    corner_value = pressure[1, 1]  # Near corner but not on boundary
    
    assert center_value > corner_value, "Pressure should be higher near source"
    
    print(f"Point source test passed")
    print(f"  Center pressure: {center_value:.6f}")
    print(f"  Corner pressure: {corner_value:.6f}")


def test_pressure_solver_convergence():
    """Test that pressure solver converges for reasonable problems"""
    ny, nx = 20, 20
    dx = 0.1
    
    # Smooth RHS function
    y, x = np.mgrid[0:ny, 0:nx]
    rhs = np.sin(np.pi * x / nx) * np.sin(np.pi * y / ny)
    
    pressure = solve_pressure(rhs, dx)
    
    # Check residual: ∇²P - RHS should be small
    residual = np.zeros_like(pressure)
    residual[1:-1, 1:-1] = (
        (pressure[1:-1, :-2] + pressure[1:-1, 2:] + 
         pressure[:-2, 1:-1] + pressure[2:, 1:-1] - 4*pressure[1:-1, 1:-1]) / (dx*dx)
        - rhs[1:-1, 1:-1]
    )
    
    max_residual = np.max(np.abs(residual))
    assert max_residual < 1e-3, f"Residual too large: {max_residual}"
    
    print(f"Convergence test passed")
    print(f"  Max residual: {max_residual:.2e}")


def test_variable_coefficient_solver():
    """Test variable coefficient Poisson solver"""
    ny, nx = 8, 8
    dx = 1.0
    
    # Variable coefficient (like 1/density)
    k = np.ones((ny, nx))
    k[:, :nx//2] = 0.5  # Different coefficient on left side
    
    # Simple RHS
    rhs = np.ones((ny, nx))
    
    pressure = solve_poisson_variable(rhs, k, dx)
    
    # Should satisfy boundary conditions
    assert np.allclose(pressure[0, :], 0), "Top boundary should be zero"
    assert np.allclose(pressure[-1, :], 0), "Bottom boundary should be zero"
    assert np.allclose(pressure[:, 0], 0), "Left boundary should be zero"
    assert np.allclose(pressure[:, -1], 0), "Right boundary should be zero"
    
    # Should have different behavior on left vs right
    left_avg = np.mean(pressure[1:-1, 1:nx//2])
    right_avg = np.mean(pressure[1:-1, nx//2:-1])
    
    assert abs(left_avg - right_avg) > 1e-6, "Variable coefficient should create asymmetry"
    
    print(f"Variable coefficient test passed")
    print(f"  Left avg: {left_avg:.6f}, Right avg: {right_avg:.6f}")


if __name__ == "__main__":
    test_pressure_solver_basic()
    test_pressure_solver_zero_rhs() 
    test_pressure_solver_point_source()
    test_pressure_solver_convergence()
    test_variable_coefficient_solver()
    print("\nAll pressure solver unit tests passed!")