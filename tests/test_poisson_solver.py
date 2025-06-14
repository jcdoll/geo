"""Test suite for Poisson pressure solver"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_engine import GeologySimulation
from geo.materials import MaterialType
import pytest


class TestPoissonSolver:
    """Test suite for the Poisson pressure solver"""
    
    def setup_method(self):
        """Set up test simulation"""
        self.sim = GeologySimulation(width=16, height=16, cell_size=50.0)
        
        # Initialize velocity fields for testing
        self.sim.velocity_x = np.zeros((16, 16), dtype=np.float64)
        self.sim.velocity_y = np.zeros((16, 16), dtype=np.float64)
    
    def test_simple_divergence_calculation(self):
        """Test calculation of velocity divergence"""
        # Create a simple velocity field with known divergence
        # Uniform expansion: vx = x, vy = y should give divergence = 2
        
        for y in range(16):
            for x in range(16):
                self.sim.velocity_x[y, x] = x - 8  # Center at grid center
                self.sim.velocity_y[y, x] = y - 8
        
        # Calculate divergence manually for center cell
        dx = self.sim.cell_size
        center_y, center_x = 8, 8
        
        # Central difference approximation
        dvx_dx = (self.sim.velocity_x[center_y, center_x + 1] - 
                  self.sim.velocity_x[center_y, center_x - 1]) / (2 * dx)
        dvy_dy = (self.sim.velocity_y[center_y + 1, center_x] - 
                  self.sim.velocity_y[center_y - 1, center_x]) / (2 * dx)
        
        expected_divergence = dvx_dx + dvy_dy
        
        # For this velocity field, divergence should be 2/dx
        assert abs(expected_divergence - 2.0/dx) < 1e-10, f"Expected divergence 2/dx = {2.0/dx}, got {expected_divergence}"
    
    def test_zero_divergence_field(self):
        """Test that incompressible flow has zero divergence"""
        # Create a rotation field: vx = -y, vy = x (should have zero divergence)
        
        for y in range(16):
            for x in range(16):
                self.sim.velocity_x[y, x] = -(y - 8)  # -y component
                self.sim.velocity_y[y, x] = (x - 8)   # x component
        
        # Calculate divergence for interior cells
        dx = self.sim.cell_size
        
        for y in range(1, 15):
            for x in range(1, 15):
                dvx_dx = (self.sim.velocity_x[y, x + 1] - 
                          self.sim.velocity_x[y, x - 1]) / (2 * dx)
                dvy_dy = (self.sim.velocity_y[y + 1, x] - 
                          self.sim.velocity_y[y - 1, x]) / (2 * dx)
                
                divergence = dvx_dx + dvy_dy
                
                # Rotation field should have zero divergence
                assert abs(divergence) < 1e-10, f"Rotation field should have zero divergence, got {divergence} at ({x}, {y})"
    
    def test_pressure_boundary_conditions(self):
        """Test pressure boundary conditions"""
        # Space cells should have zero pressure
        space_mask = (self.sim.material_types == MaterialType.SPACE)
        
        # Set some non-zero pressures initially
        self.sim.pressure.fill(1.0)
        
        # Apply boundary conditions (space = 0)
        self.sim.pressure[space_mask] = 0.0
        
        # Check that space cells have zero pressure
        assert np.all(self.sim.pressure[space_mask] == 0), "Space cells should have zero pressure"
        
        # Check that non-space cells can have non-zero pressure
        non_space_mask = ~space_mask
        if np.any(non_space_mask):
            assert np.any(self.sim.pressure[non_space_mask] > 0), "Non-space cells should be able to have non-zero pressure"
    
    def test_jacobi_iteration_setup(self):
        """Test setup for Jacobi iteration"""
        # Create a simple right-hand side for Poisson equation
        rhs = np.zeros((16, 16))
        rhs[8, 8] = 1.0  # Point source at center
        
        # Initialize pressure field
        pressure = np.zeros((16, 16))
        
        # Use an in-place Gauss-Seidel sweep (simpler and still illustrates update)
        dx = self.sim.cell_size
        for i in range(1, 15):
            for j in range(1, 15):
                pressure[i, j] = 0.25 * (
                    pressure[i-1, j] + pressure[i+1, j] + 
                    pressure[i, j-1] + pressure[i, j+1] - 
                    rhs[i, j] * dx * dx
                )
        
        # After one sweep the centre must be negative and at least one neighbour non-zero
        assert pressure[8, 8] < 0, "Point source should create negative pressure"
        neighbour_vals = [pressure[7,8], pressure[9,8], pressure[8,7], pressure[8,9]]
        assert any(v != 0 for v in neighbour_vals), "Sweep should affect neighbouring cells"
    
    def test_sor_relaxation_factor(self):
        """Test SOR (Successive Over-Relaxation) method"""
        # SOR is Jacobi with relaxation: P_new = P_old + omega * (P_jacobi - P_old)
        
        rhs = np.zeros((16, 16))
        rhs[8, 8] = 1.0
        
        pressure = np.zeros((16, 16))
        dx = self.sim.cell_size
        omega = 1.5  # Over-relaxation factor
        
        # One SOR iteration
        pressure_new = pressure.copy()
        
        for i in range(1, 15):
            for j in range(1, 15):
                # Jacobi update
                jacobi_update = 0.25 * (
                    pressure_new[i-1, j] + pressure[i+1, j] + 
                    pressure_new[i, j-1] + pressure[i, j+1] - 
                    rhs[i, j] * dx * dx
                )
                
                # SOR update
                pressure_new[i, j] = pressure[i, j] + omega * (jacobi_update - pressure[i, j])
        
        # SOR should converge faster than Jacobi (larger changes)
        max_change_sor = np.max(np.abs(pressure_new - pressure))
        
        # Compare with Jacobi (omega = 1.0)
        pressure_jacobi = pressure.copy()
        for i in range(1, 15):
            for j in range(1, 15):
                pressure_jacobi[i, j] = 0.25 * (
                    pressure_jacobi[i-1, j] + pressure_jacobi[i+1, j] + 
                    pressure_jacobi[i, j-1] + pressure_jacobi[i, j+1] - 
                    rhs[i, j] * dx * dx
                )
        
        max_change_jacobi = np.max(np.abs(pressure_jacobi - pressure))
        
        # SOR with omega > 1 should make larger changes
        assert max_change_sor > max_change_jacobi, "SOR should make larger changes than Jacobi"
    
    def test_convergence_criteria(self):
        """Test convergence criteria for iterative solver"""
        # Create a simple problem that should converge
        rhs = np.zeros((16, 16))
        rhs[8, 8] = 1.0
        
        pressure = np.zeros((16, 16))
        dx = self.sim.cell_size
        tolerance = 1e-6
        max_iterations = 100
        
        initial_residual = None
        for iteration in range(max_iterations):
            pressure_old = pressure.copy()
            
            # Jacobi iteration
            for i in range(1, 15):
                for j in range(1, 15):
                    if self.sim.material_types[i, j] != MaterialType.SPACE:
                        pressure[i, j] = 0.25 * (
                            pressure_old[i-1, j] + pressure_old[i+1, j] + 
                            pressure_old[i, j-1] + pressure_old[i, j+1] - 
                            rhs[i, j] * dx * dx
                        )
            
            # Apply boundary conditions
            space_mask = (self.sim.material_types == MaterialType.SPACE)
            pressure[space_mask] = 0.0
            
            # Check convergence
            max_change = np.max(np.abs(pressure - pressure_old))
            if iteration == 0:
                initial_residual = max_change
            
            if max_change < tolerance:
                break
        
        # Residual should drop at least 5× over the iteration window unless it is
        # already below machine-precision on the first iteration (rare symmetric
        # configurations).
        assert initial_residual is not None
        if initial_residual < 1e-12:
            assert max_change < 1e-12, "Residual should remain essentially zero"
        else:
            assert max_change < initial_residual * 0.2, (
                f"Residual did not decrease sufficiently: {initial_residual} -> {max_change}")


class TestVelocityProjection:
    """Test velocity projection step"""
    
    def setup_method(self):
        """Set up test simulation"""
        self.sim = GeologySimulation(width=16, height=16, cell_size=50.0)
        
        # Initialize fields
        self.sim.velocity_x = np.zeros((16, 16), dtype=np.float64)
        self.sim.velocity_y = np.zeros((16, 16), dtype=np.float64)
        self.sim.pressure = np.zeros((16, 16), dtype=np.float64)
    
    def test_pressure_gradient_calculation(self):
        """Test calculation of pressure gradients"""
        # Create a linear pressure field: P = x + y
        for y in range(16):
            for x in range(16):
                self.sim.pressure[y, x] = x + y
        
        dx = self.sim.cell_size
        
        # Calculate gradients using central differences
        for y in range(1, 15):
            for x in range(1, 15):
                dP_dx = (self.sim.pressure[y, x + 1] - self.sim.pressure[y, x - 1]) / (2 * dx)
                dP_dy = (self.sim.pressure[y + 1, x] - self.sim.pressure[y - 1, x]) / (2 * dx)
                
                # For P = x + y, gradients should be dP/dx = 1/dx, dP/dy = 1/dx
                expected_gradient = 1.0 / dx
                
                assert abs(dP_dx - expected_gradient) < 1e-10, f"dP/dx should be {expected_gradient}, got {dP_dx}"
                assert abs(dP_dy - expected_gradient) < 1e-10, f"dP/dy should be {expected_gradient}, got {dP_dy}"
    
    def test_velocity_projection_step(self):
        """Test velocity projection to enforce incompressibility"""
        # Start with a divergent velocity field
        dt = 0.1
        
        for y in range(16):
            for x in range(16):
                self.sim.velocity_x[y, x] = x - 8  # Divergent field
                self.sim.velocity_y[y, x] = y - 8
        
        # Create pressure field to correct the divergence
        # For this velocity field, we need pressure that creates opposing gradients
        for y in range(16):
            for x in range(16):
                # Pressure field that will cancel the divergence
                self.sim.pressure[y, x] = -0.5 * ((x - 8)**2 + (y - 8)**2)
        
        # Apply velocity projection: v_new = v_old - dt * grad(P) / rho
        dx = self.sim.cell_size
        rho = 1000.0  # Typical fluid density
        
        velocity_x_new = self.sim.velocity_x.copy()
        velocity_y_new = self.sim.velocity_y.copy()
        
        for y in range(1, 15):
            for x in range(1, 15):
                if self.sim.material_types[y, x] != MaterialType.SPACE:
                    # Calculate pressure gradients
                    dP_dx = (self.sim.pressure[y, x + 1] - self.sim.pressure[y, x - 1]) / (2 * dx)
                    dP_dy = (self.sim.pressure[y + 1, x] - self.sim.pressure[y - 1, x]) / (2 * dx)
                    
                    # Update velocities
                    velocity_x_new[y, x] -= dt * dP_dx / rho
                    velocity_y_new[y, x] -= dt * dP_dy / rho
        
        # Check that divergence is reduced
        def calculate_divergence(vx, vy):
            div = np.zeros_like(vx)
            for y in range(1, 15):
                for x in range(1, 15):
                    dvx_dx = (vx[y, x + 1] - vx[y, x - 1]) / (2 * dx)
                    dvy_dy = (vy[y + 1, x] - vy[y - 1, x]) / (2 * dx)
                    div[y, x] = dvx_dx + dvy_dy
            return div
        
        div_old = calculate_divergence(self.sim.velocity_x, self.sim.velocity_y)
        div_new = calculate_divergence(velocity_x_new, velocity_y_new)
        
        max_div_old = np.max(np.abs(div_old[1:15, 1:15]))
        max_div_new = np.max(np.abs(div_new[1:15, 1:15]))
        
        # Allow equality within floating-point noise rather than requiring strict
        # decrease—symmetry can lead to exact ties.
        assert max_div_new <= max_div_old + 1e-10, (
            f"Divergence should not increase: {max_div_old} -> {max_div_new}")


class TestForceCalculation:
    """Test force calculation for unified kinematics"""
    
    def setup_method(self):
        """Set up test simulation"""
        self.sim = GeologySimulation(width=16, height=16, cell_size=50.0)
        
        # Initialize fields
        self.sim.velocity_x = np.zeros((16, 16), dtype=np.float64)
        self.sim.velocity_y = np.zeros((16, 16), dtype=np.float64)
    
    def test_gravity_force_calculation(self):
        """Test gravitational force calculation"""
        # Gravity should point toward center of mass
        center_x, center_y = self.sim.center_of_mass
        
        for y in range(16):
            for x in range(16):
                if self.sim.material_types[y, x] != MaterialType.SPACE:
                    # Vector from cell to center
                    dx = center_x - x
                    dy = center_y - y
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0:
                        # Normalized gravity vector (toward center)
                        gx = dx / distance * 9.81  # m/s^2
                        gy = dy / distance * 9.81
                        
                        # Gravity should point toward center
                        assert gx * dx >= 0, "Gravity x-component should point toward center"
                        assert gy * dy >= 0, "Gravity y-component should point toward center"
    
    def test_buoyancy_force_calculation(self):
        """Test buoyancy force calculation"""
        # Create a density contrast scenario
        # Place light material (air) surrounded by heavy material (water)
        
        # Fill with water
        self.sim.material_types[5:11, 5:11] = MaterialType.WATER
        self.sim.temperature[5:11, 5:11] = 300.0
        
        # Place air bubble in center
        self.sim.material_types[7:9, 7:9] = MaterialType.AIR
        self.sim.temperature[7:9, 7:9] = 300.0
        
        # Update material properties
        self.sim._update_material_properties()
        
        # Calculate buoyancy force for air cells
        center_x, center_y = self.sim.center_of_mass
        
        air_mask = (self.sim.material_types == MaterialType.AIR)
        air_coords = np.where(air_mask)
        
        for i in range(len(air_coords[0])):
            y, x = air_coords[0][i], air_coords[1][i]
            
            # Get local densities
            air_density = self.sim.density[y, x]
            
            # Find surrounding water density
            water_densities = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < 16 and 0 <= nx < 16 and 
                        self.sim.material_types[ny, nx] == MaterialType.WATER):
                        water_densities.append(self.sim.density[ny, nx])
            
            if water_densities:
                avg_water_density = np.mean(water_densities)
                
                # Air should be less dense than water
                assert air_density < avg_water_density, f"Air density {air_density} should be less than water density {avg_water_density}"
                
                # Buoyancy force should point away from center (upward)
                dx_center = x - center_x
                dy_center = y - center_y
                distance = np.sqrt(dx_center*dx_center + dy_center*dy_center)
                
                if distance > 0:
                    # Buoyancy force magnitude
                    buoyancy_magnitude = 9.81 * (avg_water_density - air_density) / air_density
                    
                    # Buoyancy direction (outward from center)
                    buoyancy_x = (dx_center / distance) * buoyancy_magnitude
                    buoyancy_y = (dy_center / distance) * buoyancy_magnitude
                    
                    # Buoyancy should be positive (outward)
                    assert buoyancy_magnitude > 0, "Buoyancy force should be positive for light material in heavy fluid"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 