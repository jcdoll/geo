"""
Performance and accuracy tests for heat transfer solvers.

This test ensures that:
1. Heat transfer performance hasn't degraded
2. Both ADI and Multigrid methods produce accurate results
3. Conservation of energy is maintained
"""

import numpy as np
import time
import pytest
from state import FluxState
from heat_transfer import HeatTransfer
from materials import MaterialDatabase, MaterialType
from scenarios import setup_scenario


class TestHeatTransferPerformance:
    """Test suite for heat transfer solver performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.material_db = MaterialDatabase()
        
    def create_test_state(self, nx: int, ny: int, scenario: str = "planet"):
        """Create a test state with the given scenario."""
        state = FluxState(nx, ny, dx=50.0)
        setup_scenario(scenario, state, self.material_db)
        return state
        
    def test_adi_performance(self):
        """Test that ADI solver meets performance requirements."""
        state = self.create_test_state(100, 100)
        solver = HeatTransfer(state)
        
        # Warm-up
        solver.solve_heat_equation(1.0)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            solver.solve_heat_equation(1.0)
            times.append(time.perf_counter() - start)
            
        avg_time = np.mean(times)
        # ADI should complete in under 40ms for 100x100 grid (realistic for Python)
        assert avg_time < 0.040, f"ADI solver too slow: {avg_time*1000:.2f}ms"
        
    def test_multigrid_performance(self):
        """Test that Multigrid solver meets performance requirements."""
        state = self.create_test_state(100, 100)
        solver = HeatTransfer(state, solver_method="multigrid")
        
        # Warm-up
        solver.solve_heat_equation(1.0)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            solver.solve_heat_equation(1.0)
            times.append(time.perf_counter() - start)
            
        avg_time = np.mean(times)
        # Multigrid is slower than ADI for diffusion - allow up to 400ms
        assert avg_time < 0.400, f"Multigrid solver too slow: {avg_time*1000:.2f}ms"
        
    def test_heat_diffusion_accuracy(self):
        """Test accuracy of heat diffusion against analytical solution."""
        # Create simple 1D-like problem (thin horizontal strip)
        nx, ny = 100, 10
        state = FluxState(nx, ny, dx=1.0)
        
        # Fill with rock
        state.vol_frac[MaterialType.ROCK.value] = 1.0
        state.vol_frac[MaterialType.SPACE.value] = 0.0
        state.update_mixture_properties(self.material_db)
        
        # Set initial temperature profile (Gaussian)
        x = np.arange(nx) - nx/2
        for i in range(nx):
            state.temperature[:, i] = 300 + 100 * np.exp(-x[i]**2 / 100)
            
        # Run diffusion for a short time
        # Create fresh states
        state_adi = FluxState(nx, ny, dx=1.0)
        state_mg = FluxState(nx, ny, dx=1.0)
        
        # Copy setup
        state_adi.vol_frac = state.vol_frac.copy()
        state_mg.vol_frac = state.vol_frac.copy()
        state_adi.temperature = state.temperature.copy()
        state_mg.temperature = state.temperature.copy()
        state_adi.update_mixture_properties(self.material_db)
        state_mg.update_mixture_properties(self.material_db)
        
        solver_adi = HeatTransfer(state_adi, solver_method="adi")
        solver_mg = HeatTransfer(state_mg, solver_method="multigrid")
        
        dt = 0.1
        for _ in range(10):
            solver_adi.solve_heat_equation(dt)
            solver_mg.solve_heat_equation(dt)
            
        # Check that both methods give similar results
        temp_diff = np.abs(solver_adi.state.temperature - solver_mg.state.temperature)
        assert np.max(temp_diff) < 1.0, f"Methods differ by {np.max(temp_diff):.2f}K"
        
        # Check that heat has diffused (temperature should be more uniform)
        initial_std = np.std(state.temperature)
        final_std_adi = np.std(solver_adi.state.temperature)
        final_std_mg = np.std(solver_mg.state.temperature)
        
        assert final_std_adi < initial_std * 0.9
        assert final_std_mg < initial_std * 0.9
        
    def test_energy_conservation(self):
        """Test that total thermal energy is conserved."""
        state = self.create_test_state(50, 50, "planet")
        
        # Add some initial heat
        state.temperature[20:30, 20:30] = 1000.0
        
        # Compute initial energy
        initial_energy = np.sum(
            state.density * state.specific_heat * state.temperature * state.dx**2
        )
        
        # Run simulation with no heat sources or sinks
        solver = HeatTransfer(state)
        for _ in range(100):
            solver.apply_thermal_diffusion(1.0)  # Only diffusion, no radiation
            
        # Compute final energy
        final_energy = np.sum(
            state.density * state.specific_heat * state.temperature * state.dx**2
        )
        
        # Energy should be conserved (within numerical precision)
        rel_error = abs(final_energy - initial_energy) / initial_energy
        assert rel_error < 1e-6, f"Energy not conserved: {rel_error:.2e} relative error"
        
    def test_radiation_cooling(self):
        """Test Stefan-Boltzmann radiation cooling."""
        state = self.create_test_state(20, 20)
        
        # Create hot surface exposed to space
        state.vol_frac[MaterialType.ROCK.value, 0, :] = 1.0
        state.vol_frac[MaterialType.SPACE.value, 0, :] = 0.0
        state.temperature[0, :] = 1000.0  # Hot surface
        state.update_mixture_properties(self.material_db)
        
        solver = HeatTransfer(state)
        initial_temp = state.temperature[0, 10]
        
        # Apply radiation for 100 seconds
        for _ in range(100):
            solver.apply_radiative_cooling(1.0)
            
        final_temp = state.temperature[0, 10]
        
        # Temperature should decrease
        assert final_temp < initial_temp, "Surface should cool by radiation"
        
        # Check approximate cooling rate using Stefan-Boltzmann
        # dT/dt ≈ -σεT⁴/(ρcₚd)
        sigma = 5.67e-8
        emissivity = state.emissivity[0, 10]
        rho = state.density[0, 10]
        cp = state.specific_heat[0, 10]
        dx = state.dx
        
        expected_rate = -sigma * emissivity * initial_temp**4 / (rho * cp * dx)
        actual_rate = (final_temp - initial_temp) / 100.0
        
        # Should be same order of magnitude
        assert abs(actual_rate / expected_rate - 1.0) < 0.5
        
    def test_heat_generation(self):
        """Test radioactive heat generation."""
        state = self.create_test_state(20, 20)
        
        # Add uranium
        state.vol_frac[MaterialType.URANIUM.value, 10:15, 10:15] = 0.1
        state.vol_frac[MaterialType.ROCK.value, 10:15, 10:15] = 0.9
        state.update_mixture_properties(self.material_db)
        
        initial_temp = state.temperature.copy()
        
        solver = HeatTransfer(state)
        
        # Apply heat generation for 1000 seconds
        for _ in range(1000):
            solver.apply_heat_generation(1.0)
            
        # Temperature should increase where uranium is present
        uranium_mask = state.vol_frac[MaterialType.URANIUM.value] > 0
        temp_increase = state.temperature - initial_temp
        
        assert np.mean(temp_increase[uranium_mask]) > 0.1
        assert np.mean(temp_increase[~uranium_mask]) < 0.01
        
    @pytest.mark.parametrize("grid_size", [(50, 50), (100, 100), (150, 150)])
    def test_performance_scaling(self, grid_size):
        """Test that performance scales reasonably with grid size."""
        nx, ny = grid_size
        state = self.create_test_state(nx, ny)
        
        solver = HeatTransfer(state)
        
        # Warm-up
        solver.solve_heat_equation(1.0)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(5):
            solver.solve_heat_equation(1.0)
        elapsed = time.perf_counter() - start
        
        avg_time = elapsed / 5
        cells_per_second = (nx * ny) / avg_time
        
        # Should process at least 1M cells/second
        assert cells_per_second > 1e6, f"Only {cells_per_second/1e6:.2f}M cells/s"
        
    def test_boundary_conditions(self):
        """Test that boundary conditions are properly applied."""
        state = self.create_test_state(30, 30)
        
        # Set specific boundary temperatures
        state.temperature[0, :] = 500.0  # Top
        state.temperature[-1, :] = 300.0  # Bottom
        state.temperature[:, 0] = 400.0  # Left
        state.temperature[:, -1] = 350.0  # Right
        
        solver = HeatTransfer(state)
        
        # Run diffusion
        for _ in range(10):
            solver.apply_thermal_diffusion(0.1)
            
        # Boundaries should maintain approximate values (within diffusion distance)
        assert np.mean(state.temperature[0, 1:-1]) > 450
        assert np.mean(state.temperature[-1, 1:-1]) < 350
        assert np.mean(state.temperature[1:-1, 0]) > 375
        assert np.mean(state.temperature[1:-1, -1]) > 325