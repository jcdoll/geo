"""
Test suite for SPH backend implementations.

Tests CPU, Numba, and GPU backends for correctness and consistency.
"""

import numpy as np
import pytest
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel


class TestBackends:
    """Test all backend implementations for consistency."""
    
    @pytest.fixture(params=['cpu', 'numba', 'gpu'])
    def backend(self, request):
        """Parametrize tests over all available backends."""
        backend_name = request.param
        
        # Check if backend is available
        original_backend = sph.get_backend()
        try:
            sph.set_backend(backend_name)
            yield backend_name
        except Exception as e:
            pytest.skip(f"Backend {backend_name} not available: {e}")
        finally:
            # Restore original backend
            sph.set_backend(original_backend)
    
    @pytest.fixture
    def particles(self):
        """Create test particle system."""
        n_particles = 100
        particles = ParticleArrays.allocate(n_particles)
        
        # Initialize particles in a grid
        grid_size = int(np.sqrt(n_particles))
        spacing = 0.1
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx < n_particles:
                    particles.position_x[idx] = i * spacing
                    particles.position_y[idx] = j * spacing
                    particles.velocity_x[idx] = 0.0
                    particles.velocity_y[idx] = 0.0
                    particles.mass[idx] = 1.0
                    particles.smoothing_h[idx] = 0.15
                    idx += 1
        
        return particles, n_particles
    
    def test_density_computation(self, backend, particles):
        """Test density computation across backends."""
        particles_data, n_active = particles
        kernel = CubicSplineKernel()
        
        # Compute density
        sph.compute_density(particles_data, kernel, n_active)
        
        # Check results
        assert np.all(particles_data.density[:n_active] > 0)
        assert np.all(np.isfinite(particles_data.density[:n_active]))
        
        # Density should be roughly uniform for uniform grid
        mean_density = np.mean(particles_data.density[:n_active])
        std_density = np.std(particles_data.density[:n_active])
        assert std_density / mean_density < 0.5  # Less than 50% variation
    
    def test_pressure_computation(self, backend, particles):
        """Test pressure computation."""
        particles_data, n_active = particles
        
        # Set uniform density
        particles_data.density[:n_active] = 1000.0
        
        # Compute pressure
        sph.compute_pressure(particles_data, n_active)
        
        # Check results
        assert np.all(np.isfinite(particles_data.pressure[:n_active]))
        
        # For rest density, pressure should be near zero
        assert np.all(np.abs(particles_data.pressure[:n_active]) < 1.0)
    
    def test_forces_computation(self, backend, particles):
        """Test force computation."""
        particles_data, n_active = particles
        kernel = CubicSplineKernel()
        
        # Set some density and pressure
        particles_data.density[:n_active] = 1000.0
        particles_data.pressure[:n_active] = 100.0
        
        # Compute forces
        gravity = np.array([0.0, -9.81])
        sph.compute_forces(particles_data, kernel, n_active, gravity=gravity)
        
        # Check results
        assert np.all(np.isfinite(particles_data.force_x[:n_active]))
        assert np.all(np.isfinite(particles_data.force_y[:n_active]))
        
        # Y-forces should include gravity
        expected_fy = particles_data.mass[:n_active] * gravity[1]
        assert np.all(particles_data.force_y[:n_active] <= expected_fy)
    
    def test_integration(self, backend, particles):
        """Test integration step."""
        particles_data, n_active = particles
        
        # Set initial conditions
        initial_x = particles_data.position_x[:n_active].copy()
        initial_y = particles_data.position_y[:n_active].copy()
        
        # Set forces
        particles_data.force_x[:n_active] = 10.0  # Constant force
        particles_data.force_y[:n_active] = 0.0
        
        # Integrate
        dt = 0.01
        sph.integrate(particles_data, n_active, dt=dt)
        
        # Check that particles moved
        assert np.all(particles_data.position_x[:n_active] > initial_x)
        assert np.allclose(particles_data.position_y[:n_active], initial_y, atol=1e-10)
        
        # Check velocity updated
        expected_vx = particles_data.force_x[:n_active] / particles_data.mass[:n_active] * dt
        assert np.allclose(particles_data.velocity_x[:n_active], expected_vx, rtol=1e-5)
    
    def test_full_timestep(self, backend, particles):
        """Test a complete SPH timestep."""
        particles_data, n_active = particles
        kernel = CubicSplineKernel()
        
        # Store initial energy
        initial_ke = 0.5 * np.sum(
            particles_data.mass[:n_active] * (
                particles_data.velocity_x[:n_active]**2 + 
                particles_data.velocity_y[:n_active]**2
            )
        )
        
        # Run one timestep
        sph.compute_density(particles_data, kernel, n_active)
        sph.compute_pressure(particles_data, n_active)
        sph.compute_forces(particles_data, kernel, n_active)
        sph.integrate(particles_data, n_active, dt=0.001)
        
        # Check all results are finite
        assert np.all(np.isfinite(particles_data.density[:n_active]))
        assert np.all(np.isfinite(particles_data.pressure[:n_active]))
        assert np.all(np.isfinite(particles_data.force_x[:n_active]))
        assert np.all(np.isfinite(particles_data.force_y[:n_active]))
        assert np.all(np.isfinite(particles_data.position_x[:n_active]))
        assert np.all(np.isfinite(particles_data.position_y[:n_active]))


class TestBackendConsistency:
    """Test that all backends produce consistent results."""
    
    def setup_particles(self, n=50):
        """Create identical particle systems."""
        particles = ParticleArrays.allocate(n)
        
        # Random but reproducible positions
        np.random.seed(12345)
        particles.position_x[:n] = np.random.uniform(0, 1, n)
        particles.position_y[:n] = np.random.uniform(0, 1, n)
        particles.velocity_x[:n] = np.random.uniform(-0.1, 0.1, n)
        particles.velocity_y[:n] = np.random.uniform(-0.1, 0.1, n)
        particles.mass[:n] = 1.0
        particles.smoothing_h[:n] = 0.1
        
        return particles, n
    
    def get_available_backends(self):
        """Get list of available backends."""
        backends = ['cpu']  # CPU always available
        
        original = sph.get_backend()
        
        # Check Numba
        try:
            sph.set_backend('numba')
            backends.append('numba')
        except:
            pass
        
        # Check GPU
        try:
            sph.set_backend('gpu')
            backends.append('gpu')
        except:
            pass
        
        sph.set_backend(original)
        return backends
    
    def test_density_consistency(self):
        """Test that all backends compute same density."""
        backends = self.get_available_backends()
        if len(backends) < 2:
            pytest.skip("Need at least 2 backends for consistency test")
        
        results = {}
        kernel = CubicSplineKernel()
        
        for backend in backends:
            sph.set_backend(backend)
            particles, n = self.setup_particles()
            sph.compute_density(particles, kernel, n)
            results[backend] = particles.density[:n].copy()
        
        # Compare all backends to CPU
        cpu_density = results['cpu']
        for backend, density in results.items():
            if backend != 'cpu':
                # Allow small numerical differences
                np.testing.assert_allclose(
                    density, cpu_density, 
                    rtol=1e-5, atol=1e-8,
                    err_msg=f"{backend} density differs from CPU"
                )
    
    def test_forces_consistency(self):
        """Test that all backends compute same forces."""
        backends = self.get_available_backends()
        if len(backends) < 2:
            pytest.skip("Need at least 2 backends for consistency test")
        
        results = {}
        kernel = CubicSplineKernel()
        
        for backend in backends:
            sph.set_backend(backend)
            particles, n = self.setup_particles()
            
            # Set same density and pressure
            particles.density[:n] = 1000.0
            particles.pressure[:n] = 1000.0
            
            sph.compute_forces(particles, kernel, n)
            results[backend] = (
                particles.force_x[:n].copy(),
                particles.force_y[:n].copy()
            )
        
        # Compare all backends to CPU
        cpu_fx, cpu_fy = results['cpu']
        for backend, (fx, fy) in results.items():
            if backend != 'cpu':
                np.testing.assert_allclose(
                    fx, cpu_fx, 
                    rtol=1e-4, atol=1e-6,
                    err_msg=f"{backend} force_x differs from CPU"
                )
                np.testing.assert_allclose(
                    fy, cpu_fy, 
                    rtol=1e-4, atol=1e-6,
                    err_msg=f"{backend} force_y differs from CPU"
                )


if __name__ == "__main__":
    # Run basic tests
    test = TestBackends()
    
    print("Testing available backends...")
    for backend_name in ['cpu', 'numba', 'gpu']:
        try:
            sph.set_backend(backend_name)
            print(f"\n{backend_name.upper()} backend:")
            
            # Create test particles
            particles = ParticleArrays.allocate(100)
            particles.position_x[:] = np.random.uniform(0, 1, 100)
            particles.position_y[:] = np.random.uniform(0, 1, 100)
            particles.mass[:] = 1.0
            particles.smoothing_h[:] = 0.1
            
            # Test density
            kernel = CubicSplineKernel()
            sph.compute_density(particles, kernel, 100)
            print(f"  Density: min={particles.density.min():.2f}, max={particles.density.max():.2f}")
            
            # Test pressure
            sph.compute_pressure(particles, 100)
            print(f"  Pressure: min={particles.pressure.min():.2f}, max={particles.pressure.max():.2f}")
            
            # Test forces
            sph.compute_forces(particles, kernel, 100)
            print(f"  Forces: |F|_avg={np.sqrt(particles.force_x**2 + particles.force_y**2).mean():.2f}")
            
        except Exception as e:
            print(f"\n{backend_name.upper()} backend: NOT AVAILABLE ({e})")