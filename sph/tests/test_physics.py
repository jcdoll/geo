"""
Physics validation tests for SPH implementation.

Tests physical correctness including:
- Hydrostatic equilibrium
- Conservation laws
- Stability
"""

import numpy as np
import pytest
import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.spatial_hash_vectorized import VectorizedSpatialHash, find_neighbors_vectorized


class TestHydrostatics:
    """Test hydrostatic equilibrium."""
    
    def create_water_column(self, height=1.0, n_layers=20):
        """Create a column of water particles."""
        particles_per_layer = 10
        n_particles = n_layers * particles_per_layer
        particles = ParticleArrays.allocate(n_particles)
        
        spacing = height / n_layers
        particle_spacing = spacing * 0.8
        
        idx = 0
        for layer in range(n_layers):
            y = layer * spacing + spacing/2
            for i in range(particles_per_layer):
                x = i * particle_spacing + particle_spacing/2
                particles.position_x[idx] = x
                particles.position_y[idx] = y
                particles.velocity_x[idx] = 0.0
                particles.velocity_y[idx] = 0.0
                particles.mass[idx] = 1.0
                particles.smoothing_h[idx] = spacing * 1.5
                particles.material_id[idx] = 1  # Water
                idx += 1
        
        return particles, n_particles
    
    @pytest.mark.parametrize("backend", ['cpu', 'numba', 'gpu'])
    def test_hydrostatic_equilibrium(self, backend):
        """Test that water column reaches hydrostatic equilibrium."""
        try:
            sph.set_backend(backend)
        except:
            pytest.skip(f"Backend {backend} not available")
        
        particles, n = self.create_water_column()
        kernel = CubicSplineKernel()
        
        # Run simulation
        dt = 0.0001
        steps = 1000
        damping = 0.99  # Small damping for stability
        
        for step in range(steps):
            sph.compute_density(particles, kernel, n)
            sph.compute_pressure(particles, n, rest_density=1000.0)
            sph.compute_forces(particles, kernel, n, gravity=np.array([0, -9.81]))
            
            # Apply damping
            particles.velocity_x[:n] *= damping
            particles.velocity_y[:n] *= damping
            
            sph.integrate(particles, n, dt=dt)
        
        # Check equilibrium
        # 1. Velocities should be near zero
        max_vel = np.max(np.abs(particles.velocity_y[:n]))
        assert max_vel < 0.1, f"Velocities not settled: max_vel={max_vel}"
        
        # 2. Pressure should increase with depth
        bottom_particles = particles.position_y[:n] < 0.2
        top_particles = particles.position_y[:n] > 0.8
        
        bottom_pressure = np.mean(particles.pressure[bottom_particles])
        top_pressure = np.mean(particles.pressure[top_particles])
        
        assert bottom_pressure > top_pressure, "Pressure should increase with depth"
    
    def test_pressure_gradient(self):
        """Test pressure gradient in water column."""
        particles, n = self.create_water_column(height=2.0, n_layers=40)
        kernel = CubicSplineKernel()
        
        # Use CPU backend for reference
        sph.set_backend('cpu')
        
        # Let system settle
        dt = 0.0001
        for _ in range(500):
            sph.compute_density(particles, kernel, n)
            sph.compute_pressure(particles, n, rest_density=1000.0)
            sph.compute_forces(particles, kernel, n, gravity=np.array([0, -9.81]))
            particles.velocity_x[:n] *= 0.98
            particles.velocity_y[:n] *= 0.98
            sph.integrate(particles, n, dt=dt)
        
        # Measure pressure gradient
        heights = particles.position_y[:n]
        pressures = particles.pressure[:n]
        
        # Fit linear relationship (pressure vs depth)
        depth = 2.0 - heights
        coeffs = np.polyfit(depth, pressures, 1)
        
        # Theoretical gradient: dP/dz = -rho * g
        theoretical_gradient = 1000.0 * 9.81
        measured_gradient = coeffs[0]
        
        # Should be within 20% of theoretical
        error = abs(measured_gradient - theoretical_gradient) / theoretical_gradient
        assert error < 0.2, f"Pressure gradient error: {error*100:.1f}%"


class TestConservation:
    """Test conservation laws."""
    
    def create_isolated_system(self, n=100):
        """Create an isolated system of particles."""
        particles = ParticleArrays.allocate(n)
        
        # Random positions in a box
        particles.position_x[:n] = np.random.uniform(0, 5, n)
        particles.position_y[:n] = np.random.uniform(0, 5, n)
        
        # Random velocities (but zero mean)
        particles.velocity_x[:n] = np.random.uniform(-1, 1, n)
        particles.velocity_y[:n] = np.random.uniform(-1, 1, n)
        particles.velocity_x[:n] -= np.mean(particles.velocity_x[:n])
        particles.velocity_y[:n] -= np.mean(particles.velocity_y[:n])
        
        particles.mass[:n] = 1.0
        particles.smoothing_h[:n] = 0.3
        
        return particles, n
    
    @pytest.mark.parametrize("backend", ['cpu', 'numba', 'gpu'])
    def test_momentum_conservation(self, backend):
        """Test momentum conservation in isolated system."""
        try:
            sph.set_backend(backend)
        except:
            pytest.skip(f"Backend {backend} not available")
        
        particles, n = self.create_isolated_system()
        kernel = CubicSplineKernel()
        
        # Initial momentum
        initial_px = np.sum(particles.mass[:n] * particles.velocity_x[:n])
        initial_py = np.sum(particles.mass[:n] * particles.velocity_y[:n])
        
        # Run simulation (no external forces)
        dt = 0.001
        for _ in range(100):
            sph.compute_density(particles, kernel, n)
            sph.compute_pressure(particles, n)
            sph.compute_forces(particles, kernel, n, gravity=np.array([0, 0]))
            sph.integrate(particles, n, dt=dt)
        
        # Final momentum
        final_px = np.sum(particles.mass[:n] * particles.velocity_x[:n])
        final_py = np.sum(particles.mass[:n] * particles.velocity_y[:n])
        
        # Should be conserved to numerical precision
        assert abs(final_px - initial_px) < 1e-10
        assert abs(final_py - initial_py) < 1e-10
    
    def test_energy_conservation(self):
        """Test energy behavior in damped system."""
        particles, n = self.create_isolated_system(n=50)
        kernel = CubicSplineKernel()
        
        sph.set_backend('cpu')  # Use CPU for consistency
        
        energies = []
        dt = 0.001
        
        for _ in range(50):
            # Compute kinetic energy
            ke = 0.5 * np.sum(particles.mass[:n] * (
                particles.velocity_x[:n]**2 + particles.velocity_y[:n]**2
            ))
            energies.append(ke)
            
            # Update
            sph.compute_density(particles, kernel, n)
            sph.compute_pressure(particles, n)
            sph.compute_forces(particles, kernel, n, gravity=np.array([0, 0]))
            sph.integrate(particles, n, dt=dt, damping=0.01)
        
        # Energy should decrease monotonically with damping
        energy_diffs = np.diff(energies)
        assert np.all(energy_diffs <= 0), "Energy should decrease with damping"


class TestStability:
    """Test numerical stability."""
    
    @pytest.mark.parametrize("backend", ['cpu', 'numba', 'gpu'])
    def test_particle_at_rest(self, backend):
        """Test that particles at rest remain stable."""
        try:
            sph.set_backend(backend)
        except:
            pytest.skip(f"Backend {backend} not available")
        
        # Single particle
        particles = ParticleArrays.allocate(1)
        particles.position_x[0] = 0.0
        particles.position_y[0] = 0.0
        particles.velocity_x[0] = 0.0
        particles.velocity_y[0] = 0.0
        particles.mass[0] = 1.0
        particles.smoothing_h[0] = 0.1
        
        kernel = CubicSplineKernel()
        
        # Run many steps
        dt = 0.001
        for _ in range(1000):
            sph.compute_density(particles, kernel, 1)
            sph.compute_pressure(particles, 1)
            sph.compute_forces(particles, kernel, 1, gravity=np.array([0, 0]))
            sph.integrate(particles, 1, dt=dt)
        
        # Should remain at origin
        assert abs(particles.position_x[0]) < 1e-10
        assert abs(particles.position_y[0]) < 1e-10
        assert abs(particles.velocity_x[0]) < 1e-10
        assert abs(particles.velocity_y[0]) < 1e-10
    
    def test_large_timestep_stability(self):
        """Test stability with larger timesteps."""
        particles, n = TestConservation().create_isolated_system(n=25)
        kernel = CubicSplineKernel()
        
        sph.set_backend('cpu')
        
        # Try progressively larger timesteps
        for dt in [0.001, 0.005, 0.01]:
            # Reset velocities
            particles.velocity_x[:n] = 0.0
            particles.velocity_y[:n] = 0.0
            
            stable = True
            for _ in range(100):
                sph.compute_density(particles, kernel, n)
                sph.compute_pressure(particles, n)
                sph.compute_forces(particles, kernel, n)
                sph.integrate(particles, n, dt=dt, damping=0.1)
                
                # Check for NaN or infinity
                if not np.all(np.isfinite(particles.position_x[:n])):
                    stable = False
                    break
                
                # Check for explosion
                if np.any(np.abs(particles.velocity_x[:n]) > 100):
                    stable = False
                    break
            
            if dt <= 0.005:
                assert stable, f"System unstable with dt={dt}"


if __name__ == "__main__":
    # Run basic validation
    print("Running SPH physics validation tests...")
    
    # Test hydrostatics
    print("\n1. Testing hydrostatic equilibrium...")
    test = TestHydrostatics()
    for backend in ['cpu', 'numba', 'gpu']:
        try:
            test.test_hydrostatic_equilibrium(backend)
            print(f"   {backend}: ✓")
        except Exception as e:
            print(f"   {backend}: ✗ ({e})")
    
    # Test conservation
    print("\n2. Testing momentum conservation...")
    test = TestConservation()
    for backend in ['cpu', 'numba', 'gpu']:
        try:
            test.test_momentum_conservation(backend)
            print(f"   {backend}: ✓")
        except Exception as e:
            print(f"   {backend}: ✗ ({e})")
    
    # Test stability
    print("\n3. Testing numerical stability...")
    test = TestStability()
    test.test_large_timestep_stability()
    print("   Timestep stability: ✓")
    
    print("\nAll physics tests completed!")